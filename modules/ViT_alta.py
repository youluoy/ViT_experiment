
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math


class PatchSelfConv(nn.Module):
    def __init__(self, in_channels:int=3, num_patch_row:int=4, image_size:int=32, ffn_dim:int=384*4, stride:int=1, padding:int=0, dropout:float=0.):
        """ 
        引数:
            in_channels: 入力画像のチャンネル数
            num_patch_row: 高さ方向のパッチの数。例は2x2であるため、2をデフォルト値とした 
            image_size: 入力画像の1辺の大きさ。入力画像の高さと幅は同じであると仮定
            stride: 
            padding: 
        """
        super(PatchSelfConv, self).__init__()
        self.in_channels = in_channels
        self.num_patch_row = num_patch_row
        self.image_size = image_size
        self.stride = stride
        self.padding = padding
        #画像のRGBチャンネル一枚に対するパッチの総数
        self.num_patch = self.num_patch_row**2
        #パッチの一辺の長さ
        self.patch_size = int(self.image_size // self.num_patch_row)
        #
        self.image_pix = image_size**2
        #
        self.patch_pix = self.patch_size**2
        #
        self.hid_image_size = math.ceil((image_size + padding*2 - self.patch_size + 1) / stride)
        #
        self.hid_image_pix = self.hid_image_size**2

        #
        self.patch_split_layer = nn.Conv2d(
            in_channels = self.in_channels,
            out_channels = self.in_channels,
            kernel_size = self.patch_size,
            stride = self.patch_size
        )

        # Pointwise conv用のパラメータ　要素積計算時はバッチ方向にブロードキャストを行う
        self.ptwise_param = nn.Parameter(
            torch.randn(1, in_channels, image_size)
        )

        #
        self.ffn_layer = []
        for i in range(self.in_channels):
            self.ffn_layer.append(
                nn.Sequential(
                    nn.Linear(self.hid_image_pix, ffn_dim), 
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(ffn_dim, self.patch_pix), 
                    nn.Dropout(dropout)
                )
            )
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        引数:
            x: 入力画像。形状は、(B, C, H, W)。[式(1)]
                B: バッチサイズ、C:チャンネル数、H:高さ、W:幅
        返り値:
            z_0: ViTへの入力。形状は、(B, C, H, W)。
                B: バッチサイズ、C:チャンネル数、H:高さ、W:幅
        """



        # 入力テンソルを複製
        ## (B, C, H, W)
        #z_0 = x
        # pointwise conv でチャンネルを１に落とす
        ## (B, C, H, W) -> (B, 1, H, W)
        #z_0 = z_0 * self.ptwise_param
        #z_0 = torch.sum(z_0, 1)
        # パッチに分割し、パッチ数分のフィルタを作成する
        ## (B, 1, H, W) -> (B, P, H/P, W/P)
        #z_0 = z_0.unfold(2, kernel_size=self.num_patch_row, stride=self.num_patch_row).unfold(3, kernel_size=self.num_patch_row, stride=self.num_patch_row)
        # 作ったパッチと画像との畳み込み演算を行う　出力はパッチ数×３枚の画像
        #for i in range(self.num_patch):
        #    
        # 全結合を用いてもとの画像サイズに圧縮
        #z_0 = torch.Tensor()
        #for i in range(self.num_patch):
        #    z_0[i] = F.conv2d(x, x[:,:,i*self.num_patch_row:(i+1)*self.num_patch_row, i*self.num_patch_row:(i+1)*self.num_patch_row])

        batch_size = x.size()[0]
        # 入力テンソルを複製 detach():値共有、伝播オフ clone():値非共有、伝播cloneもとと同じ
        ## (B, C, H, W)
        z_0 = x.detach().clone()
        # パッチに分割し、パッチ数分のフィルタを作成する
        ## (B, C, H, W) -> (B, C, P, patch_size, patch_size)
        z_0 = z_0.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        z_0 = z_0.reshape(batch_size, self.in_channels, self.num_patch, self.patch_size, self.patch_size)
        #並べ替えしてF.conv2d用の形にする
        ## (B, C, P, patch_size, patch_size) -> (B, P, C, patch_size, patch_size)
        z_0 = z_0.permute(0, 2, 1, 3, 4)
        # 作ったパッチと画像との畳み込み演算を行う F.covn2d outputチャンネル方向がパッチの枚数に対応する
        ## -> (B, P, hid_image_size, hid_image_size)
        hid_feature = torch.zeros(batch_size, self.num_patch, self.hid_image_size, self.hid_image_size)
        for i in range(batch_size):
            hid_feature[i] = F.conv2d(x[i].unsqueeze(dim=0), z_0[i])
        # flatten
        ## -> (B, P, hid_image_pix)
        hid_feature = torch.flatten(hid_feature, start_dim=2)
        # 全結合を用いてもとのパッチサイズに戻す　チャンネルが増える
        ## (B, P, hid_image_pix) -> (B, C, P, patch_pix)
        out_patch = torch.zeros(batch_size, self.in_channels, self.num_patch, self.patch_pix)
        for i in range(self.in_channels):
            out_patch[:,i,:,:] = self.ffn_layer[i](hid_feature)
        # reshape 
        ## (B, C, P, patch_pix) -> (B, C, H, W)
        out_patch = out_patch.reshape(batch_size, self.in_channels, self.num_patch, self.patch_size, self.patch_size)
        out_patch = out_patch.permute(0, 2, 1, 3, 4)
        out = torch.zeros(batch_size, self.in_channels, self.image_size, self.image_size)
        for i in range(batch_size):
            out[i] = torchvision.utils.make_grid(out_patch[0], nrow=self.num_patch_row, padding=0)
        return out
        




#patchselfconv = PatchSelfConv()
#images = torch.ones(2, 3, 32, 32)
#out = patchselfconv(images)
#print(out.size())



class Vit_alta(nn.Module):
    def __init__(self, in_channels:int=3, num_classes:int=10, num_patch_row:int=4, image_size:int=32, num_blocks:int=7, ffn_dim:int=384*4, dropout:float=0.):
        """ 
        引数:
            in_channels: 入力画像のチャンネル数
            num_classes: 画像分類のクラス数
            num_patch_row: 1辺のパッチの数
            image_size: 入力画像の1辺の大きさ。入力画像の高さと幅は同じであると仮定 
            num_blocks: Patch Self Convの数
            hidden_dim: Patch Self ConvのFFNにおける中間層のベクトルの長さ 
            dropout: ドロップアウト率
        """
        super(Vit_alta, self).__init__()
        self.image_pix = image_size**2

        # Encoder (Patch Self Conv) sequential
        self.encoder = nn.Sequential(*[
            PatchSelfConv(
                in_channels,
                num_patch_row,
                image_size,
                ffn_dim,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(in_channels * self.image_pix),
            nn.Linear(in_channels*self.image_pix, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        引数:
            x: Vit_altaへの入力画像。形状は、(B, C, H, W)
                B: バッチサイズ、C:チャンネル数、H:高さ、W:幅
        返り値:
            out: Vit_altaの出力。形状は、(B, M)。[式(10)]
                B:バッチサイズ、M:クラス数 
        """

        out = self.encoder(x)
        out = torch.flatten(out, start_dim=1)
        out = self.mlp_head(out)

        return out



#num_classes = 100
#batch_size, channel, height, width = 10, 3, 32, 32
#num_patch_row = 4
#x = torch.randn(batch_size, channel, height, width)
#vit_alta = Vit_alta(in_channels=channel, num_classes=num_classes, num_patch_row=num_patch_row)
#pred = vit_alta(x)
#print(pred.size())