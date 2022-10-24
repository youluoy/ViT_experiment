
#%%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchSelfConv(nn.Module):
    def __init__(self, in_channels:int=3, num_patch_row:int=4, image_size:int=32):
        """ 
        引数:
            in_channels: 入力画像のチャンネル数
            num_patch_row: 高さ方向のパッチの数。例は2x2であるため、2をデフォルト値とした 
            image_size: 入力画像の1辺の大きさ。入力画像の高さと幅は同じであると仮定
        """
        super(PatchSelfConv, self).__init__()
        self.in_channels = in_channels
        self.num_patch_row = num_patch_row
        self.image_size = image_size

        self.num_patch = self.num_patch_row**2

        self.patch_size = int(self.image_size // self.num_patch_row)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ 
        引数:
            x: 入力画像。形状は、(B, C, H, W)。[式(1)]
                B: バッチサイズ、C:チャンネル数、H:高さ、W:幅
        返り値:
            z_0: ViTへの入力。形状は、(B, C, )。
                B:バッチサイズ、N:トークン数、D:埋め込みベクトルの長さ
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


        # 入力テンソルを複製
        ## (B, C, H, W)
        z_0 = x
        # パッチに分割し、パッチ数分のフィルタを作成する
        ## (B, C, H, W) -> (B, C, P, H/P_r, W/P_r)
        z_0 = z_0.unfold(2, kernel_size=self.patch_size, stride=self.patch_size).unfold(3, kernel_size=self.patch_size, stride=self.patch_size)
        z_0 = z_0.reshape(2, 3, -1, 2, 2)
        #並べ替えしてF.conv2d用の形にする
        ## (B, C, P, H/P_r, W/P_r) -> (B, P, C, H/P_r, W/P_r)
        z_0 = z_0.permute(0, 2, 1, 3, 4)
        # 作ったパッチと画像との畳み込み演算を行う F.covn2d outputチャンネル方向がパッチの枚数に対応する
        ## -> (B, P, C, H - (patchsize - 1), W - (patchsize - 1))
        features = []
        for i in range(x.size()[0]):
            conved = F.conv2d(x[i].unsqueeze(dim=0), z_0[i])
            features.append(conved)
        features = torch.cat(features, dim = 0)
        # 全結合を用いてもとのパッチサイズに戻して結合　チャンネルが増えてパッチ数が消える
        ## 










