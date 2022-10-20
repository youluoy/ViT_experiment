
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

        self.patch_self_conv = F.conv2d(
            in_channels = self.in_channels,
            out_channels = self.in_channels,
            kernel_size = self.patch_size,
            stride = 2
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

        F.conv2d(
            input = x,
            weight = 
        )













