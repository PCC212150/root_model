"""2D Attention U-Net（Oktay et al., 2018），接口与 common/unet.py 的 UNet 一致。

论文：Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas",
MIDL 2018。参考实现是 `Net/Attention-Gated-Networks` 的
`models/layers/grid_attention_layer.py`（PyTorch 版）。

**为什么重写而不是直接 import**：那份仓库的模型是 3D 医学影像（Conv3d +
trilinear 上采样），且依赖它自己的 `models.networks_other` / 初始化工具；本项目是
2D 三通道分割，直接套用会破坏数据管线。这里保留原实现的**加法注意力门**结构：

    theta = Conv1x1(x)                 跳连特征
    phi   = Conv1x1(gating)            粗层解码特征（已上采样到跳连分辨率）
    f     = ReLU(theta + phi)
    alpha = sigmoid(Conv1x1(f))        注意力系数 (B,1,H,W)
    y     = alpha * x                  加权后的跳连
    out   = Conv1x1(y) + Norm           原仓库的 W（输出变换）

与原仓库的唯一差异：W 里的归一化跟随本项目的 `--norm`（batch/group），而不是写死
BatchNorm —— 否则 `--norm group` 时这里会混进 BN，小 batch 下不稳定。

注意力门加在**每一个** decoder 跳连上（标准 Attention U-Net），gating signal 是
上一层的解码特征。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.unet import DoubleConv, Down, make_norm


class AttentionGate(nn.Module):
    """加法注意力门：用粗层特征 g 给跳连特征 x 生成逐像素权重。"""

    def __init__(self, x_ch: int, g_ch: int, norm: str = "batch",
                 inter: int = None):
        super().__init__()
        inter = inter or max(x_ch // 2, 1)
        self.theta = nn.Conv2d(x_ch, inter, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(g_ch, inter, kernel_size=1, bias=True)
        self.psi = nn.Conv2d(inter, 1, kernel_size=1, bias=True)
        # 原仓库的 W：1x1 卷积 + BN（这里归一化类型跟随项目配置）
        self.W = nn.Sequential(nn.Conv2d(x_ch, x_ch, kernel_size=1, bias=False),
                               make_norm(norm, x_ch))

    def forward(self, x, g):
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear",
                              align_corners=False)
        f = F.relu(self.theta(x) + self.phi(g), inplace=True)
        alpha = torch.sigmoid(self.psi(f))
        return self.W(alpha * x)


class AttentionUp(nn.Module):
    """上采样 + 注意力门控跳连 + 拼接 + DoubleConv。"""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, norm: str = "batch"):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.gate = AttentionGate(skip_ch, in_ch // 2, norm=norm)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch, norm=norm)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            dh = skip.shape[2] - x.shape[2]
            dw = skip.shape[3] - x.shape[3]
            skip = skip[:, :, dh // 2: dh // 2 + x.shape[2],
                        dw // 2: dw // 2 + x.shape[3]]
        skip = self.gate(skip, x)
        return self.conv(torch.cat([x, skip], dim=1))


class AttentionUNet(nn.Module):
    """经典 U-Net 骨架 + 4 个注意力门（输出/输入接口与 UNet 完全一致）。"""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 64,
                 norm: str = "batch"):
        super().__init__()
        features = [base, base * 2, base * 4, base * 8]
        self.down1 = Down(in_ch, features[0], norm=norm)
        self.down2 = Down(features[0], features[1], norm=norm)
        self.down3 = Down(features[1], features[2], norm=norm)
        self.down4 = Down(features[2], features[3], norm=norm)
        self.bottleneck = DoubleConv(features[3], features[3] * 2, norm=norm)
        self.up1 = AttentionUp(features[3] * 2, features[3], features[3], norm=norm)
        self.up2 = AttentionUp(features[3], features[2], features[2], norm=norm)
        self.up3 = AttentionUp(features[2], features[1], features[1], norm=norm)
        self.up4 = AttentionUp(features[1], features[0], features[0], norm=norm)
        self.out = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        p1, s1 = self.down1(x)
        p2, s2 = self.down2(p1)
        p3, s3 = self.down3(p2)
        p4, s4 = self.down4(p3)
        x = self.bottleneck(p4)
        x = self.up1(x, s4)
        x = self.up2(x, s3)
        x = self.up3(x, s2)
        x = self.up4(x, s1)
        return self.out(x)
