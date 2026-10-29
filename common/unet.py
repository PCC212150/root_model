"""经典 U-Net（归一化可选 BatchNorm / GroupNorm，见 make_norm）。

编码器 4 层(64→128→256→512) + 瓶颈(1024)，解码器对称；每层 DoubleConv。
输入输出长宽需为 16 的倍数（common.image_io.target_size 保证）；
因池化取整可能出现的奇偶尺寸错位，拼接前对 skip 特征做中心裁剪对齐。

`use_checkpoint=True`（对应 `--unet-checkpoint`）对编码器 DoubleConv、瓶颈和
整个解码器 Up 做梯度检查点：前向不保存中间激活，反向重算。不改变 state_dict，
只影响训练显存/速度；2736 + GroupNorm 在 24GB 卡上 OOM 时使用。
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def _checkpointed(module, x):
    """单输入梯度检查点；use_reentrant=False 对嵌套/非张量场景更稳。"""
    try:
        return checkpoint(module, x, use_reentrant=False)
    except TypeError:
        return checkpoint(module, x)


def _checkpointed2(module, x, y):
    """双输入梯度检查点（U-Net 的 Up 需要 x 和 skip）。"""
    try:
        return checkpoint(module, x, y, use_reentrant=False)
    except TypeError:
        return checkpoint(module, x, y)


def make_norm(kind: str, ch: int) -> nn.Module:
    """按需生成归一化层。

    batch: BatchNorm2d —— 需要 batch≥2 才有意义；batch=1 时每步只用单张图的统计量，
           训练用的统计量与推理用的滑动平均对不上（实测 batch=1 训出的模型在 eval 模式下
           只出 0.36% 前景、train 模式出 1.27%，**欠分割 3.5 倍**）。
    group: GroupNorm —— 不依赖 batch 统计量，train/eval 行为一致，小 batch 下稳定。
    """
    if kind == "group":
        return nn.GroupNorm(num_groups=min(8, ch), num_channels=ch)
    if kind == "batch":
        return nn.BatchNorm2d(ch)
    raise ValueError(f"未知的归一化类型: {kind!r}（可选 batch / group）")


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, norm: str = "batch"):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            make_norm(norm, mid_ch), nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 3, padding=1, bias=False),
            make_norm(norm, out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """返回 (pooled, feat)：pooled 送入下一层编码器；feat(池化前) 作为解码器跳跃连接。"""

    def __init__(self, in_ch, out_ch, norm: str = "batch"):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch, norm=norm)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        feat = self.conv(x)
        return self.pool(feat), feat


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, norm: str = "batch"):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch, norm=norm)

    def forward(self, x, skip):
        x = self.up(x)
        # 中心裁剪 skip 至与 x 同尺寸（仅奇偶错位时生效）
        if x.shape[-2:] != skip.shape[-2:]:
            dh, dw = skip.shape[2] - x.shape[2], skip.shape[3] - x.shape[3]
            skip = skip[:, :, dh // 2: dh // 2 + x.shape[2],
                        dw // 2: dw // 2 + x.shape[3]]
        return self.conv(torch.cat([x, skip], dim=1))


class UNet(nn.Module):
    """in_ch 输入通道(3: RGB)；out_ch 输出通道。

    输出是**多标签**（逐通道 sigmoid，不用 softmax）：默认 3 通道对应
    config.CLASS_NAMES = ("root", "stem", "check")，三类并不互斥
    —— 茎横截面与根系都落在检查范围之内。
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 64,
                 norm: str = "batch", use_checkpoint: bool = False):
        super().__init__()
        # 只影响训练显存/速度，不改变 state_dict；--unet-checkpoint 打开
        self.use_checkpoint = bool(use_checkpoint)
        features = [base, base * 2, base * 4, base * 8]  # 64,128,256,512
        self.down1 = Down(in_ch, features[0], norm=norm)
        self.down2 = Down(features[0], features[1], norm=norm)
        self.down3 = Down(features[1], features[2], norm=norm)
        self.down4 = Down(features[2], features[3], norm=norm)
        self.bottleneck = DoubleConv(features[3], features[3] * 2, norm=norm)  # 1024

        self.up1 = Up(features[3] * 2, features[3], features[3], norm=norm)
        self.up2 = Up(features[3], features[2], features[2], norm=norm)
        self.up3 = Up(features[2], features[1], features[1], norm=norm)
        self.up4 = Up(features[1], features[0], features[0], norm=norm)
        self.out = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def _down(self, down, x):
        # 只 checkpoint DoubleConv，池化很便宜；feat 作为 skip 仍会保留
        if self.use_checkpoint and self.training:
            feat = _checkpointed(down.conv, x)
        else:
            feat = down.conv(x)
        return down.pool(feat), feat

    def _up(self, up, x, skip):
        # checkpoint 整个 Up：重算 ConvTranspose + concat + DoubleConv，
        # 相比只包 conv 能省掉全分辨率 concat 的激活
        if self.use_checkpoint and self.training:
            return _checkpointed2(up, x, skip)
        return up(x, skip)

    def forward(self, x):
        p1, s1 = self._down(self.down1, x)   # skip 特征 [64]（池化前，全分辨率）
        p2, s2 = self._down(self.down2, p1)
        p3, s3 = self._down(self.down3, p2)
        p4, s4 = self._down(self.down4, p3)
        if self.use_checkpoint and self.training:
            x = _checkpointed(self.bottleneck, p4)
        else:
            x = self.bottleneck(p4)
        x = self._up(self.up1, x, s4)
        x = self._up(self.up2, x, s3)
        x = self._up(self.up3, x, s2)
        x = self._up(self.up4, x, s1)
        return self.out(x)
