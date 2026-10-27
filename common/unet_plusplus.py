"""2D UNet++（Nested U-Net），接口与 common/unet.py 的 UNet 完全一致。

论文：Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image
Segmentation", DLMIA 2018 / IEEE TMI 2019。

**为什么不直接用 Net/UNetPlusPlus 里的官方实现**：那份是 nnU-Net 框架的一部分，
面向 3D 医学影像，依赖 batchgenerators / 环境变量 / nnU-Net 数据集格式，和本项目
的「原图 + labelme + 三通道掩码」管线对不上。这里按论文结构重写 2D 版本：
编码器仍是 4 次下采样，解码器改成**嵌套的密集跳跃连接** —— 第 i 层第 j 个节点
同时接收同层前面 j 个节点的输出和下一层第 j-1 个节点的上采样结果。

    节点 X[i][j]：
        j = 0：编码器第 i 层特征（X[i][0]）
        j > 0：concat(X[i][0..j-1], upsample(X[i+1][j-1])) -> DoubleConv

    5 层（0..4，4 是瓶颈），共 15 个节点；输出取 X[0][4]。
    不做深监督（deep supervision）—— 与 U-Net baseline 用同一个损失，只比骨架。

归一化/卷积块直接复用 common/unet.py，保证两边只有「骨架」这一个变量不同。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from common.unet import DoubleConv, Down, make_norm


def _checkpointed(module, x):
    """梯度检查点：前向不保存中间激活，反向时重算。

    use_reentrant=False 对嵌套调用/非张量输出更稳；老版本 torch 没有这个参数时
    退回默认行为（输入张量 requires_grad，满足 reentrant 的要求）。
    """
    try:
        return checkpoint(module, x, use_reentrant=False)
    except TypeError:
        return checkpoint(module, x)


class UNetPlusPlus(nn.Module):
    """in_ch 输入通道；out_ch 输出通道（多标签，逐通道 sigmoid）。

    base 与 UNet 的含义相同：第 0 层 64 通道，每下采样一层翻倍，瓶颈 1024。
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 64,
                 norm: str = "batch", use_checkpoint: bool = False):
        super().__init__()
        # 只影响训练时的显存/速度，不改变 state_dict；--unetpp-checkpoint 打开
        self.use_checkpoint = bool(use_checkpoint)
        features = [base, base * 2, base * 4, base * 8, base * 16]
        self.features = features

        # ---- 编码器（与 UNet 同构）----
        self.down1 = Down(in_ch, features[0], norm=norm)
        self.down2 = Down(features[0], features[1], norm=norm)
        self.down3 = Down(features[1], features[2], norm=norm)
        self.down4 = Down(features[2], features[3], norm=norm)
        self.bottleneck = DoubleConv(features[3], features[4], norm=norm)

        # ---- 嵌套解码器：X[i][j]，i=0..4，j=1..4-i ----
        self.nodes = nn.ModuleDict()
        for i in range(4):
            for j in range(1, 5 - i):
                # 输入 = 同层前 j 个节点 + 下一层上来的 1 个节点
                in_ch_node = features[i] * j + features[i + 1]
                self.nodes[f"{i}_{j}"] = DoubleConv(in_ch_node, features[i],
                                                    norm=norm)

        self.out = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        p1, s1 = self.down1(x)          # X[0][0]
        p2, s2 = self.down2(p1)         # X[1][0]
        p3, s3 = self.down3(p2)         # X[2][0]
        p4, s4 = self.down4(p3)         # X[3][0]
        x5 = self.bottleneck(p4)        # X[4][0]

        nodes = {(0, 0): s1, (1, 0): s2, (2, 0): s3, (3, 0): s4, (4, 0): x5}
        # **必须从深层往浅层算**：X[i][j] 依赖 X[i+1][j-1]，先算 i=3 再算 2/1/0。
        for i in range(3, -1, -1):
            for j in range(1, 5 - i):
                up = F.interpolate(nodes[(i + 1, j - 1)],
                                   size=nodes[(i, 0)].shape[-2:],
                                   mode="bilinear", align_corners=False)
                parts = [nodes[(i, k)] for k in range(j)] + [up]
                node_in = torch.cat(parts, dim=1)
                if self.use_checkpoint and self.training:
                    nodes[(i, j)] = _checkpointed(
                        self.nodes[f"{i}_{j}"], node_in)
                else:
                    nodes[(i, j)] = self.nodes[f"{i}_{j}"](node_in)

        return self.out(nodes[(0, 4)])
