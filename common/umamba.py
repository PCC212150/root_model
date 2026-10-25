"""U-Mamba（Ma et al., 2024）的 2D 版本，作为 `--arch umamba` 的骨架。

官方实现（Net/U-Mamba）是 nnU-Net 的一个 trainer，依赖 `mamba_ssm` 的 CUDA
selective-scan 扩展。服务器上既没有 mamba_ssm，也没有外网/编译环境，因此这里
**按官方 UMambaBot_2d 的结构重写**了一版自包含实现：

    U-Net（9 级、残差块、InstanceNorm/GroupNorm）
      -> 最深层特征 [B, 512, H/256, W/256]
      -> MambaLayer（LayerNorm + selective scan）
      -> 解码器（逐级上采样 + skip）

与官方代码的差异只有两处，均已在代码里标注：

1. Mamba 用**纯 PyTorch 的逐步 selective scan**（没有 CUDA 扩展）。U-Mamba_Bot
   的 Mamba 只在最深层做，1536x1024 输入时只有 6x4=24 个 token，逐步循环的
   开销可以忽略；U-Mamba_Enc 在浅层也有 Mamba（上百万 token），本实现不适合，
   所以只提供 Bot 变体。
2. 归一化层跟随项目的 `--norm`（group / batch），不是官方的 InstanceNorm。

输入 H/W 不要求是 256 的倍数：forward 内部会 replicate-pad 到 256 的倍数，
输出再裁回原尺寸。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.unet import make_norm

PAD_MULTIPLE = 256  # 9 级下采样：2**8


# ---------------------------------------------------------------------------
# 纯 PyTorch Mamba（state-spaces/mamba 的 Mamba 模块，去掉 CUDA 扩展）
# ---------------------------------------------------------------------------
def selective_scan(u, delta, A, B, C, D):
    """逐步 selective scan。

    u, delta: (B, L, d_inner)
    A:        (d_inner, d_state)   —— 负数
    B, C:     (B, L, d_state)
    D:        (d_inner,)

    返回 (B, L, d_inner)。
    """
    Bsz, L, d_inner = u.shape
    d_state = A.shape[1]
    u_t = u.transpose(1, 2)          # B, d_inner, L
    dt = delta.transpose(1, 2)       # B, d_inner, L
    B_t = B.transpose(1, 2)          # B, d_state, L
    C_t = C.transpose(1, 2)          # B, d_state, L

    h = u.new_zeros(Bsz, d_inner, d_state)
    ys = []
    A = A.unsqueeze(0)               # 1, d_inner, d_state
    for t in range(L):
        dt_t = dt[..., t]            # B, d_inner
        dA = torch.exp(dt_t.unsqueeze(-1) * A)
        dB = dt_t.unsqueeze(-1) * B_t[..., t].unsqueeze(1)
        h = h * dA + dB * u_t[..., t].unsqueeze(-1)
        ys.append((h * C_t[..., t].unsqueeze(1)).sum(-1))
    y = torch.stack(ys, dim=-1)      # B, d_inner, L
    y = y + u_t * D.view(1, -1, 1)
    return y.transpose(1, 2)


class Mamba(nn.Module):
    """与 mamba_ssm.Mamba 接口一致的最小实现（d_model, d_state, d_conv, expand）。"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A 的初始化与官方一致：A[n] = -(n+1)，存 log 形式保证负数
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(torch.log(A).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # 官方把 dt_proj.bias 初始化成 softplus^-1(dt_init)，dt_init=0.001
        with torch.no_grad():
            dt_init = 0.001
            self.dt_proj.bias.fill_(math.log(math.exp(dt_init) - 1.0))

    def forward(self, x):
        # x: (B, L, d_model)
        Bsz, L, _ = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)          # B,L,d_inner

        x = x.transpose(1, 2)               # B,d_inner,L
        x = self.conv1d(x)[..., :L]         # 因果卷积，截掉右侧 padding
        x = F.silu(x)
        x = x.transpose(1, 2)               # B,L,d_inner

        x_dbl = self.x_proj(x)
        dt, B_ssm, C_ssm = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))   # B,L,d_inner

        A = -torch.exp(self.A_log.float())
        y = selective_scan(x, dt, A, B_ssm, C_ssm, self.D.float())
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaLayer(nn.Module):
    """官方 UMambaBot_2d 的 MambaLayer：图像特征 -> token -> Mamba -> 还原。"""

    def __init__(self, dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(d_model=dim, d_state=d_state,
                           d_conv=d_conv, expand=expand)

    def forward(self, x):
        orig_dtype = x.dtype
        # Mamba 的逐步 scan 在 fp16 下不稳定；官方也是先转 fp32、关闭 autocast
        x = x.float()
        B, C, *dims = x.shape
        x_flat = x.flatten(2).transpose(1, 2)       # B, L, C
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            y = self.mamba(self.norm(x_flat))
        y = y.transpose(1, 2).reshape(B, C, *dims)
        return y.to(orig_dtype)


# ---------------------------------------------------------------------------
# U-Mamba Bot 的残差 U-Net
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """官方 BasicResBlock：conv-norm-act -> conv-norm -> +skip -> act。"""

    def __init__(self, in_ch, out_ch, norm="group", stride=1, kernel=3,
                 use_1x1conv=True):
        super().__init__()
        pad = kernel // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                               padding=pad, bias=True)
        self.norm1 = make_norm(norm, out_ch)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel, padding=pad, bias=True)
        self.norm2 = make_norm(norm, out_ch)
        self.act2 = nn.LeakyReLU(inplace=True)
        self.conv3 = (nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=True)
                      if use_1x1conv else None)

    def forward(self, x):
        y = self.act1(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        if self.conv3 is not None:
            x = self.conv3(x)
        return self.act2(y + x)


class Encoder(nn.Module):
    def __init__(self, in_ch, features, n_blocks, norm="group"):
        super().__init__()
        self.stems = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.strides = [(1, 1)] + [(2, 2)] * (len(features) - 1)

        stem = [ResBlock(in_ch, features[0], norm=norm)]
        stem += [ResBlock(features[0], features[0], norm=norm)
                 for _ in range(n_blocks[0] - 1)]
        self.stems.append(nn.Sequential(*stem))

        prev = features[0]
        for s in range(len(features)):
            blocks = [ResBlock(prev, features[s], norm=norm,
                               stride=self.strides[s])]
            blocks += [ResBlock(features[s], features[s], norm=norm)
                       for _ in range(n_blocks[s] - 1)]
            self.stages.append(nn.Sequential(*blocks))
            prev = features[s]
        self.out_channels = list(features)

    def forward(self, x):
        x = self.stems[0](x)
        skips = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


class Decoder(nn.Module):
    def __init__(self, encoder, out_ch, n_blocks, norm="group"):
        super().__init__()
        features = encoder.out_channels
        n_stages = len(features)
        self.ups = nn.ModuleList()
        self.stages = nn.ModuleList()
        for s in range(1, n_stages):
            below = features[-s]
            skip = features[-(s + 1)]
            stride = encoder.strides[-s]
            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                nn.Conv2d(below, skip, kernel_size=1, bias=True)))
            in_ch = 2 * skip if s < n_stages - 1 else skip
            blocks = [ResBlock(in_ch, skip, norm=norm)]
            blocks += [ResBlock(skip, skip, norm=norm)
                       for _ in range(n_blocks[s - 1] - 1)]
            self.stages.append(nn.Sequential(*blocks))
        self.out = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, skips):
        x = skips[-1]
        for i, (up, stage) in enumerate(zip(self.ups, self.stages)):
            x = up(x)
            if i < len(self.ups) - 1:
                skip = skips[-(i + 2)]
                if x.shape[-2:] != skip.shape[-2:]:
                    skip = F.interpolate(skip, size=x.shape[-2:],
                                         mode="nearest")
                x = torch.cat([x, skip], dim=1)
            x = stage(x)
        return self.out(x)


class UMambaBot2D(nn.Module):
    """U-Mamba_Bot 2D：9 级残差 U-Net + 最深层 Mamba。

    `base=32`、9 级、特征 (32,64,128,256,512,512,512,512,512) 与 nnU-Net
    2D 的默认 plans 对齐（本项目 nnUNetPlans 的 2D 配置就是这组参数）。
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 3, base: int = 32,
                 n_stages: int = 9, norm: str = "group", d_state: int = 16):
        super().__init__()
        features = [min(base * 2 ** i, 512) for i in range(n_stages)]
        # 官方 UMambaBot：后半段每级只保留 1 个 block，控制显存
        n_blocks = [2] * n_stages
        for s in range(math.ceil(n_stages / 2), n_stages):
            n_blocks[s] = 1
        n_blocks_dec = [2] * (n_stages - 1)
        for s in range(math.ceil((n_stages - 1) / 2 + 0.5), n_stages - 1):
            n_blocks_dec[s] = 1

        self.encoder = Encoder(in_ch, features, n_blocks, norm=norm)
        self.mamba = MambaLayer(dim=features[-1], d_state=d_state)
        self.decoder = Decoder(self.encoder, out_ch, n_blocks_dec, norm=norm)
        self.in_ch = in_ch
        self.out_ch = out_ch

    def forward(self, x):
        h, w = x.shape[-2:]
        pad_h = (-h) % PAD_MULTIPLE
        pad_w = (-w) % PAD_MULTIPLE
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        skips = self.encoder(x)
        skips[-1] = self.mamba(skips[-1])
        y = self.decoder(skips)
        if pad_h or pad_w:
            y = y[..., :h, :w]
        return y
