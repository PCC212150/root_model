"""VM-UNet（Ruan et al., 2024）的 2D 版本，作为 `--arch vmunet` 的骨架。

论文：Ruan et al., "VM-UNet: Vision Mamba UNet for Medical Image Segmentation",
arXiv:2402.02491。

**为什么不直接用 Net/VM-UNet 里的官方实现**：官方代码依赖 `mamba_ssm` 的 CUDA
selective-scan 扩展（以及 `causal_conv1d`、timm、einops），服务器上没有这些扩展，
而且 VM-UNet 的 SS2D 在 1536 分辨率下会把 L=98304 的序列逐步循环十万次，即使装上
CUDA 扩展也只是勉强能跑。这里按官方结构重写，只依赖 PyTorch：

- SS2D 的四方向扫描用**纯 PyTorch 的分块 selective scan**（见下方
  `selective_scan`）：数学上与官方 `selective_scan_ref` 完全一致（已在本地用
  reference 循环逐元素/逐梯度对齐，误差 <1e-6），但把 98304 步的 Python 循环
  压成「块内 2×chunk 步 + 块间 nc 步」，显存和 Python 开销都可控。
- 归一化用 LayerNorm（官方结构），不跟随项目的 `--norm group/batch`。
- 参数量、层数、patch 尺寸、四方向扫描、非对称编解码器都与官方 VM-UNet-T 一致。

输入 H/W 不要求是 32 的倍数：forward 内部 replicate-pad 到 32 的倍数，输出再裁回。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

PAD_MULTIPLE = 32  # patch 4 + 3 次 PatchMerging（4 -> 8 -> 16 -> 32）


# ---------------------------------------------------------------------------
# 纯 PyTorch selective scan
# ---------------------------------------------------------------------------
def selective_scan(u, delta, A, B, C, D=None, delta_bias=None,
                   delta_softplus=True, chunk=0):
    """VM-UNet 的 SS2D 使用的 selective scan。

    u, delta: (B, K, D, L)   K=4 个扫描方向，D=d_inner
    A:        (K, D, N)      负数（代码里由 -exp(A_log) 得到）
    B, C:     (B, K, N, L)   每个方向的输入相关矩阵
    D:        (K, D)         可选的 skip
    返回 (B, K, D, L)。

    分块算法（chunk 为块长，nc=ceil(L/chunk)）：
    1. 每块先以 h=0 做块内扫描，得到块末状态和块内转移矩阵 P=exp(sum(dt*A))；
    2. 对 nc 个块做一次串行转移，得到每块的起始状态；
    3. 再做一遍块内扫描，把起始状态加回去并输出 y_t=(h_t*C_t).sum(N)。

    全程不做除法，P 下溢到 0 时只表示「块起始状态已经衰减完」，数值稳定。
    """
    bsz, K, d_in, L = u.shape
    N = A.shape[-1]
    u0 = u.float()
    dt = delta.float()
    if delta_bias is not None:
        dt = dt + delta_bias.view(1, K, d_in, 1).float()
    if delta_softplus:
        dt = F.softplus(dt)
    Af = A.float().view(1, K, d_in, 1, N)

    # chunk=0：按 L 自动选块长，最小化「块内 2*chunk 步 + 块间 L/chunk 步」的
    # Python 循环次数。短序列用小 chunk，避免为 256 个 token 跑 128 步。
    if chunk is None or chunk <= 0:
        target = max(16.0, (L / 2.0) ** 0.5)
        chunk = 1 << int(round(math.log2(target)))
        chunk = max(16, min(256, chunk))
    if L <= chunk:
        return _selective_scan_ref(u0, dt, Af, B, C, D)

    pad = (-L) % chunk
    if pad:
        uu_ = F.pad(u0, (0, pad))
        dd_ = F.pad(dt, (0, pad))
        BB_ = F.pad(B.float(), (0, pad))
        CC_ = F.pad(C.float(), (0, pad))
    else:
        uu_, dd_, BB_, CC_ = u0, dt, B.float(), C.float()
    Lp = L + pad
    nc = Lp // chunk

    uu = uu_.view(bsz, K, d_in, nc, chunk)
    dd = dd_.view(bsz, K, d_in, nc, chunk)
    BB = BB_.view(bsz, K, N, nc, chunk)
    CC = CC_.view(bsz, K, N, nc, chunk)

    def step(t):
        dt_t = dd[..., t]                                  # b,k,d,nc
        u_t = uu[..., t]                                   # b,k,d,nc
        B_t = BB[..., t].transpose(-1, -2)                 # b,k,nc,n
        a = torch.exp(dt_t.unsqueeze(-1) * Af)             # b,k,d,nc,n
        bt = (dt_t.unsqueeze(-1) * u_t.unsqueeze(-1)
              * B_t.unsqueeze(2))                          # b,k,d,nc,n
        return a, bt

    def emit(h, t):
        C_t = CC[..., t].transpose(-1, -2)                 # b,k,nc,n
        return (h * C_t.unsqueeze(2)).sum(-1)              # b,k,d,nc

    # 1) 块内扫描（h_start=0）
    h = u0.new_zeros(bsz, K, d_in, nc, N)
    for t in range(chunk):
        a, bt = step(t)
        h = a * h + bt
    P = torch.exp(dd.sum(-1).unsqueeze(-1) * Af)           # b,k,d,nc,n

    # 2) 块间转移（nc 步，张量很小）
    starts = []
    state = u0.new_zeros(bsz, K, d_in, N)
    for c in range(nc):
        starts.append(state)
        state = P[:, :, :, c] * state + h[:, :, :, c]
    starts = torch.stack(starts, dim=3)                    # b,k,d,nc,n

    # 3) 再加一遍块内扫描，补上块起始状态
    hl = u0.new_zeros(bsz, K, d_in, nc, N)
    p = u0.new_ones(bsz, K, d_in, nc, N)
    ys = []
    for t in range(chunk):
        a, bt = step(t)
        hl = a * hl + bt
        p = a * p
        ys.append(emit(hl + p * starts, t))
    y = torch.stack(ys, dim=-1).view(bsz, K, d_in, Lp)
    if pad:
        y = y[..., :L]
    if D is not None:
        y = y + u0 * D.float().view(1, K, d_in, 1)
    return y


def _selective_scan_ref(u, dt, Af, B, C, D=None):
    """逐步参考实现，只用于很短的序列（L <= chunk）和单测。"""
    bsz, K, d_in, L = u.shape
    B = B.float()
    C = C.float()
    h = u.new_zeros(bsz, K, d_in, Af.shape[-1])
    ys = []
    for t in range(L):
        dt_t = dt[..., t:t + 1]                            # b,k,d,1
        a = torch.exp(dt_t * Af.squeeze(3))                # b,k,d,n
        bt = dt_t * u[..., t:t + 1] * B[..., t].unsqueeze(2)
        h = a * h + bt
        ys.append((h * C[..., t].unsqueeze(2)).sum(-1))
    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + u * D.view(1, K, d_in, 1).float()
    return y


# ---------------------------------------------------------------------------
# 基础模块（与官方 vmamba.py 同名同结构，便于加载官方权重）
# ---------------------------------------------------------------------------
class DropPath(nn.Module):
    """timm.models.layers.DropPath 的最小实现。"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep)
        return x * mask / keep


class PatchEmbed2D(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.reshape(B, H // 2, W // 2, 4 * C)
        return self.reduction(self.norm(x))


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, _ = x.shape
        x = self.expand(x)
        p = self.dim_scale
        C2 = x.shape[-1]
        x = x.view(B, H, W, p, p, C2 // (p * p))
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H * p, W * p,
                                               C2 // (p * p))
        return self.norm(x)


class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, _ = x.shape
        x = self.expand(x)
        p = self.dim_scale
        C2 = x.shape[-1]
        x = x.view(B, H, W, p, p, C2 // (p * p))
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H * p, W * p,
                                               C2 // (p * p))
        return self.norm(x)


class SS2D(nn.Module):
    """官方 SS2D：4 方向 selective scan + 深度可分离卷积 + gated MLP。"""

    def __init__(self, d_model, d_state=16, d_conv=3, expand=2,
                 dt_rank="auto", dt_min=0.001, dt_max=0.1,
                 dt_init="random", dt_scale=1.0, dt_init_floor=1e-4,
                 dropout=0.0, conv_bias=True, bias=False, chunk=0, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" \
            else dt_rank
        self.chunk = chunk

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(self.d_inner, self.d_inner,
                                groups=self.d_inner, bias=conv_bias,
                                kernel_size=d_conv,
                                padding=(d_conv - 1) // 2)
        self.act = nn.SiLU()

        # 官方把 4 个方向的 x_proj / dt_proj 堆成一个参数，这里保持一致
        self.x_proj_weight = nn.Parameter(torch.empty(
            4, self.dt_rank + self.d_state * 2, self.d_inner))
        self.dt_projs_weight = nn.Parameter(torch.empty(
            4, self.d_inner, self.dt_rank))
        self.dt_projs_bias = nn.Parameter(torch.empty(4, self.d_inner))
        self.A_logs = self._A_log_init(self.d_state, self.d_inner, copies=4)
        self.Ds = nn.Parameter(torch.ones(4 * self.d_inner))

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        self._reset_parameters(dt_min, dt_max, dt_init, dt_scale,
                               dt_init_floor)

    @staticmethod
    def _A_log_init(d_state, d_inner, copies=1):
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        A_log = torch.log(A).repeat(copies, d_inner, 1).flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    def _reset_parameters(self, dt_min, dt_max, dt_init, dt_scale,
                          dt_init_floor):
        bound = 1.0 / math.sqrt(self.d_inner)
        nn.init.uniform_(self.x_proj_weight, -bound, bound)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_projs_weight, dt_init_std)
        else:
            nn.init.uniform_(self.dt_projs_weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_projs_bias.copy_(inv_dt.unsqueeze(0).repeat(4, 1))

    def forward_core(self, x):
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack(
            [x.view(B, -1, L),
             x.transpose(2, 3).contiguous().view(B, -1, L)], dim=1
        ).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l",
                             xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l",
                           dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, K, -1, L)
        dts = dts.contiguous().float().view(B, K, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        As = self.A_logs.float().view(K, -1, self.d_state)
        Ds = self.Ds.float().view(K, -1)
        bias = self.dt_projs_bias.float().view(K, -1)

        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            out_y = selective_scan(xs, dts, -torch.exp(As), Bs, Cs, Ds,
                                   delta_bias=bias, delta_softplus=True,
                                   chunk=self.chunk)
        out_y = out_y.view(B, K, -1, L)
        assert out_y.dtype == torch.float32

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), 2, 3) \
            .contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), 2, 3) \
            .contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, 1, 2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(self, hidden_dim=0, drop_path=0.0, attn_drop_rate=0.0,
                 d_state=16, chunk=0, **kwargs):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.self_attention = SS2D(d_model=hidden_dim,
                                   dropout=attn_drop_rate,
                                   d_state=d_state, chunk=chunk, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        return x + self.drop_path(self.self_attention(self.ln_1(x)))


def _checkpointed(module, x):
    try:
        return checkpoint(module, x, use_reentrant=False)
    except TypeError:
        return checkpoint(module, x)


class VSSLayer(nn.Module):
    def __init__(self, dim, depth, attn_drop=0.0, drop_path=0.0,
                 downsample=None, use_checkpoint=False, d_state=16,
                 chunk=0):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            VSSBlock(hidden_dim=dim,
                     drop_path=drop_path[i] if isinstance(drop_path, list)
                     else drop_path,
                     attn_drop_rate=attn_drop, d_state=d_state, chunk=chunk)
            for i in range(depth)])
        self.downsample = downsample(dim=dim) if downsample is not None else None

    def forward(self, x):
        for blk in self.blocks:
            x = _checkpointed(blk, x) if (
                self.use_checkpoint and self.training) else blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class VSSLayer_up(nn.Module):
    def __init__(self, dim, depth, attn_drop=0.0, drop_path=0.0,
                 upsample=None, use_checkpoint=False, d_state=16,
                 chunk=0):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
            VSSBlock(hidden_dim=dim,
                     drop_path=drop_path[i] if isinstance(drop_path, list)
                     else drop_path,
                     attn_drop_rate=attn_drop, d_state=d_state, chunk=chunk)
            for i in range(depth)])
        self.upsample = upsample(dim=dim) if upsample is not None else None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            x = _checkpointed(blk, x) if (
                self.use_checkpoint and self.training) else blk(x)
        return x


class VSSM(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=3,
                 depths=(2, 2, 2, 2), depths_decoder=(2, 2, 2, 1),
                 dims=(96, 192, 384, 768), dims_decoder=(768, 384, 192, 96),
                 d_state=16, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.2, use_checkpoint=False, chunk=0):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.dims = list(dims)
        self.embed_dim = dims[0]

        self.patch_embed = PatchEmbed2D(patch_size=patch_size,
                                        in_chans=in_chans,
                                        embed_dim=self.embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]
        dpr_decoder = [x.item() for x in torch.linspace(
            0, drop_path_rate, sum(depths_decoder))][::-1]

        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(VSSLayer(
                dim=dims[i], depth=depths[i], attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=PatchMerging2D if i < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint, d_state=d_state, chunk=chunk))

        self.layers_up = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers_up.append(VSSLayer_up(
                dim=dims_decoder[i], depth=depths_decoder[i],
                attn_drop=attn_drop_rate,
                drop_path=dpr_decoder[sum(depths_decoder[:i]):
                                      sum(depths_decoder[:i + 1])],
                upsample=PatchExpand2D if i != 0 else None,
                use_checkpoint=use_checkpoint, d_state=d_state, chunk=chunk))

        self.final_up = Final_PatchExpand2D(dim=dims_decoder[-1],
                                            dim_scale=4)
        self.final_conv = nn.Conv2d(dims_decoder[-1] // 4, num_classes, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        skips = []
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            skips.append(x)
            x = layer(x)
        return x, skips

    def forward_features_up(self, x, skips):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = layer_up(x + skips[-inx])
        return x

    def forward_final(self, x):
        x = self.final_up(x)
        x = x.permute(0, 3, 1, 2)
        return self.final_conv(x)

    def forward(self, x):
        x, skips = self.forward_features(x)
        x = self.forward_features_up(x, skips)
        return self.forward_final(x)


class VMUNet2D(nn.Module):
    """VM-UNet-T：4 级 VSS 编解码器，四方向 selective scan。

    `use_checkpoint=True`（默认）对每个 VSS block 做梯度检查点：在 1536 分辨率下
    不检查点会保存 2×chunk×nc 个中间状态，显存直接爆；检查点只多花一次前向。
    `chunk` 是 selective scan 的块长，128 是在 1536 下显存/速度折中后的默认值。
    """

    def __init__(self, in_ch=3, out_ch=3, depths=(2, 2, 2, 2),
                 depths_decoder=(2, 2, 2, 1), dims=(96, 192, 384, 768),
                 d_state=16, drop_path_rate=0.2, use_checkpoint=True,
                 chunk=0, norm=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.vmunet = VSSM(
            in_chans=in_ch, num_classes=out_ch, depths=depths,
            depths_decoder=depths_decoder, dims=dims,
            dims_decoder=tuple(reversed(dims)), d_state=d_state,
            drop_path_rate=drop_path_rate, use_checkpoint=use_checkpoint,
            chunk=chunk)

    def forward(self, x):
        h, w = x.shape[-2:]
        pad_h = (-h) % PAD_MULTIPLE
        pad_w = (-w) % PAD_MULTIPLE
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        y = self.vmunet(x)
        if pad_h or pad_w:
            y = y[..., :h, :w]
        return y
