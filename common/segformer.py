"""SegFormer（Xie et al., NeurIPS 2021）的 2D 实现，接口与 common/unet.py 一致。

结构：
    MiT 编码器（4 个 stage，overlap patch embedding + 高效自注意力 + Mix-FFN）
    -> SegFormer 解码头（4 个 1x1 投影到 256 维，上采样到 1/4 后拼接，再融合）
    -> 1x1 分类头，最后双线性上采样回输入分辨率

代码 vendored 自 `Net/SegFormer/mmseg/models/backbones/mix_transformer.py` 与
`Net/SegFormer/mmseg/models/decode_heads/segformer_head.py`（NVIDIA 源许可证），
去掉 mmseg/mmcv 的注册器与 ConvModule 依赖，改成项目自己的 `make_norm`；
结构、通道数、默认超参与官方 SegFormer 一致。

**为什么不用 timm 的 mit_b*：当前环境（timm 1.0.30）里没有 MiT 模型**，
所以编码器直接内置，避免为了一个模型降级/换 timm（Swin-Unet 还在用它）。

支持的变体（`variant`）：b0 / b1 / b2 / b3 / b4 / b5。默认 b2（约 27M 参数，
与 U-Net baseline 的 31M 最接近）。输入尺寸没有固定要求（全卷积），
只要能被 4 整除即可；本项目常用的 1024x1536 满足。

`--segformer-pretrained` 可加载官方 mmseg 权重（键名 `backbone.*` /
`decode_head.*`，或裸 MiT 的 `patch_embed1.*`）；只加载形状一致的键，
编码器覆盖率不足会直接报错，避免「以为用了预训练、其实从零开始」。
"""
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from common.unet import make_norm


# ---------------------------------------------------------------------------
# MiT 编码器
# ---------------------------------------------------------------------------
class Mlp(nn.Module):
    """Mix-FFN：Linear -> 3x3 depthwise conv -> GELU -> Linear。"""

    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """高效自注意力：K/V 先按 sr_ratio 做空间缩减，降低复杂度。"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} 不能被 num_heads {num_heads} 整除"
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                    C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return self.proj_drop(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop,
                              proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """重叠 patch embedding：strides 4/2/2/2 对应 1/4 ~ 1/32 四个 stage。"""

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3,
                 embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class MixVisionTransformer(nn.Module):
    """MiT 编码器：返回 1/4、1/8、1/16、1/32 四个特征图。"""

    def __init__(self, img_size=224, in_chans=3,
                 embed_dims=(64, 128, 320, 512),
                 num_heads=(1, 2, 5, 8),
                 mlp_ratios=(4, 4, 4, 4),
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=(3, 4, 6, 3), sr_ratios=(8, 4, 2, 1)):
        super().__init__()
        self.depths = depths
        self.patch_embed1 = OverlapPatchEmbed(
            img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
            embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(
            img_size=img_size // 4, patch_size=3, stride=2,
            in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(
            img_size=img_size // 8, patch_size=3, stride=2,
            in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(
            img_size=img_size // 16, patch_size=3, stride=2,
            in_chans=embed_dims[2], embed_dim=embed_dims[3])

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            blocks = nn.ModuleList([
                Block(dim=embed_dims[i], num_heads=num_heads[i],
                      mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                      drop=drop_rate, attn_drop=attn_drop_rate,
                      drop_path=dpr[cur + j], norm_layer=norm_layer,
                      sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            setattr(self, f"block{i + 1}", blocks)
            setattr(self, f"norm{i + 1}", norm_layer(embed_dims[i]))
            cur += depths[i]

    def forward(self, x):
        B = x.shape[0]
        outs = []
        for i in range(4):
            x, H, W = getattr(self, f"patch_embed{i + 1}")(x)
            for blk in getattr(self, f"block{i + 1}"):
                x = blk(x, H, W)
            x = getattr(self, f"norm{i + 1}")(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs


# ---------------------------------------------------------------------------
# SegFormer 解码头
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """逐 stage 的线性投影（1x1 卷积等价），输出通道统一到 embed_dim。"""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.proj(x.flatten(2).transpose(1, 2))


class SegFormerHead(nn.Module):
    def __init__(self, in_channels, embed_dim=256, out_ch=3,
                 norm="batch", dropout=0.1):
        super().__init__()
        c1, c2, c3, c4 = in_channels
        self.linear_c1 = MLP(c1, embed_dim)
        self.linear_c2 = MLP(c2, embed_dim)
        self.linear_c3 = MLP(c3, embed_dim)
        self.linear_c4 = MLP(c4, embed_dim)
        # 名字与 mmseg 的 ConvModule 对齐（linear_fuse.conv / linear_fuse.bn），
        # 这样官方 SegFormer 权重里的解码器投影也能直接加载。
        self.linear_fuse = nn.Sequential()
        self.linear_fuse.add_module("conv", nn.Conv2d(embed_dim * 4, embed_dim,
                                                       kernel_size=1, bias=False))
        self.linear_fuse.add_module("bn", make_norm(norm, embed_dim))
        self.linear_fuse.add_module("act", nn.ReLU(inplace=True))
        self.dropout = nn.Dropout2d(dropout)
        self.linear_pred = nn.Conv2d(embed_dim, out_ch, kernel_size=1)

    def forward(self, features):
        c1, c2, c3, c4 = features
        n, _, h1, w1 = c1.shape
        flat = []
        for feat, proj in zip((c1, c2, c3, c4),
                              (self.linear_c1, self.linear_c2,
                               self.linear_c3, self.linear_c4)):
            y = proj(feat).permute(0, 2, 1).reshape(n, -1, feat.shape[2], feat.shape[3])
            if y.shape[-2:] != (h1, w1):
                y = F.interpolate(y, size=(h1, w1), mode="bilinear",
                                  align_corners=False)
            flat.append(y)
        x = self.linear_fuse(torch.cat(flat, dim=1))
        return self.linear_pred(self.dropout(x))


VARIANTS = {
    # variant: (embed_dims, num_heads, depths, sr_ratios)
    "b0": ((32, 64, 160, 256), (1, 2, 5, 8), (2, 2, 2, 2), (8, 4, 2, 1)),
    "b1": ((64, 128, 320, 512), (1, 2, 5, 8), (2, 2, 2, 2), (8, 4, 2, 1)),
    "b2": ((64, 128, 320, 512), (1, 2, 5, 8), (3, 4, 6, 3), (8, 4, 2, 1)),
    "b3": ((64, 128, 320, 512), (1, 2, 5, 8), (3, 4, 18, 3), (8, 4, 2, 1)),
    "b4": ((64, 128, 320, 512), (1, 2, 5, 8), (3, 8, 27, 3), (8, 4, 2, 1)),
    "b5": ((64, 128, 320, 512), (1, 2, 5, 8), (3, 6, 40, 3), (8, 4, 2, 1)),
}


class SegFormer2D(nn.Module):
    """项目接口包装：与 UNet / UNet++ / AttentionUNet / SwinUNet2D 同签名。"""

    def __init__(self, in_ch: int = 3, out_ch: int = 3,
                 variant: str = "b2", norm: str = "batch",
                 embed_dim: int = 256, dropout: float = 0.1,
                 drop_path_rate: float = 0.1):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"未知 SegFormer 变体 {variant!r}，可选 {sorted(VARIANTS)}")
        dims, heads, depths, sr_ratios = VARIANTS[variant]
        self.variant = variant
        self.backbone = MixVisionTransformer(
            in_chans=in_ch, embed_dims=dims, num_heads=heads,
            depths=depths, sr_ratios=sr_ratios,
            drop_path_rate=drop_path_rate)
        self.decode_head = SegFormerHead(
            in_channels=dims, embed_dim=embed_dim, out_ch=out_ch,
            norm=norm, dropout=dropout)

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.decode_head(feats)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear",
                             align_corners=False)


def _unwrap_state_dict(ckpt):
    if not isinstance(ckpt, dict):
        raise SystemExit("[错误] SegFormer 预训练权重不是字典格式")
    for key in ("state_dict", "model"):
        if isinstance(ckpt.get(key), dict):
            return ckpt[key]
    return ckpt


def load_pretrained_segformer(model, ckpt_path, verbose: bool = True):
    """加载官方 SegFormer/mmseg 权重（可选），返回统计 dict。

    支持两种键名：
      · mmseg 完整权重：`backbone.patch_embed1.*` / `decode_head.*`
      · 裸 MiT 权重：`patch_embed1.*`（自动补 `backbone.` 前缀）
    只加载形状一致的键 —— 分类数不同导致的 `decode_head.linear_pred.*`
    会被跳过。**编码器参数覆盖率低于 90% 直接报错**，避免静默从零训练。
    """
    from pathlib import Path
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise SystemExit(f"[错误] 找不到 SegFormer 预训练权重: {ckpt_path}")
    sd = _unwrap_state_dict(
        torch.load(ckpt_path, map_location="cpu", weights_only=True))
    if sd and all(k.startswith("module.") for k in sd):
        sd = {k[len("module."):]: v for k, v in sd.items()}

    model_sd = model.state_dict()
    to_load, skipped_shape = {}, []
    for k, v in sd.items():
        key = k if k in model_sd else f"backbone.{k}"
        if key not in model_sd:
            continue
        if model_sd[key].shape != v.shape:
            skipped_shape.append(key)
            continue
        to_load[key] = v

    enc_total = [k for k in model_sd if k.startswith("backbone.")]
    enc_loaded = [k for k in enc_total if k in to_load]
    ratio = len(enc_loaded) / max(len(enc_total), 1)
    if ratio < 0.9:
        raise SystemExit(
            f"[错误] SegFormer 预训练权重与编码器不匹配：只命中 "
            f"{len(enc_loaded)}/{len(enc_total)}（{ratio:.0%}），请确认是 MiT-B* 权重")
    model.load_state_dict(to_load, strict=False)
    if verbose:
        n_loaded = sum(model_sd[k].numel() for k in to_load)
        print(f"[SegFormer 预训练] {ckpt_path.name}: 载入 {len(to_load)} 个张量 / "
              f"{n_loaded / 1e6:.2f}M 参数 | 编码器 {len(enc_loaded)}/{len(enc_total)}"
              + (f" | 形状不符跳过 {len(skipped_shape)} 个"
                 if skipped_shape else ""))
    return {"loaded": len(to_load), "skipped_shape": skipped_shape,
            "encoder_ratio": ratio}
