"""ESegFormer（RPT 的 root 分割骨架）的 2D 实现，接口与 common/unet.py 一致。

来源：`Net/RPT/paddleseg/models/esegformer.py` 与
`Net/RPT/paddleseg/models/backbones/emix_transformer.py`。

与标准 SegFormer 的区别只有两处：

1. 骨干多了一个 **spatial_attn** 分支：输入先过 3->32(stride2)、32->64(stride2)、
   64->128(dilation2)、128->256(dilation5) 四层 3x3 卷积 + CBAM，输出 1/4 分辨率、
   256 通道的 c5；
2. 解码头变成 **5 级**（c5 + c1..c4），全部投影到 `embedding_dim`（B2 是 768，
   不是标准 SegFormer 的 256），拼接 5*768 后 1x1 融合，再 1x1 分类。

其余（MiT 的 4 个 stage、overlap patch embedding、sr_ratio 注意力、Mix-FFN、
LayerNorm 位置）与官方 MiT-B* 一致，所以编码器可以直接复用
`common/segformer.py` 的 `MixVisionTransformer`，只新增 spatial_attn 和 5 级解码头。

**预训练权重**：RPT 提供的是 PaddlePaddle 的 `Net/RPT/model.pdparams`（ESegFormer-B2，
2 类 root/background）。用 `tool/convert_esegformer_paddle.py` 转成 PyTorch 后，由
`--esegformer-pretrained` 加载；分类头是 2 类、本项目是 3 通道，`linear_pred` 会被
跳过并重新初始化，骨干/spatial_attn/解码头投影仍然加载。

输入约定：RPT 推理用 `Normalize(mean=0.5,std=0.5)`，即模型输入是 [-1,1]；本项目
数据管线给的是 [0,1]。默认 `input_norm="pm1"` 在模型内部做 `x*2-1`，与 RPT 对齐；
从零训练也保持同一约定，避免换预训练时输入尺度不一致。
"""
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.segformer import MLP, MixVisionTransformer, VARIANTS
from common.unet import make_norm

EMBED_DIM = {
    "b0": 256, "b1": 256,
    "b2": 768, "b3": 768, "b4": 768, "b5": 768,
}


class ChannelAttention(nn.Module):
    """CBAM 的通道注意力：avg/max pool -> 共享 1x1 MLP -> sigmoid。"""

    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM 的空间注意力：通道 avg/max 拼成 2 通道，7x7 conv -> sigmoid。"""

    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.amax(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(attn))


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


class EMixVisionTransformer(MixVisionTransformer):
    """MiT-B* + RPT 的 spatial_attn 分支；返回 [c5, c1, c2, c3, c4]。"""

    def __init__(self, in_chans=3, embed_dims=(64, 128, 320, 512),
                 num_heads=(1, 2, 5, 8), depths=(3, 4, 6, 3),
                 sr_ratios=(8, 4, 2, 1), drop_path_rate=0.1,
                 norm="batch",
                 norm_layer=nn.LayerNorm):
        super().__init__(
            in_chans=in_chans, embed_dims=embed_dims, num_heads=num_heads,
            depths=depths, sr_ratios=sr_ratios,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer)
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            make_norm(norm, 32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            make_norm(norm, 64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2,
                      dilation=2),
            make_norm(norm, 128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=5,
                      dilation=5),
            make_norm(norm, 256),
            nn.ReLU(),
            CBAM(256),
        )

    def forward(self, x):
        c5 = self.spatial_attn(x)
        feats = super().forward(x)
        return [c5] + feats


class ESegFormerHead(nn.Module):
    """RPT 的 5 级解码头（c5 + c1..c4）。"""

    def __init__(self, in_channels, embed_dim=768, out_ch=3,
                 norm="batch", dropout=0.1):
        super().__init__()
        c5, c1, c2, c3, c4 = in_channels
        self.linear_c5 = MLP(c5, embed_dim)
        self.linear_c4 = MLP(c4, embed_dim)
        self.linear_c3 = MLP(c3, embed_dim)
        self.linear_c2 = MLP(c2, embed_dim)
        self.linear_c1 = MLP(c1, embed_dim)
        # 名字与 RPT 的 ConvBNReLU 对齐：linear_fuse.conv / linear_fuse.bn
        self.linear_fuse = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(embed_dim * 5, embed_dim, 1, bias=False)),
            ("bn", make_norm(norm, embed_dim)),
            ("act", nn.ReLU(inplace=True)),
        ]))
        self.dropout = nn.Dropout2d(dropout)
        self.linear_pred = nn.Conv2d(embed_dim, out_ch, 1)

    def forward(self, features):
        c5, c1, c2, c3, c4 = features
        n, _, h1, w1 = c1.shape
        flat = []
        # 拼接顺序必须与 RPT 一致：c5, c4, c3, c2, c1
        for feat, proj in zip((c5, c4, c3, c2, c1),
                              (self.linear_c5, self.linear_c4,
                               self.linear_c3, self.linear_c2,
                               self.linear_c1)):
            y = proj(feat).permute(0, 2, 1).reshape(
                n, -1, feat.shape[2], feat.shape[3])
            if y.shape[-2:] != (h1, w1):
                y = F.interpolate(y, size=(h1, w1), mode="bilinear",
                                  align_corners=False)
            flat.append(y)
        x = self.linear_fuse(torch.cat(flat, dim=1))
        return self.linear_pred(self.dropout(x))


class ESegFormer2D(nn.Module):
    """项目接口包装：`--arch esegformer` 使用。"""

    def __init__(self, in_ch: int = 3, out_ch: int = 3,
                 variant: str = "b2", norm: str = "batch",
                 dropout: float = 0.1, drop_path_rate: float = 0.1,
                 input_norm: str = "pm1", freeze_backbone: bool = False):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(
                f"未知 ESegFormer 变体 {variant!r}，可选 {sorted(VARIANTS)}")
        dims, heads, depths, sr_ratios = VARIANTS[variant]
        self.variant = variant
        self.input_norm = input_norm
        self.backbone = EMixVisionTransformer(
            in_chans=in_ch, embed_dims=dims, num_heads=heads,
            depths=depths, sr_ratios=sr_ratios,
            drop_path_rate=drop_path_rate, norm=norm,
            # RPT 的 EMixVisionTransformer_B*_ 显式传 epsilon=1e-6
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.decode_head = ESegFormerHead(
            in_channels=(256,) + tuple(dims),
            embed_dim=EMBED_DIM[variant], out_ch=out_ch,
            norm=norm, dropout=dropout)
        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        """复现 RPT 的 freeze_exceptEAM。

        RPT 冻结骨干和 linear_c1..c4，只训练 spatial_attn、linear_c5、
        linear_fuse、linear_pred。这里保持一致；不调用本方法时全参数微调。
        """
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.spatial_attn.parameters():
            p.requires_grad = True
        for name, p in self.decode_head.named_parameters():
            if name.startswith(("linear_c1.", "linear_c2.",
                                "linear_c3.", "linear_c4.")):
                p.requires_grad = False

    def forward(self, x):
        if self.input_norm == "pm1":
            x = x * 2.0 - 1.0
        elif self.input_norm != "none":
            raise ValueError(f"未知 input_norm {self.input_norm!r}")
        feats = self.backbone(x)
        logits = self.decode_head(feats)
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear",
                             align_corners=False)


def load_pretrained_esegformer(model, ckpt_path, verbose: bool = True):
    """加载 `tool/convert_esegformer_paddle.py` 转好的 PyTorch 权重。

    只加载 key/shape 都匹配的张量；2 类预训练头 `decode_head.linear_pred.*` 在
    3 通道模型里形状不符，会自动跳过并保持随机初始化。骨干覆盖率 <90% 直接报错，
    避免「以为用了预训练、其实骨干没加载」。
    """
    from pathlib import Path
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise SystemExit(f"[错误] 找不到 ESegFormer 预训练权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    if not isinstance(sd, dict):
        raise SystemExit(f"[错误] {ckpt_path.name} 不是 state_dict 格式")

    model_sd = model.state_dict()
    to_load, skipped = {}, []
    for k, v in sd.items():
        if k not in model_sd:
            skipped.append(k)
            continue
        if model_sd[k].shape != v.shape:
            skipped.append(k)
            continue
        to_load[k] = v

    enc_total = [k for k in model_sd if k.startswith("backbone.")]
    enc_loaded = [k for k in enc_total if k in to_load]
    ratio = len(enc_loaded) / max(len(enc_total), 1)
    if ratio < 0.9:
        raise SystemExit(
            f"[错误] ESegFormer 预训练权重与骨干不匹配：只命中 "
            f"{len(enc_loaded)}/{len(enc_total)}（{ratio:.0%}），请确认是 B2 权重")
    model.load_state_dict(to_load, strict=False)
    if verbose:
        n_loaded = sum(model_sd[k].numel() for k in to_load)
        print(f"[ESegFormer 预训练] {ckpt_path.name}: 载入 {len(to_load)} 个张量 / "
              f"{n_loaded / 1e6:.2f}M 参数 | 骨干 {len(enc_loaded)}/{len(enc_total)}"
              + (f" | 跳过 {len(skipped)} 个（含 2 类分类头）"
                 if skipped else ""))
    return {"loaded": len(to_load), "skipped": skipped,
            "backbone_ratio": ratio}
