"""SAM（Segment Anything, Kirillov et al., ICCV 2023）作为分割骨架。

**这不是 promptable 的零样本 SAM**。原始 SAM 需要点/框/掩码提示，输出的是
「一个提示对应一个物体」的类无关掩码；本项目的交付是**三通道逐像素分割**
（root / stem / check），没有可用的提示来源（检查框本身就是模型要预测的输出）。
因此这里采用「SAM 预训练图像编码器 + 监督解码头」的用法：

    image -> SAM ViT 图像编码器（默认冻结，ImageNet-21k + SA-1B 预训练）
          -> 1/16 特征图（256 通道）
          -> 4 次上采样卷积解码器
          -> 1x1 输出 3 通道 logits

这样它才能和 U-Net / UNet++ / Attention U-Net / Swin-Unet / SegFormer /
TransUNet 用同一套训练、验证、测试流程直接对比。**只有解码头参与训练**
（`--sam-finetune-encoder` 可解冻编码器，但 71 张图不建议）。

代码 vendored 自 `Net/segment-anything-main/segment_anything/modeling/`
（Apache-2.0），改动：
1. `ImageEncoderViT.forward` 的绝对位置编码按输入网格双线性插值 —— 官方只支持
   1024x1024 方形；这样 1024x1536 等非方形/任意尺寸可直接前向；
2. 去掉 prompt encoder / mask decoder（本项目不用提示）；
3. 加监督解码头与项目接口 `SAMSeg2D(in_ch/out_ch/norm/pretrained)`；
4. 输入是项目统一的 [0,1] 张量，内部按 SAM 的 pixel_mean/std 归一化。

可选权重：`sam_vit_b_01ec64.pth`（375MB，官方 ViT-B）。ViT-L/H 也支持，
但 3×3090 上 ViT-H 的 636M 参数 + 1024x1536 特征图很吃紧，默认 vit_b。
"""
import math
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.unet import DoubleConv, make_norm


# ---------------------------------------------------------------------------
# SAM 图像编码器（vendored）
# ---------------------------------------------------------------------------
class MLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int, act: Type[nn.Module] = nn.GELU):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class PatchEmbed(nn.Module):
    def __init__(self, kernel_size=(16, 16), stride=(16, 16), padding=(0, 0),
                 in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding)

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)      # B H W C


def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size,
               window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


def get_rel_pos(q_size, k_size, rel_pos):
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist, mode="linear")
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    attn = (attn.view(B, q_h, q_w, k_h, k_w)
            + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]).view(
        B, q_h * q_w, k_h * k_w)
    return attn


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, use_rel_pos=False,
                 rel_pos_zero_init=True, input_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            assert input_size is not None
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x):
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w,
                                          (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(
            0, 2, 3, 1, 4).reshape(B, H, W, -1)
        return self.proj(x)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU, use_rel_pos=False,
                 rel_pos_zero_init=True, window_size=0, input_size=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size))
        self.norm2 = norm_layer(dim)
        self.mlp = MLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio),
                            act=act_layer)
        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class ImageEncoderViT(nn.Module):
    def __init__(self, img_size=1024, patch_size=16, in_chans=3, embed_dim=768,
                 depth=12, num_heads=12, mlp_ratio=4.0, out_chans=256,
                 qkv_bias=True, norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 use_abs_pos=True, use_rel_pos=False, rel_pos_zero_init=True,
                 window_size=0, global_attn_indexes=()):
        super().__init__()
        self.img_size = img_size
        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size), stride=(patch_size, patch_size),
            in_chans=in_chans, embed_dim=embed_dim)
        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            self.pos_embed = nn.Parameter(torch.zeros(
                1, img_size // patch_size, img_size // patch_size, embed_dim))
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, norm_layer=norm_layer, act_layer=act_layer,
                use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size)))
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans))

    def forward(self, x):
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            # [patch] 官方只支持 1024x1024（pos_embed 固定 64x64）。这里按输入网格
            # 做双线性插值，非方形/任意尺寸都能前向。
            if x.shape[1:3] != self.pos_embed.shape[1:3]:
                pos = self.pos_embed.permute(0, 3, 1, 2)
                pos = F.interpolate(pos, size=x.shape[1:3], mode="bilinear",
                                    align_corners=False)
                x = x + pos.permute(0, 2, 3, 1)
            else:
                x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.neck(x.permute(0, 3, 1, 2))


# ---------------------------------------------------------------------------
# 监督解码头
# ---------------------------------------------------------------------------
class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, norm="batch"):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch // 2, out_ch, norm=norm)

    def forward(self, x):
        return self.conv(self.up(x))


class SAMDecoder(nn.Module):
    """1/16 特征图 -> 4 次上采样 -> 原分辨率三通道 logits。"""

    def __init__(self, in_ch=256, out_ch=3, norm="batch"):
        super().__init__()
        self.up1 = UpBlock(in_ch, in_ch // 2, norm=norm)        # 256 -> 128
        self.up2 = UpBlock(in_ch // 2, in_ch // 4, norm=norm)   # 128 -> 64
        self.up3 = UpBlock(in_ch // 4, in_ch // 8, norm=norm)   # 64 -> 32
        self.up4 = UpBlock(in_ch // 8, in_ch // 16, norm=norm)  # 32 -> 16
        self.out = nn.Conv2d(in_ch // 16, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.up4(self.up3(self.up2(self.up1(x))))
        return self.out(x)


VARIANTS = {
    # variant: (embed_dim, depth, num_heads, global_attn_indexes)
    "vit_b": (768, 12, 12, (2, 5, 8, 11)),
    "vit_l": (1024, 24, 16, (5, 11, 17, 23)),
    "vit_h": (1280, 32, 16, (7, 15, 23, 31)),
}


class SAMSeg2D(nn.Module):
    """项目接口包装：SAM 图像编码器（默认冻结）+ 监督解码头。"""

    def __init__(self, in_ch: int = 3, out_ch: int = 3,
                 variant: str = "vit_b", norm: str = "batch",
                 pretrained=None, freeze_encoder: bool = True,
                 norm_mode: str = "sam"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"未知 SAM 变体 {variant!r}，可选 {sorted(VARIANTS)}")
        if norm_mode not in ("sam", "medsam"):
            raise ValueError(f"未知归一化方式 {norm_mode!r}，可选 sam / medsam")
        self.norm_mode = norm_mode
        embed_dim, depth, num_heads, global_idx = VARIANTS[variant]
        self.variant = variant
        self.image_encoder = ImageEncoderViT(
            img_size=1024, patch_size=16, in_chans=in_ch, embed_dim=embed_dim,
            depth=depth, num_heads=num_heads, mlp_ratio=4.0, out_chans=256,
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU, use_abs_pos=True, use_rel_pos=True,
            rel_pos_zero_init=True, window_size=14,
            global_attn_indexes=global_idx)
        self.decoder = SAMDecoder(in_ch=256, out_ch=out_ch, norm=norm)
        # SAM 的预处理：像素值 0~255 上减均值除标准差。项目数据是 [0,1]。
        self.register_buffer("pixel_mean",
                             torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std",
                             torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1))
        if norm_mode == "medsam":
            # MedSAM 官方预处理：缩放到 1024 后按图做 min-max 到 [0,1]，
            # 不再减 ImageNet 均值 / 除标准差（见 Net/MedSAM/MedSAM_Inference.py）
            self.pixel_mean.zero_()
            self.pixel_std.fill_(1.0)
        if pretrained is not None:
            load_pretrained_sam(self, pretrained)
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # 冻结时编码器保持 eval：即使外层 train() 也不让 LayerNorm 行为变化
        # （ViT 只有 LayerNorm，没有 dropout；这里主要是语义上的明确）
        if self.freeze_encoder:
            self.image_encoder.eval()
        return self

    def forward(self, x):
        if self.norm_mode == "medsam":
            # 项目输入已是 [0,1]；按图 min-max，与 MedSAM 训练/推理的预处理一致
            mn = x.amin(dim=(1, 2, 3), keepdim=True)
            mx = x.amax(dim=(1, 2, 3), keepdim=True)
            x = (x - mn) / (mx - mn).clamp_min(1e-8)
        else:
            x = (x * 255.0 - self.pixel_mean) / self.pixel_std
        if self.freeze_encoder:
            with torch.no_grad():
                feat = self.image_encoder(x)
        else:
            feat = self.image_encoder(x)
        logits = self.decoder(feat)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear",
                                   align_corners=False)
        return logits


class MedSAMSeg2D(SAMSeg2D):
    """MedSAM（医学图像上微调的 SAM ViT-B，Ma et al., Nat. Commun. 2024）。

    与 ``SAMSeg2D`` 的唯一区别是预处理和默认权重：

    - 输入按图 min-max 到 [0,1]（MedSAM 官方做法），不减 ImageNet 均值；
    - 权重用官方 ``medsam_vit_b.pth``（同样是 SAM 格式，只取
      ``image_encoder.*``）。

    仍然只训练监督解码头；MedSAM 原版的 prompt encoder / mask decoder 不使用，
    原因见模块开头（本项目没有可用的框/点提示来源）。
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 3, norm: str = "batch",
                 pretrained=None, freeze_encoder: bool = True):
        super().__init__(in_ch=in_ch, out_ch=out_ch, variant="vit_b", norm=norm,
                         pretrained=None, freeze_encoder=freeze_encoder,
                         norm_mode="medsam")
        if pretrained is not None:
            load_pretrained_medsam(self, pretrained)


def _load_sam_encoder(model, ckpt_path, label: str, url: str,
                      verbose: bool = True):
    """从 SAM 格式 checkpoint 里取图像编码器权重（MedSAM 也是这个格式）。"""
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise SystemExit(
            f"[错误] 找不到 {label} 预训练权重: {ckpt_path}\n"
            f"       下载: curl -L -o \"{ckpt_path}\" \"{url}\"\n"
            f"       或加 --{label.lower()}-pretrained none 从零训练")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and isinstance(sd.get("model"), dict):
        sd = sd["model"]
    if any(k.startswith("image_encoder.") for k in sd):
        enc = {k[len("image_encoder."):]: v for k, v in sd.items()
               if k.startswith("image_encoder.")}
    else:
        enc = dict(sd)

    missing, unexpected = model.image_encoder.load_state_dict(enc, strict=False)
    if missing:
        raise SystemExit(
            f"[错误] {label} 权重与图像编码器不匹配：缺少 {len(missing)} 个键，"
            f"例如 {missing[:5]}")
    if verbose:
        n = sum(p.numel() for p in model.image_encoder.parameters())
        print(f"[{label} 预训练] {ckpt_path.name}: 图像编码器 {len(enc)} 个张量 / "
              f"{n / 1e6:.2f}M 参数 | 忽略 prompt/mask decoder")
    return {"encoder_tensors": len(enc), "unexpected": len(unexpected)}


def load_pretrained_sam(model, ckpt_path, verbose: bool = True):
    """加载官方 SAM checkpoint 里的图像编码器权重。

    官方文件是完整 `Sam.state_dict()`，键名以 `image_encoder.` 开头；
    也兼容裸 `ImageEncoderViT` 权重（没有前缀）。prompt/mask decoder 忽略。
    编码器键必须全部命中，否则报错（避免张冠李戴）。
    """
    import config
    return _load_sam_encoder(model, ckpt_path, "SAM",
                             config.SAM_PRETRAINED_URL, verbose)


def load_pretrained_medsam(model, ckpt_path, verbose: bool = True):
    """加载官方 MedSAM `medsam_vit_b.pth` 里的图像编码器权重。"""
    import config
    return _load_sam_encoder(model, ckpt_path, "MedSAM",
                             config.MEDSAM_PRETRAINED_URL, verbose)
