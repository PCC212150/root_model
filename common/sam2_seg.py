"""SAM 2（Ravi et al., 2024）作为分割骨架。

**与 common/sam_seg.py（SAM 1）同样的定位**：不是 promptable 的零样本 SAM 2，
而是「SAM 2 预训练 Hiera 图像编码器 + 监督解码头」。原版 SAM 2 需要点/框/掩码
提示，输出类无关的单物体掩码；本项目是三通道逐像素分割（root / stem / check），
没有提示来源（检查框本身就是要预测的输出）。因此：

    image -> Hiera 编码器（默认冻结，SAM 2.1 预训练）
          -> FPN neck，输出 1/4、1/8、1/16 三级特征（各 256 通道）
          -> 带跳跃连接的上采样解码器
          -> 1x1 输出 3 通道 logits

这样它能和 U-Net / UNet++ / Attention U-Net / Swin-Unet / SegFormer /
TransUNet / SAM(SAM1) 用同一套训练与评测流程直接对比。

代码 vendored 自 `Net/sam2-main/sam2/modeling/backbones/`（Apache-2.0）：
1. 去掉 `iopath`（只用于 `weights_path` 分支）与 hydra/omegaconf；
2. `_get_pos_embed` 的 window embedding 由 `tile` 改为「向上取整再裁切」，
   输入边长不必是 window(8) 的整数倍也能前向；
3. 去掉 prompt encoder / mask decoder / memory，只保留 trunk+neck；
4. 加监督解码头与项目接口 `SAM2Seg2D(in_ch/out_ch/norm/pretrained)`；
5. 输入按 SAM 2 的 ImageNet mean/std 归一化（项目数据是 [0,1]）。

可选权重：SAM 2.1 Hiera-Tiny（149MB）/ Small / Base+ / Large，官方地址见
config.SAM2_PRETRAINED_*。默认 tiny（约 39M 编码器参数）。
"""
import math
from functools import partial
from pathlib import Path
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.unet import DoubleConv, make_norm


# ---------------------------------------------------------------------------
# 基础组件（vendored from sam2/modeling/sam2_utils.py & backbones/utils.py）
# ---------------------------------------------------------------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0, scale_by_keep=True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 activation=nn.ReLU, sigmoid_output=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, kernel_size=(7, 7), stride=(4, 4), padding=(3, 3),
                 in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding)

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size,
               Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(B, Hp // window_size, Wp // window_size,
                        window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


# ---------------------------------------------------------------------------
# Hiera 编码器（vendored from backbones/hieradet.py）
# ---------------------------------------------------------------------------
def do_pool(x, pool, norm=None):
    if pool is None:
        return x
    x = x.permute(0, 3, 1, 2)
    x = pool(x)
    x = x.permute(0, 2, 3, 1)
    if norm:
        x = norm(x)
    return x


class MultiScaleAttention(nn.Module):
    def __init__(self, dim, dim_out, num_heads, q_pool=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.q_pool = q_pool
        self.qkv = nn.Linear(dim, dim_out * 3)
        self.proj = nn.Linear(dim_out, dim_out)

    def forward(self, x):
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1)
        q, k, v = torch.unbind(qkv, 2)
        if self.q_pool:
            q = do_pool(q.reshape(B, H, W, -1), self.q_pool)
            H, W = q.shape[1:3]
            q = q.reshape(B, H * W, self.num_heads, -1)
        x = F.scaled_dot_product_attention(q.transpose(1, 2),
                                           k.transpose(1, 2),
                                           v.transpose(1, 2))
        x = x.transpose(1, 2).reshape(B, H, W, -1)
        return self.proj(x)


class MultiScaleBlock(nn.Module):
    def __init__(self, dim, dim_out, num_heads, mlp_ratio=4.0, drop_path=0.0,
                 norm_layer: Union[nn.Module, str] = "LayerNorm",
                 q_stride: Tuple[int, int] = None, act_layer=nn.GELU,
                 window_size=0):
        super().__init__()
        if isinstance(norm_layer, str):
            norm_layer = partial(getattr(nn, norm_layer), eps=1e-6)
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        self.window_size = window_size
        self.pool, self.q_stride = None, q_stride
        if self.q_stride:
            self.pool = nn.MaxPool2d(kernel_size=q_stride, stride=q_stride,
                                     ceil_mode=False)
        self.attn = MultiScaleAttention(dim, dim_out, num_heads, q_pool=self.pool)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim_out)
        self.mlp = MLP(dim_out, int(dim_out * mlp_ratio), dim_out,
                       num_layers=2, activation=act_layer)
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.dim != self.dim_out:
            shortcut = do_pool(self.proj(x), self.pool)
        window_size = self.window_size
        if window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, window_size)
        x = self.attn(x)
        if self.q_stride:
            window_size = self.window_size // self.q_stride[0]
            H, W = shortcut.shape[1:3]
            pad_h = (window_size - H % window_size) % window_size
            pad_w = (window_size - W % window_size) % window_size
            pad_hw = (H + pad_h, W + pad_w)
        if self.window_size > 0:
            x = window_unpartition(x, window_size, pad_hw, (H, W))
        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))


class Hiera(nn.Module):
    def __init__(self, embed_dim=96, num_heads=1, drop_path_rate=0.0, q_pool=3,
                 q_stride=(2, 2), stages=(2, 3, 16, 3), dim_mul=2.0,
                 head_mul=2.0, window_pos_embed_bkg_spatial_size=(7, 7),
                 window_spec=(8, 4, 14, 7), global_att_blocks=(12, 16, 20),
                 return_interm_layers=True):
        super().__init__()
        assert len(stages) == len(window_spec)
        self.window_spec = window_spec
        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.global_att_blocks = global_att_blocks
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(torch.zeros(
            1, embed_dim, *self.window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = nn.Parameter(torch.zeros(
            1, embed_dim, self.window_spec[0], self.window_spec[0]))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        cur_stage = 1
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dim_out = embed_dim
            window_size = self.window_spec[cur_stage - 1]
            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size
            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1
            self.blocks.append(MultiScaleBlock(
                dim=embed_dim, dim_out=dim_out, num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size))
            embed_dim = dim_out
        self.channel_list = ([self.blocks[i].dim_out for i in self.stage_ends[::-1]]
                             if return_interm_layers else [self.blocks[-1].dim_out])

    def _get_pos_embed(self, hw):
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        # [patch] 官方用 tile(h // 8, w // 8)，输入不是 8 的倍数时会短一截。
        # 这里向上取整再裁到 (h, w)，任意尺寸都能前向。
        rep_h = (h + window_embed.shape[2] - 1) // window_embed.shape[2]
        rep_w = (w + window_embed.shape[3] - 1) // window_embed.shape[3]
        window = window_embed.tile([1, 1, rep_h, rep_w])[:, :, :h, :w]
        pos_embed = pos_embed + window
        return pos_embed.permute(0, 2, 3, 1)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self._get_pos_embed(x.shape[1:3])
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (
                    i in self.stage_ends and self.return_interm_layers):
                outputs.append(x.permute(0, 3, 1, 2))
        return outputs


# ---------------------------------------------------------------------------
# FPN neck（只保留卷积部分；位置编码用于 prompt decoder，本项目不需要）
# ---------------------------------------------------------------------------
class FpnNeck(nn.Module):
    def __init__(self, d_model, backbone_channel_list, kernel_size=1, stride=1,
                 padding=0, fpn_interp_model="nearest", fuse_type="sum",
                 fpn_top_down_levels: Optional[List[int]] = None):
        super().__init__()
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        self.d_model = d_model
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module("conv", nn.Conv2d(
                dim, d_model, kernel_size=kernel_size, stride=stride,
                padding=padding))
            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type
        if fpn_top_down_levels is None:
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs):
        out = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        prev = None
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev is not None:
                top_down = F.interpolate(
                    prev.to(dtype=torch.float32), scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(None if self.fpn_interp_model == "nearest"
                                   else False), antialias=False)
                prev = lateral + top_down
                if self.fuse_type == "avg":
                    prev /= 2
            else:
                prev = lateral
            out[i] = prev
        return out


class SAM2ImageEncoder(nn.Module):
    def __init__(self, trunk, neck, scalp=1):
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert trunk.channel_list == neck.backbone_channel_list, \
            f"trunk/neck 通道不匹配: {trunk.channel_list} vs {neck.backbone_channel_list}"

    def forward(self, sample):
        features = self.neck(self.trunk(sample))
        if self.scalp > 0:
            features = features[:-self.scalp]
        return features            # [1/4, 1/8, 1/16]，各 d_model 通道


# ---------------------------------------------------------------------------
# 监督解码头（带 FPN 跳跃连接）
# ---------------------------------------------------------------------------
class SAM2Decoder(nn.Module):
    def __init__(self, in_ch=256, out_ch=3, norm="batch"):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_ch, 128, 2, 2)
        self.conv1 = DoubleConv(128 + in_ch, 128, norm=norm)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv2 = DoubleConv(64 + in_ch, 64, norm=norm)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv3 = DoubleConv(32, 32, norm=norm)
        self.up4 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.conv4 = DoubleConv(16, 16, norm=norm)
        self.out = nn.Conv2d(16, out_ch, kernel_size=1)

    def forward(self, features):
        f4, f8, f16 = features          # 1/4, 1/8, 1/16
        x = self.up1(f16)
        if x.shape[-2:] != f8.shape[-2:]:
            x = F.interpolate(x, size=f8.shape[-2:], mode="bilinear",
                              align_corners=False)
        x = self.conv1(torch.cat([x, f8], dim=1))
        x = self.up2(x)
        if x.shape[-2:] != f4.shape[-2:]:
            x = F.interpolate(x, size=f4.shape[-2:], mode="bilinear",
                              align_corners=False)
        x = self.conv2(torch.cat([x, f4], dim=1))
        x = self.conv4(self.up4(self.conv3(self.up3(x))))
        return self.out(x)


VARIANTS = {
    # variant: (embed_dim, num_heads, stages, global_att_blocks)
    "tiny": (96, 1, (1, 2, 7, 2), (5, 7, 9)),
    "small": (96, 1, (1, 2, 11, 2), (7, 10, 13)),
    "base_plus": (112, 2, (2, 3, 16, 3), (12, 16, 20)),
    "large": (144, 2, (2, 6, 36, 4), (23, 33, 43)),
}


class SAM2Seg2D(nn.Module):
    """项目接口包装：SAM 2.1 Hiera 编码器（默认冻结）+ 监督解码头。"""

    def __init__(self, in_ch: int = 3, out_ch: int = 3, variant: str = "tiny",
                 norm: str = "batch", pretrained=None,
                 freeze_encoder: bool = True):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"未知 SAM2 变体 {variant!r}，可选 {sorted(VARIANTS)}")
        embed_dim, num_heads, stages, global_att = VARIANTS[variant]
        self.variant = variant
        trunk = Hiera(embed_dim=embed_dim, num_heads=num_heads,
                      stages=stages, global_att_blocks=global_att,
                      window_pos_embed_bkg_spatial_size=(7, 7),
                      drop_path_rate=0.0)
        neck = FpnNeck(
            d_model=256,
            backbone_channel_list=[embed_dim * 8, embed_dim * 4,
                                   embed_dim * 2, embed_dim],
            fpn_top_down_levels=[2, 3], fpn_interp_model="nearest")
        self.image_encoder = SAM2ImageEncoder(trunk, neck, scalp=1)
        self.decoder = SAM2Decoder(in_ch=256, out_ch=out_ch, norm=norm)
        self.register_buffer("pixel_mean",
                             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("pixel_std",
                             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if pretrained is not None:
            load_pretrained_sam2(self, pretrained)
        self.freeze_encoder = bool(freeze_encoder)
        if self.freeze_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.image_encoder.eval()
        return self

    def forward(self, x):
        x = (x - self.pixel_mean) / self.pixel_std
        if self.freeze_encoder:
            with torch.no_grad():
                feats = self.image_encoder(x)
        else:
            feats = self.image_encoder(x)
        logits = self.decoder(feats)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear",
                                   align_corners=False)
        return logits


def load_pretrained_sam2(model, ckpt_path, verbose: bool = True):
    """加载官方 SAM 2.1 checkpoint 里的 **trunk + neck**。

    官方文件是 `{"model": state_dict, ...}`，键名 `trunk.*` / `neck.*`；
    prompt/mask decoder、memory 等全部忽略。trunk/neck 键必须全部命中。
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        import config
        raise SystemExit(
            f"[错误] 找不到 SAM 2 预训练权重: {ckpt_path}\n"
            f"       下载: curl -L -o \"{ckpt_path}\" {config.SAM2_PRETRAINED_URL}\n"
            f"       或加 --sam2-pretrained none 从零训练")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    # 官方键名是 image_encoder.trunk.* / image_encoder.neck.*；也兼容裸 trunk./neck.
    enc_sd = {}
    for k, v in sd.items():
        if k.startswith("image_encoder."):
            enc_sd[k[len("image_encoder."):]] = v
        elif k.startswith(("trunk.", "neck.")):
            enc_sd[k] = v
    missing, unexpected = model.image_encoder.load_state_dict(enc_sd,
                                                              strict=False)
    if missing:
        raise SystemExit(
            f"[错误] SAM 2 权重与 Hiera/neck 不匹配：缺少 {len(missing)} 个键，"
            f"例如 {missing[:5]}")
    if verbose:
        n = sum(p.numel() for p in model.image_encoder.parameters())
        print(f"[SAM2 预训练] {ckpt_path.name}: 编码器 {len(enc_sd)} 个张量 / "
              f"{n / 1e6:.2f}M 参数 | 忽略 prompt/mask/memory")
    return {"encoder_tensors": len(enc_sd), "unexpected": len(unexpected)}
