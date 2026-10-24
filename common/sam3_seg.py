"""SAM 3（Meta, 2025）视觉骨干 + 监督解码头。

**定位与 SAM 1/2 相同**：不是 promptable 的零样本文本/点/框 SAM 3。
SAM 3 原版是「文本/几何提示 -> 检测器 + 分割头」的开放词汇分割模型；本项目
是三通道逐像素分割（root / stem / check），没有提示来源。因此只取它的
**预训练视觉骨干**：

    image -> resize 1008x1008（官方 SAM 3 预处理）
          -> ViT-L/14 (32 层) + ViTDet neck
          -> 4 个尺度（1/3.5、1/7、1/14、1/28，各 256 通道）
          -> 带跳跃连接的上采样解码器
          -> 1x1 输出 3 通道 logits -> 双线性上采样回输入尺寸

ViT 默认冻结，只训练解码头。ViT-L 约 4.4 亿参数，从零训练 71 张图没有意义，
**必须使用官方预训练权重**。

代码 vendored 自 `Net/sam3/sam3/model/vitdet.py` 与 `necks.py`（Apache-2.0）：
1. 去掉官方包的 `NestedTensor` / `AttentionType` / `LayerScale` / `perflib`
   / `sam.rope` 依赖，相关小件内联到 `common/sam3_vitdet.py`；
2. 只保留 `Sam3DualViTDetNeck` 的 SAM3 一支（不含 SAM2 dual neck）；
3. 加监督解码头与项目接口 `SAM3Seg2D(in_ch/out_ch/norm/pretrained)`；
4. 输入按 SAM3 官方预处理：缩放到 1008x1008，归一化到 [-1, 1]。

权重：ModelScope 的 `facebook/sam3`（约 3.45GB，国内可下）里的
`detector.backbone.visual.trunk.*` / `...convs.*`。官方 HF 仓库是 gated。
"""
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.sam3_vitdet import ViT
from common.unet import DoubleConv, make_norm


# ---------------------------------------------------------------------------
# ViTDet neck（vendored from sam3/model/necks.py，只保留 SAM3 一支）
# ---------------------------------------------------------------------------
class Sam3DualViTDetNeck(nn.Module):
    def __init__(self, trunk, d_model, scale_factors=(4.0, 2.0, 1.0, 0.5)):
        super().__init__()
        self.trunk = trunk
        self.scale_factors = scale_factors
        self.convs = nn.ModuleList()
        dim = self.trunk.channel_list[-1]
        for scale in scale_factors:
            current = nn.Sequential()
            if scale == 4.0:
                current.add_module("dconv_2x2_0",
                                   nn.ConvTranspose2d(dim, dim // 2, 2, 2))
                current.add_module("gelu", nn.GELU())
                current.add_module("dconv_2x2_1",
                                   nn.ConvTranspose2d(dim // 2, dim // 4, 2, 2))
                out_dim = dim // 4
            elif scale == 2.0:
                current.add_module("dconv_2x2",
                                   nn.ConvTranspose2d(dim, dim // 2, 2, 2))
                out_dim = dim // 2
            elif scale == 1.0:
                out_dim = dim
            elif scale == 0.5:
                current.add_module("maxpool_2x2", nn.MaxPool2d(2, 2))
                out_dim = dim
            else:
                raise NotImplementedError(f"scale_factor={scale}")
            current.add_module("conv_1x1",
                               nn.Conv2d(out_dim, d_model, 1, bias=True))
            current.add_module("conv_3x3",
                               nn.Conv2d(d_model, d_model, 3, padding=1, bias=True))
            self.convs.append(current)

    def forward(self, tensor):
        xs = self.trunk(tensor)
        x = xs[-1]                 # 1/14 特征图
        return [conv(x) for conv in self.convs]   # 4 个尺度


class SAM3ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        trunk = ViT(
            img_size=1008, pretrain_img_size=336, patch_size=14,
            embed_dim=1024, depth=32, num_heads=16, mlp_ratio=4.625,
            norm_layer="LayerNorm", drop_path_rate=0.1, qkv_bias=True,
            use_abs_pos=True, tile_abs_pos=True,
            global_att_blocks=(7, 15, 23, 31), rel_pos_blocks=(),
            use_rope=True, use_interp_rope=True, window_size=24,
            pretrain_use_cls_token=True, retain_cls_token=False,
            ln_pre=True, ln_post=False, return_interm_layers=False,
            bias_patch_embed=False)
        # **只挂一份**：trunk 作为 neck 的子模块保存（neck.trunk.*），
        # 避免同一个 ViT 在 state_dict 里出现 trunk.* 与 neck.trunk.* 两份。
        self.neck = Sam3DualViTDetNeck(
            trunk=trunk, d_model=256,
            scale_factors=(4.0, 2.0, 1.0, 0.5))

    def forward(self, x):
        return self.neck(x)


# ---------------------------------------------------------------------------
# 解码头
# ---------------------------------------------------------------------------
class SAM3Decoder(nn.Module):
    """4 个 neck 尺度 -> 跳跃上采样 -> 3 通道 logits（1008x1008）。"""

    def __init__(self, in_ch=256, out_ch=3, norm="batch"):
        super().__init__()
        self.up1 = nn.ConvTranspose2d(in_ch, 128, 2, 2)
        self.conv1 = DoubleConv(128 + in_ch, 128, norm=norm)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.conv2 = DoubleConv(64 + in_ch, 64, norm=norm)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.conv3 = DoubleConv(32 + in_ch, 32, norm=norm)
        self.out = nn.Conv2d(32, out_ch, kernel_size=1)

    def forward(self, feats):
        f0, f1, f2, f3 = feats        # 1/3.5, 1/7, 1/14, 1/28
        x = self.up1(f3)
        if x.shape[-2:] != f2.shape[-2:]:
            x = F.interpolate(x, size=f2.shape[-2:], mode="bilinear",
                              align_corners=False)
        x = self.conv1(torch.cat([x, f2], dim=1))
        x = self.up2(x)
        if x.shape[-2:] != f1.shape[-2:]:
            x = F.interpolate(x, size=f1.shape[-2:], mode="bilinear",
                              align_corners=False)
        x = self.conv2(torch.cat([x, f1], dim=1))
        x = self.up3(x)
        if x.shape[-2:] != f0.shape[-2:]:
            x = F.interpolate(x, size=f0.shape[-2:], mode="bilinear",
                              align_corners=False)
        x = self.conv3(torch.cat([x, f0], dim=1))
        return self.out(x)


class SAM3Seg2D(nn.Module):
    """项目接口包装：SAM3 ViT-L 视觉骨干（默认冻结）+ 监督解码头。"""

    def __init__(self, in_ch: int = 3, out_ch: int = 3,
                 norm: str = "batch", pretrained=None,
                 freeze_encoder: bool = True, input_size: int = 1008):
        super().__init__()
        self.input_size = int(input_size)
        self.image_encoder = SAM3ImageEncoder()
        self.decoder = SAM3Decoder(in_ch=256, out_ch=out_ch, norm=norm)
        if pretrained is not None:
            load_pretrained_sam3(self, pretrained)
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
        input_size = x.shape[-2:]
        # 官方 SAM3 预处理：直接缩放到 1008x1008，再归一化到 [-1,1]
        enc_in = F.interpolate(x, size=(self.input_size, self.input_size),
                               mode="bilinear", align_corners=False)
        enc_in = (enc_in - 0.5) / 0.5
        if self.freeze_encoder:
            with torch.no_grad():
                feats = self.image_encoder(enc_in)
        else:
            feats = self.image_encoder(enc_in)
        logits = self.decoder(feats)
        return F.interpolate(logits, size=input_size, mode="bilinear",
                             align_corners=False)


def load_pretrained_sam3(model, ckpt_path, verbose: bool = True):
    """加载官方 SAM3 checkpoint 里的 **视觉骨干**（trunk + neck）。

    官方文件是 `{"model": state_dict, ...}`，视觉键的常见前缀：
        detector.backbone.visual.trunk.* / ...convs.*
    也兼容 backbone.visual.* 或裸 trunk.* / convs.*。
    prompt/text/detector 其余部分全部忽略。
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        raise SystemExit(f"[错误] 找不到 SAM3 预训练权重: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    prefixes = ("detector.backbone.vision_backbone.",
                "backbone.vision_backbone.",
                "detector.backbone.visual.", "backbone.visual.",
                "detector.", "")
    enc_sd = {}
    for k, v in sd.items():
        for p in prefixes:
            if p == "" or k.startswith(p):
                key = k[len(p):]
                # 本项目的 trunk 挂在 Sam3DualViTDetNeck 下面：
                #   官方 trunk.*  -> 本地 neck.trunk.*
                #   官方 convs.*  -> 本地 neck.convs.*
                if key.startswith(("trunk.", "convs.")):
                    enc_sd["neck." + key] = v
                break

    missing, unexpected = model.image_encoder.load_state_dict(enc_sd,
                                                              strict=False)
    if missing:
        raise SystemExit(
            f"[错误] SAM3 权重与视觉骨干不匹配：缺少 {len(missing)} 个键，"
            f"例如 {missing[:5]}")
    if verbose:
        n = sum(p.numel() for p in model.image_encoder.parameters())
        print(f"[SAM3 预训练] {ckpt_path.name}: 视觉骨干 {len(enc_sd)} 个张量 / "
              f"{n / 1e6:.2f}M 参数 | 忽略 text/detector/seg head")
    return {"encoder_tensors": len(enc_sd), "unexpected": len(unexpected)}
