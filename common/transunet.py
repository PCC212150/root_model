"""TransUNet（Chen et al., 2021）的 2D 实现，接口与 common/unet.py 一致。

结构（R50-ViT-B_16）：
    ResNetV2（pre-activation）下采样到 1/16，提供 1/8、1/4、1/2 的跳跃特征
    -> ViT-B/16 全局自注意力（输入 1/16 特征图，H/16 × W/16 个 token）
    -> DecoderCup（4 次上采样，前 3 次拼 ResNet 跳跃特征）
    -> 3x3 分割头，输出 3 通道 logits

代码 vendored 自 `Net/TransUNet/networks/vit_seg_modeling.py` 与
`vit_seg_modeling_resnet_skip.py`，改动：
1. 用项目自己的轻量 `Cfg` 代替 `ml_collections`（少一个依赖）；
2. `DecoderCup` 原来用 `int(sqrt(n_patch))` 还原方形特征图，这里改为
   **显式记录 patch 网格 (H/16, W/16)**，支持本项目的非方形输入；
3. `ResNetV2.forward` 的 padding 从「正方形 right_size」改成 H/W 分别计算；
4. `Conv2dReLU` 的归一化跟随项目 `--norm`（batch/group），不再写死 BatchNorm；
5. ViT 编码器支持 **梯度检查点**（`use_checkpoint=True`，默认开）——
   1024x1536 输入有 64x96=6144 个 token，不开检查点时 12 层注意力
   的中间激活很难塞进 24GB；
6. 包装成 `TransUNet2D(in_ch/out_ch/img_size/norm)`，并在末尾双线性上采样回
   输入分辨率（输入需能被 16 整除）。

与 Swin-Unet 一样是**固定输入尺寸**模型（位置编码、DecoderCup 的网格在构造时
固定），因此不支持 `--crop`；尺寸写进权重的 `arch_img_size`。

预训练：官方 `R50+ViT-B_16.npz`（约 440MB），由 `load_pretrained_transunet()`
按官方 `load_from` 的键名加载；位置编码从 14x14 双线性插值到训练网格。
"""
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from common.unet import make_norm

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


class Cfg(dict):
    """同时支持 `cfg.x` 与 `cfg['x']` 的嵌套配置（替代 ml_collections）。"""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def __setattr__(self, name, value):
        self[name] = value


def to_2tuple(x):
    return (x, x) if isinstance(x, int) else tuple(x)


def np2th(weights, conv=False):
    """NumPy 权重 HWIO -> PyTorch OIHW（卷积）。"""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": F.gelu, "relu": F.relu, "swish": swish}


# ---------------------------------------------------------------------------
# ResNetV2（R50 的 pre-activation 版本，提供多尺度跳跃特征）
# ---------------------------------------------------------------------------
class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1,
                     bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0,
                     bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation ResNet-v2 bottleneck block。"""

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout // 4
        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or cin != cout:
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):
        residual = x
        if hasattr(self, "downsample"):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))
        return self.relu(residual + y)

    def load_from(self, weights, n_block, n_unit):
        def conv(name):
            return np2th(weights[f"{n_block}/{n_unit}/{name}"], conv=True)

        self.conv1.weight.copy_(conv("conv1/kernel"))
        self.conv2.weight.copy_(conv("conv2/kernel"))
        self.conv3.weight.copy_(conv("conv3/kernel"))
        for gn in ("gn1", "gn2", "gn3"):
            getattr(self, gn).weight.copy_(
                np2th(weights[f"{n_block}/{n_unit}/{gn}/scale"]).view(-1))
            getattr(self, gn).bias.copy_(
                np2th(weights[f"{n_block}/{n_unit}/{gn}/bias"]).view(-1))
        if hasattr(self, "downsample"):
            self.downsample.weight.copy_(conv("conv_proj/kernel"))
            self.gn_proj.weight.copy_(
                np2th(weights[f"{n_block}/{n_unit}/gn_proj/scale"]).view(-1))
            self.gn_proj.bias.copy_(
                np2th(weights[f"{n_block}/{n_unit}/gn_proj/bias"]).view(-1))


class ResNetV2(nn.Module):
    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width
        self.root = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3),
            nn.GroupNorm(32, width, eps=1e-6),
            nn.ReLU(inplace=True))
        # **必须保留 block1/unit1 这些名字**：官方 npz 的键是
        # block1/unit1/conv1/kernel，按位置命名会找不到权重。
        body = []
        for bi, units in enumerate(block_units, start=1):
            cout = width * 2 ** (bi + 1)      # block1: 4w, block2: 8w, block3: 16w
            cin = width if bi == 1 else cout // 2
            cmid = cout // 4
            stride = 1 if bi == 1 else 2
            blocks = [("unit1", PreActBottleneck(cin=cin, cout=cout,
                                                 cmid=cmid, stride=stride))]
            blocks += [(f"unit{u}", PreActBottleneck(cin=cout, cout=cout,
                                                     cmid=cmid))
                       for u in range(2, units + 1)]
            body.append((f"block{bi}", nn.Sequential(OrderedDict(blocks))))
        self.body = nn.Sequential(OrderedDict(body))

    def forward(self, x):
        features = []
        b, c, in_h, in_w = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body) - 1):
            x = self.body[i](x)
            # [patch] 官方按正方形算 right_size；这里 H/W 分开，非方形才不报错
            right_h = int(in_h / 4 / (i + 1))
            right_w = int(in_w / 4 / (i + 1))
            if x.size()[2] != right_h or x.size()[3] != right_w:
                dh = right_h - x.size()[2]
                dw = right_w - x.size()[3]
                assert 0 <= dh < 3 and 0 <= dw < 3, \
                    f"ResNetV2 特征 {tuple(x.shape)} 与目标 {(right_h, right_w)} 差太多"
                feat = torch.zeros((b, x.size()[1], right_h, right_w),
                                   device=x.device, dtype=x.dtype)
                feat[:, :, :x.size()[2], :x.size()[3]] = x
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


# ---------------------------------------------------------------------------
# ViT 编码器
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, config, vis):
        super().__init__()
        self.vis = vis
        self.num_heads = config.transformer.num_heads
        self.head_size = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = nn.Dropout(config.transformer.attention_dropout_rate)
        self.proj_dropout = nn.Dropout(config.transformer.attention_dropout_rate)

    def forward(self, hidden_states):
        mixed_query = self.query(hidden_states)
        mixed_key = self.key(hidden_states)
        mixed_value = self.value(hidden_states)

        def split(x):
            return x.view(*x.size()[:-1], self.num_heads,
                          self.head_size).permute(0, 2, 1, 3)

        q, k, v = split(mixed_query), split(mixed_key), split(mixed_value)
        attn = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size)
        probs = attn.softmax(dim=-1)
        weights = probs if self.vis else None
        context = torch.matmul(self.attn_dropout(probs), v)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(*context.size()[:-2], self.all_head_size)
        return self.proj_dropout(self.out(context)), weights


class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.transformer.mlp_dim)
        self.fc2 = nn.Linear(config.transformer.mlp_dim, config.hidden_size)
        self.act = ACT2FN["gelu"]
        self.dropout = nn.Dropout(config.transformer.dropout_rate)

    def forward(self, x):
        return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, config, vis):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        return x + h, weights


class Encoder(nn.Module):
    def __init__(self, config, vis, use_checkpoint=True):
        super().__init__()
        self.vis = vis
        self.use_checkpoint = use_checkpoint
        self.layer = nn.ModuleList(
            [Block(config, vis) for _ in range(config.transformer.num_layers)])
        self.encoder_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        attn_weights = []
        for block in self.layer:
            if self.use_checkpoint and self.training:
                # 只回传第一个输出（Block 返回 (x, weights)），checkpoint 要求张量输出
                hidden_states = checkpoint.checkpoint(
                    lambda h, b=block: b(h)[0], hidden_states,
                    use_reentrant=False)
            else:
                hidden_states, weights = block(hidden_states)
                if self.vis:
                    attn_weights.append(weights)
        return self.encoder_norm(hidden_states), attn_weights


class Embeddings(nn.Module):
    def __init__(self, config, img_size, in_channels=3):
        super().__init__()
        self.config = config
        img_size = to_2tuple(img_size)
        self.hybrid = config.patches.get("grid") is not None
        if self.hybrid:
            grid_size = config.patches.grid
            patch_size = (img_size[0] // 16 // grid_size[0],
                          img_size[1] // 16 // grid_size[1])
            patch_real = (patch_size[0] * 16, patch_size[1] * 16)
            self.patch_grid = (img_size[0] // patch_real[0],
                               img_size[1] // patch_real[1])
            self.patch_size_real = patch_real
        else:
            patch_size = to_2tuple(config.patches.size)
            self.patch_grid = (img_size[0] // patch_size[0],
                               img_size[1] // patch_size[1])
            self.patch_size_real = patch_size

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = nn.Conv2d(in_channels, config.hidden_size,
                                          kernel_size=patch_size,
                                          stride=patch_size)
        n_patches = self.patch_grid[0] * self.patch_grid[1]
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, n_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.transformer.dropout_rate)

    def forward(self, x):
        features = None
        if self.hybrid:
            x, features = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2).transpose(-1, -2)
        x = self.dropout(x + self.position_embeddings)
        return x, features


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis, use_checkpoint=True):
        super().__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis, use_checkpoint=use_checkpoint)

    def forward(self, x):
        x, features = self.embeddings(x)
        encoded, attn = self.encoder(x)
        return encoded, attn, features


class Conv2dReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
                 stride=1, use_batchnorm=True, norm="batch"):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                         stride=stride, padding=padding,
                         bias=not use_batchnorm)
        norm_layer = (make_norm(norm, out_channels) if use_batchnorm
                      else nn.Identity())
        super().__init__(conv, norm_layer, nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0,
                 use_batchnorm=True, norm="batch"):
        super().__init__()
        self.conv1 = Conv2dReLU(in_channels + skip_channels, out_channels, 3,
                                padding=1, use_batchnorm=use_batchnorm,
                                norm=norm)
        self.conv2 = Conv2dReLU(out_channels, out_channels, 3, padding=1,
                                use_batchnorm=use_batchnorm, norm=norm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                         padding=kernel_size // 2)
        up = (nn.UpsamplingBilinear2d(scale_factor=upsampling)
              if upsampling > 1 else nn.Identity())
        super().__init__(conv, up)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.norm = config.norm
        head_channels = 512
        self.conv_more = Conv2dReLU(config.hidden_size, head_channels,
                                    kernel_size=3, padding=1,
                                    norm=self.norm)
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels
        skip_channels = list(config.skip_channels)
        if config.n_skip != 0:
            for i in range(4 - config.n_skip):
                skip_channels[3 - i] = 0
        else:
            skip_channels = [0, 0, 0, 0]
        self.blocks = nn.ModuleList([
            DecoderBlock(ic, oc, sc, norm=self.norm)
            for ic, oc, sc in zip(in_channels, out_channels, skip_channels)])

    def forward(self, hidden_states, features, grid):
        """grid=(h, w)：ViT token 的二维网格。"""
        b, n_patch, hidden = hidden_states.size()
        h, w = grid
        assert n_patch == h * w, f"token 数 {n_patch} 与网格 {(h, w)} 不符"
        x = hidden_states.permute(0, 2, 1).contiguous().view(b, hidden, h, w)
        x = self.conv_more(x)
        for i, block in enumerate(self.blocks):
            skip = features[i] if (features is not None and i < self.config.n_skip) else None
            x = block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size, num_classes, in_chans=3,
                 use_checkpoint=True):
        super().__init__()
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis=False,
                                       use_checkpoint=use_checkpoint)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config.decoder_channels[-1],
            out_channels=num_classes, kernel_size=3)
        self.grid = self.transformer.embeddings.patch_grid
        self.norm = config.norm

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        input_size = x.shape[-2:]
        encoded, _, features = self.transformer(x)
        y = self.decoder(encoded, features, self.grid)
        logits = self.segmentation_head(y)
        if logits.shape[-2:] != input_size:       # 未触发时是恒等操作
            logits = F.interpolate(logits, size=input_size, mode="bilinear",
                                   align_corners=False)
        return logits


def get_r50_b16_config(out_ch=3, norm="batch"):
    c = Cfg()
    c.patches = Cfg(size=(16, 16))
    c.hidden_size = 768
    c.transformer = Cfg(mlp_dim=3072, num_heads=12, num_layers=12,
                        attention_dropout_rate=0.0, dropout_rate=0.1)
    c.classifier = "seg"
    c.representation_size = None
    c.resnet = Cfg(num_layers=(3, 4, 9), width_factor=1)
    c.resnet_pretrained_path = None
    c.pretrained_path = None
    c.decoder_channels = (256, 128, 64, 16)
    c.skip_channels = [512, 256, 64, 16]
    c.n_classes = out_ch
    c.n_skip = 3
    c.activation = "sigmoid"
    c.norm = norm
    return c


class TransUNet2D(nn.Module):
    """项目接口包装：与其它骨架同签名。

    img_size 必须是训练时的精确 (H, W)，且 H/W 能被 16 整除。
    TransUNet 是固定输入尺寸模型（位置编码 + DecoderCup 网格），
    与 Swin-Unet 一样不支持 --crop。
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 3,
                 img_size=(1024, 1536), norm: str = "batch",
                 use_checkpoint: bool = True):
        super().__init__()
        if not isinstance(img_size, (tuple, list)):
            img_size = (img_size, img_size)
        img_size = (int(img_size[0]), int(img_size[1]))
        if img_size[0] % 16 or img_size[1] % 16:
            raise ValueError(f"TransUNet 要求 H/W 能被 16 整除，收到 {img_size}")
        self.img_size = img_size
        cfg = get_r50_b16_config(out_ch=out_ch, norm=norm)
        # R50-ViT-B_16：1/16 特征图上的每个位置就是一个 token
        cfg.patches.grid = (img_size[0] // 16, img_size[1] // 16)
        self.model = VisionTransformer(cfg, img_size=img_size,
                                       num_classes=out_ch, in_chans=in_ch,
                                       use_checkpoint=use_checkpoint)

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------------------------------------
# 预训练权重加载
# ---------------------------------------------------------------------------
def _load_encoder_block(block, weights, idx):
    root = f"Transformer/encoderblock_{idx}"

    def w(name):
        return np2th(weights[f"{root}/{name}"])

    for proj, key in (("query", ATTENTION_Q), ("key", ATTENTION_K),
                      ("value", ATTENTION_V), ("out", ATTENTION_OUT)):
        linear = getattr(block.attn, proj)
        linear.weight.copy_(w(f"{key}/kernel").view(
            block.hidden_size, block.hidden_size).t())
        linear.bias.copy_(w(f"{key}/bias").view(-1))
    block.ffn.fc1.weight.copy_(w(f"{FC_0}/kernel").t())
    block.ffn.fc1.bias.copy_(w(f"{FC_0}/bias").t())
    block.ffn.fc2.weight.copy_(w(f"{FC_1}/kernel").t())
    block.ffn.fc2.bias.copy_(w(f"{FC_1}/bias").t())
    block.attention_norm.weight.copy_(w(f"{ATTENTION_NORM}/scale"))
    block.attention_norm.bias.copy_(w(f"{ATTENTION_NORM}/bias"))
    block.ffn_norm.weight.copy_(w(f"{MLP_NORM}/scale"))
    block.ffn_norm.bias.copy_(w(f"{MLP_NORM}/bias"))


def load_pretrained_transunet(model, ckpt_path, verbose: bool = True):
    """加载官方 `R50+ViT-B_16.npz`（Google Storage / Drive）。

    键名与官方 `VisionTransformer.load_from` 完全一致；位置编码从
    14x14 双线性插值到当前训练网格 (H/16, W/16)。缺少任何 ViT/ResNet
    权重会直接 KeyError（官方 npz 必须完整）。
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.is_file():
        import config
        raise SystemExit(
            f"[错误] 找不到 TransUNet 预训练权重: {ckpt_path}\n"
            f"       下载: curl -L -o \"{ckpt_path}\" "
            f"{config.TRANSUNET_PRETRAINED_URL}\n"
            f"       或加 --transunet-pretrained none 从零训练")
    weights = np.load(ckpt_path, allow_pickle=True)
    inner = model.model if isinstance(model, TransUNet2D) else model

    with torch.no_grad():
        emb = inner.transformer.embeddings
        emb.patch_embeddings.weight.copy_(
            np2th(weights["embedding/kernel"], conv=True))
        emb.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
        inner.transformer.encoder.encoder_norm.weight.copy_(
            np2th(weights["Transformer/encoder_norm/scale"]))
        inner.transformer.encoder.encoder_norm.bias.copy_(
            np2th(weights["Transformer/encoder_norm/bias"]))

        posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
        pos_new = emb.position_embeddings
        if posemb.shape == pos_new.shape:
            pos_new.copy_(posemb)
            pos_mode = "直接加载"
        else:
            n_old = posemb.shape[1]
            gs = int(round(math.sqrt(n_old)))
            if gs * gs != n_old:
                # ImageNet-21k 的 pos_embed 带 1 个 cls token（n_old = 1 + gs²）
                gs = int(round(math.sqrt(n_old - 1)))
                if gs * gs == n_old - 1:
                    posemb = posemb[:, 1:]
                else:
                    raise SystemExit(
                        f"[错误] 预训练位置编码不是方形网格: {n_old} 个 token")
            if posemb.shape[1] == pos_new.shape[1]:
                pos_mode = "去掉 cls 后直接加载"
            else:
                gh, gw = emb.patch_grid
                grid = posemb.reshape(1, gs, gs, -1).permute(0, 3, 1, 2)
                grid = F.interpolate(grid, size=(gh, gw), mode="bilinear",
                                     align_corners=False)
                posemb = grid.permute(0, 2, 3, 1).reshape(
                    1, gh * gw, -1)
                pos_mode = f"{gs}x{gs} -> {gh}x{gw} 插值"
            pos_new.copy_(posemb)

        for i, block in enumerate(inner.transformer.encoder.layer):
            _load_encoder_block(block, weights, i)

        if emb.hybrid:
            hm = emb.hybrid_model
            hm.root[0].weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
            hm.root[1].weight.copy_(np2th(weights["gn_root/scale"]).view(-1))
            hm.root[1].bias.copy_(np2th(weights["gn_root/bias"]).view(-1))
            for bname, body in hm.body.named_children():
                for uname, unit in body.named_children():
                    unit.load_from(weights, n_block=bname, n_unit=uname)

    if verbose:
        n_vit = sum(p.numel() for p in inner.transformer.parameters())
        n_all = sum(p.numel() for p in inner.parameters())
        print(f"[TransUNet 预训练] {ckpt_path.name}: ViT {n_vit / 1e6:.2f}M / "
              f"总 {n_all / 1e6:.2f}M 参数 | 位置编码 {pos_mode}")
    return {"position_embedding": pos_mode}
