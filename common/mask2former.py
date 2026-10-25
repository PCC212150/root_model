"""Mask2Former（Cheng et al., CVPR 2022）的 2D 简化实现，接口为 `--arch mask2former`。

官方代码（Net/Mask2Former）依赖 detectron2 + 编译版 MultiScaleDeformableAttention，
服务器上两者都没有。这里按论文结构重写了一个自包含版本：

    ResNet 风格骨干（1/4,1/8,1/16,1/32）
      -> FPN 像素解码器（256 通道，四尺度 + 1/4 mask 特征）
      -> 3 层 transformer 解码器（100 个 query，self-attn + 多尺度 deformable
         cross-attn + FFN）
      -> 每个 query 输出 class logits 和 1/4 分辨率 mask logits
      -> 按 class 对 query 做 softmax 加权，得到逐像素概率，再转成 logits

与官方的差异（为了能在这个项目里直接训练）：

1. **没有用匈牙利匹配/集合损失**。官方是「query 分类 + mask 匹配」的集合预测；
   本项目的训练循环是逐像素 BCE + Dice，所以这里把 query 的 mask 聚合成
   out_ch 个逐像素 logits，直接喂给现有损失。多标签（root/stem/check 可重叠）
   也能工作：每个 class 独立对 query 做 softmax。
2. **没有 masked attention**：官方用上一层的 mask 限制 cross-attention 的采样区域；
   这里用标准的多尺度 deformable attention（纯 PyTorch grid_sample 实现）。
3. 骨干从零训练，不加载 ImageNet/detectron2 权重。

输入 H/W 会内部 pad 到 32 的倍数，输出再裁回原尺寸。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.unet import make_norm

PAD_MULTIPLE = 32


def _interpolate(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


def ms_deform_attn_core_pytorch(value, spatial_shapes, sampling_locations,
                                attention_weights):
    """多尺度 deformable attention 的纯 PyTorch 版本（改编自官方 ops 里的
    `ms_deform_attn_core_pytorch`，只有前向，自动求导走 grid_sample）。

    value:              (N, S, M, D)   S = sum(H*W)，M = heads，D = head_dim
    spatial_shapes:     [(H, W), ...]
    sampling_locations: (N, Lq, M, L, P, 2)，值域 [0,1]
    attention_weights:  (N, Lq, M, L, P)
    """
    N, S, M, D = value.shape
    _, Lq, _, L, P, _ = sampling_locations.shape
    value_list = value.split([H * W for H, W in spatial_shapes], dim=1)
    sampling_grids = 2.0 * sampling_locations - 1.0
    sampled = []
    for lid, (H, W) in enumerate(spatial_shapes):
        # N,H*W,M,D -> N*M,D,H,W
        v = value_list[lid].flatten(2).transpose(1, 2).reshape(N * M, D, H, W)
        # N,Lq,M,P,2 -> N*M,Lq,P,2
        grid = sampling_grids[:, :, :, lid].transpose(1, 2).flatten(0, 1)
        sampled.append(F.grid_sample(v, grid, mode="bilinear",
                                     padding_mode="zeros", align_corners=False))
    attn = attention_weights.transpose(1, 2).reshape(N * M, 1, Lq, L * P)
    out = (torch.stack(sampled, dim=-2).flatten(-2) * attn).sum(-1)
    return out.view(N, M * D, Lq).transpose(1, 2).contiguous()


class MSDeformAttn(nn.Module):
    """多尺度可变形注意力（对应官方 MSDeformAttn，去掉 CUDA op）。"""

    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model 必须能被 n_heads 整除")
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        self.sampling_offsets = nn.Linear(
            d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(
            d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)

    def forward(self, query, reference_points, value_features, spatial_shapes):
        """query: (B, Lq, C)；reference_points: (B, Lq, 2) in [0,1]；
        value_features: list of (B, C, H, W)，长度 = n_levels。"""
        orig_dtype = query.dtype
        device_type = "cuda" if query.is_cuda else "cpu"
        # grid_sample 要求 value 和 grid 同 dtype；整个 deformable attention
        # 统一在 fp32 下算，避免 AMP 把 linear 输出成 fp16 后与 fp32 value 冲突
        with torch.autocast(device_type=device_type, enabled=False):
            query = query.float()
            values = [v.float() for v in value_features]

            B, Lq, _ = query.shape
            value = torch.cat([v.flatten(2).transpose(1, 2) for v in values],
                              dim=1)
            value = self.value_proj(value).view(
                B, -1, self.n_heads, self.head_dim)

            offsets = self.sampling_offsets(query).view(
                B, Lq, self.n_heads, self.n_levels, self.n_points, 2)
            attn = self.attention_weights(query).view(
                B, Lq, self.n_heads, self.n_levels * self.n_points)
            attn = F.softmax(attn, dim=-1).view(
                B, Lq, self.n_heads, self.n_levels, self.n_points)

            # 把归一化坐标上的偏移换算成各尺度的像素偏移
            normalizer = torch.tensor(
                [[W, H] for H, W in spatial_shapes],
                dtype=torch.float32, device=query.device)
            ref = reference_points.float()[:, :, None, None, None, :]
            sampling = ref + offsets / normalizer[None, None, None, :, None, :]

            out = ms_deform_attn_core_pytorch(
                value, spatial_shapes, sampling, attn)
            out = self.output_proj(out)
        return out.to(orig_dtype)


# ---------------------------------------------------------------------------
# 骨干：ResNet-18 风格（4 个 stage，通道 64/128/256/512）
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, norm="group"):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1,
                               bias=False)
        self.norm1 = make_norm(norm, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.norm2 = make_norm(norm, out_ch)
        self.short = None
        if stride != 1 or in_ch != out_ch:
            self.short = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                make_norm(norm, out_ch))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.act(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        if self.short is not None:
            x = self.short(x)
        return self.act(y + x)


class ResNetBackbone(nn.Module):
    def __init__(self, in_ch=3, norm="group", base=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 7, stride=2, padding=3, bias=False),
            make_norm(norm, base), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1))
        self.layer1 = nn.Sequential(
            ResBlock(base, base, norm=norm),
            ResBlock(base, base, norm=norm))
        self.layer2 = nn.Sequential(
            ResBlock(base, base * 2, stride=2, norm=norm),
            ResBlock(base * 2, base * 2, norm=norm))
        self.layer3 = nn.Sequential(
            ResBlock(base * 2, base * 4, stride=2, norm=norm),
            ResBlock(base * 4, base * 4, norm=norm))
        self.layer4 = nn.Sequential(
            ResBlock(base * 4, base * 8, stride=2, norm=norm),
            ResBlock(base * 8, base * 8, norm=norm))
        self.out_channels = [base, base * 2, base * 4, base * 8]

    def forward(self, x):
        x = self.stem(x)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return [c1, c2, c3, c4]


# ---------------------------------------------------------------------------
# FPN 像素解码器
# ---------------------------------------------------------------------------
class FPNPixelDecoder(nn.Module):
    def __init__(self, in_channels, d_model=256, norm="group"):
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Conv2d(c, d_model, 1) for c in in_channels)
        self.smooth = nn.ModuleList(
            nn.Sequential(nn.Conv2d(d_model, d_model, 3, padding=1),
                          make_norm(norm, d_model),
                          nn.ReLU(inplace=True))
            for _ in in_channels)
        self.mask_conv = nn.Sequential(
            nn.Conv2d(d_model * len(in_channels), d_model, 3, padding=1),
            make_norm(norm, d_model),
            nn.ReLU(inplace=True))

    def forward(self, features):
        laterals = [lat(f) for lat, f in zip(self.lateral, features)]
        for i in range(len(laterals) - 2, -1, -1):
            laterals[i] = laterals[i] + _interpolate(
                laterals[i + 1], laterals[i].shape[-2:])
        outs = [smooth(l) for smooth, l in zip(self.smooth, laterals)]
        # mask 特征：所有尺度上采样到 1/4 后拼接
        target = outs[0].shape[-2:]
        mask_feat = self.mask_conv(torch.cat(
            [_interpolate(o, target) for o in outs], dim=1))
        return outs, mask_feat


# ---------------------------------------------------------------------------
# Transformer 解码器
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, n_heads=8, dim_ffn=1024,
                 n_levels=4, n_points=4, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = MSDeformAttn(
            d_model, n_levels=n_levels, n_heads=n_heads, n_points=n_points)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ffn), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(dim_ffn, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, query, query_pos, reference_points,
                value_features, spatial_shapes):
        q = query + query_pos
        attn, _ = self.self_attn(q, q, query, need_weights=False)
        query = self.norm1(query + self.drop(attn))
        q = query + query_pos
        query = self.norm2(query + self.drop(
            self.cross_attn(q, reference_points, value_features, spatial_shapes)))
        query = self.norm3(query + self.drop(self.ffn(query)))
        return query


class Mask2Former2D(nn.Module):
    """query-based mask 解码器，聚合后输出 out_ch 个逐像素 logits。"""

    def __init__(self, in_ch=3, out_ch=3, norm="group", d_model=256,
                 n_heads=8, n_layers=3, n_queries=100, n_points=4):
        super().__init__()
        self.backbone = ResNetBackbone(in_ch=in_ch, norm=norm)
        self.pixel_decoder = FPNPixelDecoder(
            self.backbone.out_channels, d_model=d_model, norm=norm)
        self.query_feat = nn.Embedding(n_queries, d_model)
        self.query_pos = nn.Embedding(n_queries, d_model)
        self.ref_point_head = MLP(d_model, d_model, 2, 2)
        self.layers = nn.ModuleList(
            DecoderLayer(d_model, n_heads=n_heads, n_levels=4,
                         n_points=n_points)
            for _ in range(n_layers))
        self.mask_embed = MLP(d_model, d_model, d_model, 3)
        self.class_head = nn.Linear(d_model, out_ch)
        self.n_queries = n_queries
        self.n_levels = 4

        nn.init.normal_(self.query_feat.weight, std=0.02)
        nn.init.normal_(self.query_pos.weight, std=0.02)
        nn.init.xavier_uniform_(self.class_head.weight)
        nn.init.constant_(self.class_head.bias, 0.0)

    def forward(self, x):
        h, w = x.shape[-2:]
        pad_h = (-h) % PAD_MULTIPLE
        pad_w = (-w) % PAD_MULTIPLE
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        feats = self.backbone(x)
        value_features, mask_feat = self.pixel_decoder(feats)
        spatial_shapes = [f.shape[-2:] for f in value_features]

        B = x.shape[0]
        query = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1)
        query_pos = self.query_pos.weight.unsqueeze(0).expand(B, -1, -1)
        reference_points = torch.sigmoid(self.ref_point_head(query_pos))

        for layer in self.layers:
            query = layer(query, query_pos, reference_points,
                          value_features, spatial_shapes)

        class_logits = self.class_head(query)          # B,Q,C
        mask_embed = self.mask_embed(query)            # B,Q,d_model
        mask_pred = torch.einsum(
            "bqc,bchw->bqhw", mask_embed, mask_feat)   # B,Q,1/4,1/4

        # 每个 class 对 query 做 softmax 加权；多标签时各 class 独立。
        # 直接对 mask logits 加权（不要先 sigmoid 再 logit：那会把梯度压平，
        # 实测单张图都过拟合不了，输出会塌成常数）。
        weights = F.softmax(class_logits, dim=1)       # B,Q,C
        logits = torch.einsum("bqc,bqhw->bchw", weights, mask_pred)

        logits = F.interpolate(logits, size=(h, w), mode="bilinear",
                               align_corners=False)
        return logits
