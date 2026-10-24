# 预训练权重

| 模型 | 文件 | 大小 | 用途 |
| --- | --- | --- | --- |
| Swin-Unet | `swin_tiny_patch4_window7_224.pth` | 114342173 B | `--arch swin_unet` 默认加载（ImageNet-1K Swin-T） |
| TransUNet | `R50+ViT-B_16.npz` | 461217452 B | `--arch transunet` 默认加载（ImageNet-21k R50+ViT-B/16） |
| SAM | `sam_vit_b_01ec64.pth` | 375042383 B | `--arch sam` 默认加载（SA-1B ViT-B 图像编码器） |
| SAM 2 | `sam2.1_hiera_tiny.pt` | 156008466 B | `--arch sam2` 默认加载（Hiera-Tiny trunk+neck） |

`*.pth` / `*.npz` 均已被 `.gitignore` 忽略，权重文件不入库；新环境需要单独下载/拷贝。

## Swin-Unet 编码器

`--arch swin_unet` 默认加载这里的 ImageNet Swin-T 编码器权重。

| 文件 | 大小 | md5 |
| --- | --- | --- |
| `swin_tiny_patch4_window7_224.pth` | 114342173 B | `34f0d33ced06751da78d8e7149b5f11a` |

来源：官方 Swin Transformer Release（ImageNet-1K，Acc@1 81.3%）

```bash
curl -L -o swin_tiny_patch4_window7_224.pth \
  https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
```

GitHub 不通时走已有的代理，或在能上网的机器下好后拷到服务器的
`root_model/pretrained_ckpt/` 下。

## 加载行为

- 只取编码器（`patch_embed.*` / `layers.*` / `norm.*`），丢弃 1000 类分类头；
- 编码器权重**同时镜像**到解码器 `layers_up.{3-i}`（官方 Swin-Unet 的初始化方式）；
- `attn_mask` 的窗口数随输入分辨率变化（ImageNet 是 224×224），自动跳过，不影响加载；
- 编码器参数没有 100% 命中会直接报错，不会静默地从零训练。

`*.pth` 已被 `.gitignore` 忽略，权重文件不入库；新环境需要单独下载/拷贝。

## SAM ViT-B

`sam_vit_b_01ec64.pth`，375042383 B，md5 `01ec64d29a2fca3f0661936605ae66f8`。

```bash
curl -L -o sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

只取其中的**图像编码器**（`image_encoder.*`，177 个张量 / 89.67M 参数）；
prompt encoder 与 mask decoder 在本项目中不使用（我们是三通道监督分割，
没有点/框提示来源，见 `common/sam_seg.py` 的说明）。
`--sam-pretrained none` 可关闭预训练；`--sam-variant vit_l/vit_h` 可选更大编码器。

## SAM 2.1 Hiera-Tiny

`sam2.1_hiera_tiny.pt`，156008466 B，md5 `6aa6761c9da74fbaa74b4c790a0a2007`。

```bash
curl -L -o sam2.1_hiera_tiny.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

只取 `image_encoder.trunk.*` 与 `image_encoder.neck.*`（162 个张量 / 27.22M 参数）；
prompt/mask decoder、memory 全部忽略。`--sam2-variant small|base_plus|large` 可换更大编码器
（文件名对应 `sam2.1_hiera_{small,base_plus,large}.pt`，但默认路径仍指向 tiny，
换变体时用 `--sam2-pretrained` 显式指定）。
`--sam2-pretrained none` 可关闭预训练。

## TransUNet R50-ViT-B_16

`R50+ViT-B_16.npz`，461217452 B，md5 `fb93b71c850df1b0390448d4bafc4262`。

```bash
curl -L -o R50+ViT-B_16.npz \
  https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz
```

加载时位置编码会从 14×14 双线性插值到训练网格（如 64×96），
并自动处理 cls token；ViT 与 ResNetV2 的键名和官方 `load_from` 一致。
`--transunet-pretrained none` 可关闭（从零训练）。
