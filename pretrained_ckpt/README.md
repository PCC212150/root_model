# Swin-Unet 预训练编码器

`--arch swin_unet` 默认加载这里的 ImageNet Swin-T 编码器权重。

## 文件

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
