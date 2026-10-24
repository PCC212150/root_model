# 训练入口与各网络操作细则

本文只讲**怎么跑**。数据划分、损失函数、后处理的设计原因见项目根目录
[readme.md](../readme.md) 与 [config.py](../config.py)；预训练权重的下载路径见
[pretrained_ckpt/README.md](../pretrained_ckpt/README.md)。

训练入口始终是 `train/train.py`，用 `--arch` 切换骨干。所有骨干共用同一套
数据管线、损失（BCE + Dice + clDice）、增强、优化器、验证/选模型逻辑和 checkpoint
格式，因此**同一 `--size`、同一 split、同一 seed 下可以直接对比**。

---

## 1. 通用约定

### 1.1 数据与真值

- 默认训练目录：`datasets/root/train`（`GT/` 人工掩码 + `scr/` 伪标签）。
- **GT 的 root 默认用 `masks/<名>.png` 像素掩码**；json 多边形只在该图没有掩码时兜底。
  日志会打印 `[真值口径] root 使用像素掩码 masks/*.png: GT: 21/21 | scr: 0/50`。
  `--no-gt-mask` 只用于「掩码 vs 多边形」的 A/B 对照，平时不要加。
- `scr` 是折线伪标签，用 `--src-weight 0.3` 给它的损失打折（GT 恒为 1.0）。
- 验证集按**植株**整组切分（`config.VAL_PLANTS` 钉死 4 株），与训练集同株的时点不会
  落在验证集里；测试集见 `datasets/root/test/GT`。

### 1.2 输入尺寸

- `--size N`：整图缩放到**长边 N**（默认 `config.MAX_SIDE`）。缩放后的 H/W 会按
  `config.STRIDE=16` 取整，因此常规模型都能吃。
- `--crop N`：**切片训练**，原图不缩放，随机裁 `N×N` 的块；`N` 必须是 16 的倍数。
  适合想要「原始分辨率根宽」的实验。切片模型推理时必须显式给 `--size`（块边长）。
- 固定尺寸模型（Swin-Unet / TransUNet）**不支持 `--crop`**，且 H/W 分别要求被 32 / 16 整除；
  训练尺寸会写进 checkpoint 的 `arch_img_size`，测试/推理自动按它重建。
- 本项目常用 `--size 1536` → 实际输入 **1024×1536**（原图 5472×3648 按长边缩到 1536）。

### 1.3 显存与 batch

- `--batch` 是**每张卡**的 batch；等效 batch = `--batch × --accum × 卡数`。
- 单卡 `--norm batch` 时 batch 必须 ≥2；DDP 会自动把 BatchNorm 换成
  SyncBatchNorm，因此每卡 batch=1 也可用（全局统计量跨卡同步）。
- `--norm group` 不依赖 batch 统计量，但显存比 BatchNorm 高约一倍。
- 显存不够时优先降 `--size`，其次用 `--accum` 换等效 batch（不能救 BatchNorm 的 batch=1）。
- ⚠️ **冻结骨干的模型（sam / sam2 / sam3）建议用 `--norm group`**：解码头只有
  BatchNorm 时，训练用 batch 统计量、验证/测试用滑动平均，batch=1（含 DDP 每卡 1 张）
  下两者差异很大，实测 SAM2 的 check 通道在 eval 模式直接塌成全 1
  （val check Dice 恒定 0.7847）。GroupNorm 训练/推理行为一致。
- check 通道默认**没有 Dice 项**（`LOSS_DICE_W=(1,1,0)`），前景占 60~76% 时
  BCE 的天然最优就是全 1。U-Net 靠高分辨率跳跃能学出矩形；冻结骨干的小解码器
  容易直接塌成常数。需要 check 通道可用 `--dice-weight 1,1,1`（这是协议变更，
  对比实验要么所有模型都加、要么单独标注）。

### 1.4 三卡训练模板

服务器（3×RTX 3090）上的统一跑法：

```bash
PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,2,3 torchrun --nproc_per_node=3 train/train.py \
  --arch <骨架> --size 1536 --batch 1 --accum 4 --workers 2 --src-weight 0.3 --epochs 2000 \
  2>&1 | tee train_<骨架>.log
```

- `--workers 2`：服务器上读图与增强并行；本机 Windows 保持默认 0。
- 启动日志里会打印实际选中的卡、参数量、输入尺寸、通道权重、学习率与 clDice 设置。
- 两个训练任务同时跑会互相抢算力，epoch 时间不再可比；对比实验建议串行。
- `--gpu` / `CUDA_VISIBLE_DEVICES` 指定卡；不加时按当前空闲显存自动挑。

### 1.5 依赖

- U-Net / UNet++ / Attention U-Net：torch 即可。
- Swin-Unet：`pip install timm einops`；SegFormer：`timm`（装过 Swin 的 einops 就不用了）。
- TransUNet / SAM / SAM2 / SAM3：**不需要额外装包**——实现已 vendored 到 `common/`。
  注意 SAM 2/3 官方包要求 Python≥3.10，本项目的 vendored 版本在 Python 3.9 上也能跑。

### 1.6 checkpoint 与测试

每次训练生成 `model/model_YYYYMMDDHHMM/`：

- `model_*.pth`：包含 `state_dict`、`arch`、`hparams`、`arch_img_size`（固定尺寸模型）；
- `model_*_log.txt`、`hparams.json`。

测试/推理会自动从权重里识别 `arch` 和 `variant`，**不需要再给 `--arch`**：

```bash
# 像素掩码口径（本项目的主尺子）
python test/test.py --model model_202610241829 --data-dir datasets/root/test/GT

# 推理整个文件夹
python inference.py --model model_202610241829 --dir <图片文件夹> --size 1536
```

---

## 2. 各网络操作细则

### 2.1 `--arch unet`（经典 U-Net，baseline）

```bash
torchrun --nproc_per_node=3 train/train.py \
  --arch unet --size 1536 --batch 1 --accum 4 --workers 2 --src-weight 0.3 --epochs 2000
```

- 全卷积，支持 `--crop`；输入 H/W 需被 16 整除（`image_io.target_size` 保证）。
- 参数量 **31.04M**；无预训练。
- 显存参考（本机 batch1，1024×1536）：约 4~5GB；3090 上 `--size 1536 --batch 4` 实测约 18GB。
- 单卡 batch=1 + `--norm batch` 会欠分割，用 batch≥2 或 DDP。

### 2.2 `--arch unet++`（嵌套跳跃连接）

```bash
torchrun --nproc_per_node=3 train/train.py \
  --arch unet++ --size 1536 --batch 1 --accum 4 --workers 2 --src-weight 0.3 --epochs 2000
```

- 全卷积，支持 `--crop`；参数量 **36.62M**；无预训练。
- **显存显著高于 U-Net**：全分辨率节点多，2048/batch1 实测约 24GB 会 OOM，
  1792 在 24GB 上卡住；3×3090（0 号卡另有占用）**从 1536 起步**。
- 其它超参与 U-Net 相同，只换骨架。

### 2.3 `--arch attention_unet`（注意力门控）

```bash
torchrun --nproc_per_node=3 train/train.py \
  --arch attention_unet --size 1536 --batch 1 --accum 4 --workers 2 --src-weight 0.3 --epochs 2000
```

- 全卷积，支持 `--crop`；参数量 **31.74M**；无预训练。
- 显存与 U-Net 接近（每个 decoder 跳连多一个小的加法注意力门）。
- 归一化跟随 `--norm`，与 U-Net 可直接对比。

### 2.4 `--arch swin_unet`（Swin Transformer U-Net）

```bash
pip install timm einops          # 只需装一次

torchrun --nproc_per_node=3 train/train.py \
  --arch swin_unet --size 1536 --batch 1 --accum 4 --workers 2 --src-weight 0.3 --epochs 2000
```

- **固定输入尺寸**：H/W 必须被 **32** 整除；`--size 1536` → 1024×1536。不支持 `--crop`。
- 参数量 **41.38M**；默认加载 ImageNet Swin-T 编码器（`config.SWIN_PRETRAINED_CKPT`）。
  日志应出现 `编码器 171/171 个张量、镜像解码器 140 个 | 共 39.35M / 41.38M 参数`。
- 关闭预训练：`--swin-pretrained none`（做「预训练 vs 从零」对照时用）。
- 显存（batch1，1024×1536）：约 **8.3GB**，3×3090 很宽裕。
- 测试/推理必须与训练同尺寸；checkpoint 里记了 `arch_img_size`，`test.py` 自动使用。
- 权重缺失时会直接报错并给出下载命令；服务器上需要先放好 `.pth`（见 pretrained_ckpt/README.md）。

### 2.5 `--arch segformer`（MiT + 轻量解码头）

```bash
torchrun --nproc_per_node=3 train/train.py \
  --arch segformer --size 1536 --batch 1 --accum 4 --workers 2 --src-weight 0.3 --epochs 2000
```

- **全卷积**，无固定尺寸，支持 `--crop`。
- 默认变体 **b2**（24.72M），`--segformer-variant b0|b1|b2|b3|b4|b5` 可换：
  b0 3.71M / b1 13.68M / b2 24.72M / b3 44.60M（b4/b5 更大）。
  训练图只有 71 组时，b0/b1 往往比 b2 更稳。
- 预训练默认关闭（`--segformer-pretrained none`）。官方 mmseg 权重要用
  `--segformer-pretrained <path>` 显式指定；支持 `backbone.*` 与裸 `backbone.` 键名，
  编码器覆盖率低于 90% 会报错。
- 显存（b2，batch1，1024×1536）：约 **8.5GB**。
- 测试/推理全卷积，尺寸可不同，但建议与训练一致。

### 2.6 `--arch transunet`（R50-ViT-B_16 + DecoderCup）

```bash
torchrun --nproc_per_node=3 train/train.py \
  --arch transunet --size 1536 --batch 1 --accum 4 --workers 2 --src-weight 0.3 --epochs 2000
```

- **固定输入尺寸**：H/W 必须被 **16** 整除；`--size 1536` → 1024×1536。不支持 `--crop`。
- 参数量：`--size 768` → 106.31M；`--size 1536` → **109.84M**（位置编码随网格变大）。
- 默认加载官方 `R50+ViT-B_16.npz`（ImageNet-21k）；日志应出现
  `位置编码 14x14 -> 64x96 插值`。关闭：`--transunet-pretrained none`。
- 1024×1536 有 **6144 个 ViT token**，默认开梯度检查点；显存（batch1）约 **10.9GB**。
- 不要加 `--transunet-no-checkpoint`（省一点速度，但 24GB 很可能 OOM）。
- 权重是 461MB 的 `.npz`，默认路径见 `config.TRANSUNET_PRETRAINED_CKPT`。

### 2.7 `--arch sam`（SAM 1 图像编码器 + 监督解码头）

```bash
torchrun --nproc_per_node=3 train/train.py \
  --arch sam --size 1536 --batch 1 --accum 4 --workers 2 --src-weight 0.3 --epochs 2000
```

- **不是 promptable 零样本 SAM**：原版需要点/框提示；本项目用 SAM 的 ViT-B 图像编码器
  做骨干，接一个监督解码头，输出仍然是三通道 logits。原因见
  [common/sam_seg.py](../common/sam_seg.py) 的模块说明。
- 全卷积，支持 `--crop`；总参数 **90.24M**，默认**冻结编码器**，可训练只有 **0.57M**。
- 默认加载官方 `sam_vit_b_01ec64.pth`（SA-1B ViT-B）；日志应出现
  `图像编码器 177 个张量 / 89.67M 参数`。关闭：`--sam-pretrained none`。
- 显存（batch1，1024×1536）：约 **5.6GB**。
- `--sam-variant vit_b|vit_l|vit_h`（默认 vit_b；vit_h 636M，24GB 上很可能 OOM）。
- `--sam-finetune-encoder` 解冻编码器一起训练：默认不推荐（71 张图容易过拟合，显存也高）。
- ⚠️ **SAM1 编码器只输出 1/16 特征图，解码器没有高分辨率跳跃**；1024×1536 下根宽
  只有约 2px，到 1/16 特征图上不足 1px。实测它在细根任务上明显弱于 U-Net，
  且 check 通道容易塌成常数（check 的 Dice 权重为 0，BCE 只占 0.3）。
  要试 SAM 系列，优先用 `--arch sam2`（Hiera 的 FPN 提供 1/4、1/8、1/16 三级特征）。
- 跑 SAM1 时同样建议 `--norm group --dice-weight 1,1,1`（理由见 1.3 与 SAM2 一节）。

### 2.8 `--arch sam2`（SAM 2.1 Hiera + FPN 跳跃解码头）

```bash
torchrun --nproc_per_node=3 train/train.py \
  --arch sam2 --size 1536 --batch 1 --accum 4 --workers 2 --src-weight 0.3 --epochs 2000 \
  --norm group --dice-weight 1,1,1
```

- **必须加 `--norm group`**：解码头用 BatchNorm 时，batch=1（DDP 每卡 1 张）下
  eval 的滑动平均与训练 batch 统计差很多，实测 check 通道在 eval 直接全 1；
  GroupNorm 没有这个问题。
- **`--dice-weight 1,1,1` 是按需的协议变更**：不加时 check 仍会塌成全 1
  （BCE 在 60~76% 前景下的最优解）；加上后约 6 轮 check Dice 从 0.79 升到 0.9+。
  如果整张对比表要保持同一损失，要么所有模型都加这一项、要么都不加。
- 同样**不是 promptable 零样本 SAM 2**，定位与 `--arch sam` 一致：Hiera 图像编码器 +
  监督解码头；区别是 SAM 2 的 FPN 提供 1/4、1/8、1/16 三级跳跃特征。
- 全卷积，支持 `--crop`；总参数 **28.23M**，默认冻结编码器，可训练 **1.01M**。
- 默认加载 `sam2.1_hiera_tiny.pt`（Hiera-Tiny）；日志应出现
  `编码器 162 个张量 / 27.22M 参数`。关闭：`--sam2-pretrained none`。
- 显存（batch1，1024×1536）：约 **1.4GB**，是这批模型里最轻的。
- `--sam2-variant tiny|small|base_plus|large`（默认 tiny）。换变体时默认路径仍指向 tiny，
  要同时用 `--sam2-pretrained pretrained_ckpt/sam2.1_hiera_small.pt` 等显式指定。
- `--sam2-finetune-encoder` 解冻编码器；默认冻结。

### 2.9 `--arch sam3`（SAM 3 ViT-L 视觉骨干 + 监督解码头）

```bash
torchrun --nproc_per_node=3 train/train.py \
  --arch sam3 --size 1536 --batch 1 --accum 4 --workers 2 --src-weight 0.3 --epochs 2000
```

- 同样**不是 promptable 零样本 SAM 3**：只取官方 ViT-L/14 + ViTDet neck 作为骨干，
  接监督解码头输出三通道 logits。
- **内部固定缩放到 1008×1008**（SAM3 官方预处理），再归一化到 [-1,1]；输出 logits
  会双线性上采样回你的 `--size` 实际尺寸，所以外部 `--size` 仍可自由选。
- 总参数 **455.12M**，默认冻结 ViT-L，可训练只有解码头 **1.08M**。
- 默认加载 ModelScope 的 `sam3.pt`（3.45GB）；日志应出现
  `视觉骨干 442 个张量 / 454.04M 参数`。关闭：`--sam3-pretrained none`。
- 显存（batch1，1024×1536）约 **2.1GB**（冻结编码器）。保存的 checkpoint 约 1.8GB。
- `--sam3-finetune-encoder` 解冻 ViT-L：默认不推荐（455M 参数、71 张图、显存高）。
- 同样建议 `--norm group --dice-weight 1,1,1`（BatchNorm batch=1 的 eval 塌陷 +
  check 通道 BCE-only 塌成全 1，原因见 1.3 与 SAM2 一节）。
- 官方 HF 仓库是 gated；下载走 ModelScope，见 pretrained_ckpt/README.md。

---

## 3. 一页速查

| `--arch` | 全卷积 | 尺寸约束 | `--crop` | 预训练（默认） | 参数量（总/可训练） | 1024×1536 峰值* |
| --- | --- | --- | --- | --- | --- | --- |
| `unet` | ✓ | H/W %16 | ✓ | 无 | 31.04M / 31.04M | ~4–5GB |
| `unet++` | ✓ | H/W %16 | ✓ | 无 | 36.62M / 36.62M | 1536 可跑，2048 OOM |
| `attention_unet` | ✓ | H/W %16 | ✓ | 无 | 31.74M / 31.74M | 与 U-Net 接近 |
| `swin_unet` | ✗ | H/W %32 | ✗ | ImageNet Swin-T | 41.38M / 41.38M | 8.3GB |
| `segformer` | ✓ | 建议 %16 | ✓ | 无（可指定 mmseg） | b2 24.72M / 24.72M | 8.5GB |
| `transunet` | ✗ | H/W %16 | ✗ | ImageNet-21k R50+ViT | 109.84M / 109.84M | 10.9GB |
| `sam` | ✓ | 建议 %16 | ✓ | SA-1B ViT-B | 90.24M / **0.57M** | 5.6GB |
| `sam2` | ✓ | 建议 %16 | ✓ | SAM 2.1 Hiera-T | 28.23M / **1.01M** | 1.4GB |
| `sam3` | ✓ | 建议 %16 | ✓ | SAM 3 ViT-L | 455.12M / **1.08M** | 2.1GB |

\* 本机 RTX 5060（8GB，batch1，前向+反向）。8GB 卡上的数字超过 8GB 时会发生共享内存
回退、速度变慢但未必 OOM；3090（24GB）上实际更快、更宽裕。UNet++ 是唯一需要在
3090 上特别小心的骨架。

### 常用附加参数

| 参数 | 作用 |
| --- | --- |
| `--src-weight 0.3` | scr 伪标签损失权重；GT 恒为 1.0 |
| `--norm group` | GroupNorm；单卡 batch=1 时用，显存比 BatchNorm 高约一倍 |
| `--no-amp` | 关闭混合精度（调试数值问题用） |
| `--pos-weight 1,10,1` | 覆盖逐通道 BCE 正样本权重（默认见 config） |
| `--bce-weight 1,1,0.3` | 覆盖逐通道 BCE 权重（默认见 config，格式同上） |
| `--dice-weight 1,1,1` | 覆盖逐通道 Dice 权重；check 默认 0，冻结骨干模型可加 1 防止塌成全 1 |
| `--no-gt-mask` | 忽略 GT 像素掩码、改用 json 多边形；**只用于 A/B 对照** |
| `--val-size N` | 验证集植株数（默认 config.VAL_SIZE；当前由 VAL_PLANTS 钉死） |
| `--patience` / `--lr-patience` | 早停 / 降 LR 的耐心轮数 |
| `--gpu N` | 单进程指定卡；DDP 用 `CUDA_VISIBLE_DEVICES` |
| `--out-dir` | 模型输出根目录（默认 `model/`） |

### 新加骨干时的检查清单

1. 模型必须提供 `in_ch=3, out_ch=3, norm=...` 构造接口，输出 `(B,3,H,W)` logits；
2. 在 `common/ckpt.py` 里注册 `arch` 分支，并能从 `state_dict` 找到输出层；
3. 固定尺寸模型要把 `arch_img_size` 写进 checkpoint，并在 `train.py` 的固定尺寸检查里登记；
4. 在 `train.py` 的 `--arch` choices 与帮助文本里登记；
5. 跑一遍：1 轮 smoke 训练 → `test.py --model <新模型>` 能加载并评测。
