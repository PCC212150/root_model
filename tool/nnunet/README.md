# nnU-Net 官方流程（root / stem / check 三个二分类数据集）

nnU-Net 是**完整框架**，不是能塞进 `--arch` 的单个网络。这里按官方流程做：

1. 把项目数据导出成 nnU-Net raw 格式；
2. 用 nnU-Net 自己的 plan / preprocess / 5 折训练 / 预测；
3. 用项目自己的指标脚本评测预测掩码。

## 为什么是三个数据集

nnU-Net 默认是 **softmax 多类分割**，要求类别互斥；本项目的 root / stem / check
是**互相重叠**的三个通道（茎也落在检查范围内），不能合成一张 label map。
所以每个通道单独训一个二分类 nnU-Net：

| 数据集 | 通道 | 标签 |
| --- | --- | --- |
| `Dataset501_Root` | root | 0=背景, 1=root |
| `Dataset502_Stem` | stem | 0=背景, 1=stem |
| `Dataset503_Check` | check | 0=背景, 1=check |

## 1. 导出数据

导出脚本不依赖 nnunetv2，在项目的 `rootnav` 环境（有 torch/numpy/PIL）里跑即可：

```bash
cd ~/project-PCC/Deep_learning_model_for_sugarcane/root_model

python tool/nnunet/export_dataset.py \
  --out nnUNet_raw \
  --src datasets/root/train \
  --test-src datasets/root/test/GT
```

默认用 `datasets/root/train` 下**全部**样本（GT + scr）。可选：

- `--gt-only`：去掉 scr，只保留 GT；
- `--exclude-val`：排除 `config.VAL_PLANTS` 对应的验证植株（与训练脚本同口径）；
- `--test-only`：只重新导出 `imagesTs` 并更新 `case_map.json` 的 `test` 部分，
  不动 `imagesTr/labelsTr`（训练集已经导出、只是漏了测试图时用）；
- `--limit N`：只导出前 N 组，调试用。

输出里每个数据集都有 `case_map.json`（case 名 ↔ 原始文件名），评测时要用。
**测试集三份 case_map 的 `test` 部分相同**，用任意一份都行。

数据量：77 组 × 3 份 PNG，约 **4~5GB**（原图 PNG 单张约 20MB）。磁盘不够可先把
`--limit` 用上做小规模验证。

## 2. 建 nnU-Net 环境

nnU-Net 2.8.1 要求 **Python ≥3.10**，不能装进服务器的 `rootnav`（3.9）。
单独建一个环境：

```bash
conda create -n nnunet python=3.11 -y
conda activate nnunet

# 选与本机 CUDA 匹配的 torch（服务器 cu12x）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 用仓库里的 2.8.1 源码（推荐，和 Net/nnUNet 一致）
pip install -e ~/project-PCC/Deep_learning_model_for_sugarcane/root_model/Net/nnUNet
# 或者直接 pip install nnunetv2==2.8.1
```

设置环境变量（建议写进 `~/.bashrc`）：

```bash
export nnUNet_raw=~/project-PCC/Deep_learning_model_for_sugarcane/root_model/nnUNet_raw
export nnUNet_preprocessed=~/project-PCC/Deep_learning_model_for_sugarcane/root_model/nnUNet_preprocessed
export nnUNet_results=~/project-PCC/Deep_learning_model_for_sugarcane/root_model/nnUNet_results
```

## 3. 规划 + 预处理

```bash
for d in 501 502 503; do
  nnUNetv2_plan_and_preprocess -d $d --verify_dataset_integrity
done
```

会生成 `nnUNet_preprocessed/Dataset50x_*/nnUNetPlans.json`。我们的数据是 2D PNG，
nnU-Net 会自动选 2D 配置；可以打开 plans 看一眼 patch size / batch size。

## 4. 训练

先跑每个通道的 **fold 0**，快速拿到一组数：

```bash
cd ~/project-PCC/Deep_learning_model_for_sugarcane/root_model
source ~/anaconda3/etc/profile.d/conda.sh && conda activate nnunet

# 先 nvidia-smi 看哪张卡空；下面按 2 号卡举例
for d in 501 502 503; do
  CUDA_VISIBLE_DEVICES=2 nnUNetv2_train $d 2d 0 --npz
done
```

确认没问题后再跑完整 5 折（最终指标用 5 折集成）：

```bash
for d in 501 502 503; do
  for f in 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=2 nnUNetv2_train $d 2d $f --npz
  done
done
```

- nnU-Net 2D 默认 1000 epoch；本数据 patch 512 左右，单折在 3090 上预计数小时。
- 三张卡可以并行跑三个通道/折，但每张卡只放一个训练进程，别和别人的任务抢。
- 显存不够时改 `nnUNet_plans.json` 的 batch_size，或 `nnUNetv2_train ... -p nnUNetPlans`
  配合自定义 plans；先按默认跑。

## 5. 预测

5 折都训完后，用 5 折集成预测 `imagesTs`：

```bash
for d in 501 502 503; do
  nnUNetv2_predict -d $d -c 2d \
    -i nnUNet_raw/Dataset${d}_*/imagesTs \
    -o nnUNet_results/preds_$d \
    -f 0 1 2 3 4 -chk checkpoint_best.pth
done
```

只跑了 fold 0 就用 `-f 0`。

## 6. 评测

预测结果是原图分辨率下的 0/1 标签图，用项目自己的 GT 口径评测：

```bash
python tool/nnunet/eval_predictions.py \
  --pred-root  nnUNet_results/preds_501 \
  --pred-stem  nnUNet_results/preds_502 \
  --pred-check nnUNet_results/preds_503 \
  --case-map   nnUNet_raw/Dataset501_Root/case_map.json \
  --data-dir   datasets/root/test/GT \
  --out        nnUNet_results/nnunet_test_summary.csv
```

输出每通道 Dice / IoU / 像素准确率、root/stem 的 clDice，以及 root 的根数/总长
（与 `test/test.py` 同样的「GT 检查范围 + 骨架→分链」口径）。

## 7. 结果怎么和现有模型比

- nnU-Net 用自己的预处理、归一化、增强、5 折 CV，**不是**我们 `--size` 那套；
  表里建议单列「nnU-Net 官方流程」，不要和 `--arch` 行混在一起说同一协议。
- 导出时默认把 scr 当普通标签（nnU-Net 没有 `--src-weight`）。如果要比「只用 GT」，
  用 `--gt-only` 重新导出/训练。
- 训练集口径：默认 77 组全进 nnU-Net 的 5 折 CV；我们的模型通常留 6/8 组做验证。
  想严格对齐就用 `--exclude-val`。
- 预测结果和 GT 都在原图分辨率，像素指标可以直接和 `test.py` 的 Dice/IoU/clDice 比；
  但 `test.py` 的「root Dice」是在模型输入分辨率上算的，数值会有少量差异，
  以本脚本的原图口径为准。
