"""训练 U-Net 分割甘蔗根系（多标签三通道：根系 / 茎横截面 / 检查范围）。

运行（pcc 环境，建议在项目根目录）：
    conda activate pcc
    python train/train.py                        # 默认 200 轮 / 早停 60
    python train/train.py --size 1536 --batch 8  # 部署到 4xRTX3090 服务器时

每轮输出：轮次 / 损失 / 验证Dice（根系通道） / 本轮耗时(s)（控制台 + 模型文件夹内日志），
行尾另附茎与检查范围通道的 Dice 与**联合指标** `select_dice`。
模型保存：model/model_YYYYMMDDHHMM/（验证最优轮权重 .pth + 参数日志 .txt + hparams.json）

**保存 / 早停 / 降 LR 用同一个判据 `select_dice`**，不是单看根系通道：
把**茎当准入门槛** —— 茎 ≥ `config.STEM_MIN_DICE` 时判据就是纯根系 Dice（与历史行为
一致），达不到才按缺口打折。防的是「根已到顶、茎还没练好」的权重：茎本身是要交付的
输出通道，塌茎权重实测与整体欠训练相关（当年还有「锚定依赖茎」这条传导链，
锚定已于 2026-10-05 删除，门槛暂按原值保留，见 config.py）。常数依据见 `config.py`，
完整来龙去脉（含一个"这个数据规模测不出 0.02 以下差异"的教训）见 README 的
「选模型判据」一节。

数据划分**按植株整组进出**（同一植株的不同时点不会分处训练/验证两侧），
`--val-size` 是验证集**植株数**。

服务器上运行：
CUDA_VISIBLE_DEVICES=0 python train.py --size  --batch  --workers 
-1 则为cpu

CUDA_VISIBLE_DEVICES=0,2,3 torchrun --nproc_per_node=3 train/train.py --size 2736 --batch 1 --accum 4 --workers 2 --src-weight 0.3 --epochs 2000

"""
import argparse
import contextlib
import json
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1] #获取当前文件所在目录的上两级目录路径:root_model
sys.path.insert(0, str(PROJECT_ROOT))              #把项目根目录插到 Python 模块搜索路径最前面（索引 0），这样 `import` 时优先从这个目录找包，解决 模块导入找不到 问题

import config  # noqa: E402
from common import device as device_mod, naming  # noqa: E402     表示忽略 E402 错误，即允许在文件非顶部位置导入模块而不触发警告
from common.dataset import RootDataset, plant_key  # noqa: E402
from common.attention_unet import AttentionUNet  # noqa: E402
from common.unet import UNet  # noqa: E402
from common.unet_plusplus import UNetPlusPlus  # noqa: E402

N_CH = len(config.CLASS_NAMES)          # 数据里的三通道
TASK_TO_CH = {"root": 0, "stem": 1, "check_background": 2}


def dice_loss(prob, target):
    """prob: sigmoid 后概率 (B,C,H,W), target: 0/1。返回逐样本逐通道 Dice 损失 (B,C)。"""
    eps = 1.0
    inter = (prob * target).sum(dim=(2, 3))
    den = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * inter + eps) / (den + eps)
    return 1.0 - dice


def _soft_erode(img):
    """软腐蚀：3x3 十字的最小池化。用 max_pool 实现（-maxpool(-x) = minpool），
    这样梯度能穿过池化——这是软骨架能当损失用的关键。"""
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))     # 竖直方向 3 邻域取最小
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))     # 水平方向 3 邻域取最小
    return torch.min(p1, p2)


def _soft_dilate(img):
    # **软膨胀 soft dilate**：3×3 方形窗口最大池化，可微分，用来把概率图前景向外扩张
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def _soft_open(img):
    # **软开运算 soft open：先软腐蚀，再软膨胀**，可微分形态学操作，用来**抹除细小虚假分支 / 孤立噪点，保留主体根系形状**。
    return _soft_dilate(_soft_erode(img))


def soft_skel(img, iters):
    """软骨架化（clDice 论文的实现）。

    每轮腐蚀一层，把「这一层的残差」并进骨架：细结构在第一轮就整体成为骨架，
    粗结构则要腐蚀若干轮才露出中心线。**因此 iters 必须 ≥ 结构的最大半径**，
    否则骨架为 0（见 config.CLDICE_ITERS 的说明）。
    """
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)       # 软 OR：把新残差并进骨架
    return skel


def cldice_loss(prob, target, iters, chans):
    """按通道算 clDice 损失，返回 (loss[B,K], 退化标记[B,K] bool)。

    prob/target: (B,C,H,W)；chans 是要算的通道号列表。
    tprec = 预测骨架落在真值掩码内的比例（惩罚"多画出的骨架"）
    tsens = 真值骨架被预测掩码覆盖的比例（惩罚"断掉的骨架"）
    退化 = 真值骨架为空（iters 太小），此时损失恒为 0，必须让调用方知道。
    """
    eps = 1e-6
    losses, degen = [], []
    for c in chans:
        p, t = prob[:, c:c + 1], target[:, c:c + 1]
        sk_p, sk_t = soft_skel(p, iters), soft_skel(t, iters)
        sp = sk_p.sum(dim=(1, 2, 3))
        st = sk_t.sum(dim=(1, 2, 3))
        # 预测骨架为空时必须显式置 0：否则 (0+eps)/(0+eps)=1 —— 一个"什么都没预测出骨架"
        # 的平摊概率图会白拿满分，等于奖励模型把背景概率整体抬高。
        tprec = torch.where(sp > 0.5,
                            (sk_p * t).sum(dim=(1, 2, 3)) / sp.clamp(min=eps),
                            torch.zeros_like(sp))
        tsens = ((sk_t * p).sum(dim=(1, 2, 3)) + eps) / (st + eps)
        denom = tprec + tsens
        cl = 2.0 * tprec * tsens / denom.clamp(min=eps)
        d = st < 0.5                                   # 真值骨架为空 = 退化
        # 退化样本不给损失（否则会按 clDice=1 白白产生一个 0，掩盖问题）
        losses.append(torch.where(d, torch.zeros_like(cl), 1.0 - cl))
        degen.append(d)
    return torch.stack(losses, dim=1), torch.stack(degen, dim=1)


def channel_weights(base, valid):
    """(C,) 基准权重 × (B,C) 通道有效性 -> (B,C)；缺标注的通道权重置 0（不参与损失）。"""
    w = torch.as_tensor(base, dtype=torch.float32,
                        device=valid.device).view(1, -1)
    return w * valid


def _accumulate(pb, gt, vd, d_all, i_all):
    """把**一张图**的逐通道 Dice / IoU 追加进 d_all / i_all；缺标注的通道记 nan。

    切片模型与整图模型两条验证路径共用，保证两种模式的指标口径完全一致。
    pb / gt 是 (C, h, w) 的 bool，vd 是 (C,)。在**原图分辨率**上算。
    """
    ds, is_ = [], []
    for c in range(len(vd)):
        if vd[c] < 0.5:
            ds.append(np.nan)
            is_.append(np.nan)
            continue
        tp = float((pb[c] & gt[c]).sum())
        ds.append(2.0 * tp / (pb[c].sum() + gt[c].sum() + 1e-8))
        is_.append(tp / (pb[c].sum() + gt[c].sum() - tp + 1e-8))
    d_all.append(ds)
    i_all.append(is_)


def _reduce_val_metric(arrs, ddp):
    """把各卡上的**逐样本**指标并起来，返回 (C,) 的全局均值（缺标注的通道 = nan）。

    DDP 下每张卡只验证自己那一份样本，直接把均值再平均是错的（各卡样本数可能不等，
    而且某些通道在某些卡上全是 nan）。所以按 (非 nan 的和, 非 nan 的个数) 两个量分别
    跨卡求和再相除 —— 结果与"单卡跑完整验证集"的 nanmean 完全一致。

    world=1 时**原样返回**（连 numpy 转换都不做），单卡路径一个字节都没变。
    """
    if not ddp.enabled:
        return None
    import torch.distributed as dist
    if not arrs:
        return None
    # 注意用 asarray 而不是 concatenate：arrs 是 [(C,), (C,), ...]，
    # concatenate(axis=0) 会把它**压成一维 (n*C,)** —— 三个通道被混成一个数，
    # 而且不报错（除了最后 list() 那次）。asarray 才是按行堆成 (n, C)。
    a = np.asarray(arrs, dtype=np.float64)
    s = np.nansum(a, axis=0)
    c = np.sum(~np.isnan(a), axis=0)
    # **张量必须和通信后端在同一个设备上**：NCCL 只做 GPU 张量，喂 CPU 张量会直接
    # 报 "No backend type associated with device type cpu"（2026-10-08 在服务器上
    # 第一次跑 NCCL 才暴露 —— 本机验证用的是 gloo，而 gloo 恰好能算 CPU，所以一直没踩到）。
    dev = (torch.device("cuda", torch.cuda.current_device())
           if ddp.backend == "nccl" else torch.device("cpu"))
    t = torch.from_numpy(np.stack([s, c])).to(dev)  # (2, C)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    s, c = t[0].cpu().numpy(), t[1].cpu().numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c > 0, s / np.maximum(c, 1.0), np.nan)


# 训练大概要多少显存（低于这个值就告警）。1024/batch2 实测 4.4GB；
# 1536/batch4 在 24G 卡上约 10~12GB。这里给个下限，不够时由 device.pick 告警。
NEED_GB = 4.5


def parse_args():
    p = argparse.ArgumentParser(description="训练甘蔗根系 U-Net（三通道多标签）")
    p.add_argument("--size", type=int, default=config.MAX_SIDE, help="输入长边像素")
    p.add_argument("--crop", type=int, default=config.CROP_SIZE,
                   help="**切片训练**的块边长(px，必须是 16 的倍数)。>0 时从原图"
                        "**不缩放**随机裁 --crop × --crop 训练，让模型看到原始分辨率的"
                        "根宽（5472x3648 的图缩到 1024 时，原图 10px 的根只剩 1.9px）。"
                        "此时 --size 只写进权重、不参与预处理。"
                        "注意：推理时**必须显式给 --size**：模型是全卷积的，越接近原始分辨率"
                        "越好，24G 上 --size 3648 约 18GB。默认见 config.CROP_SIZE")
    p.add_argument("--crop-repeat", type=int, default=config.CROP_REPEAT,
                   help="切片模式下每张原图在一个 epoch 里抽多少个块（默认见 "
                        "config.CROP_REPEAT）。**必须 ≥2**：只有一个块时一个 epoch 才 "
                        "11 步，LR 平台期与早停会按轮很快触发，等于没训")
    p.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    p.add_argument("--accum", type=int, default=config.ACCUM,
                   help="梯度累积步数：等效 batch = --batch × --accum。显存不够时用它换等效 batch"
                        "（默认见 config.ACCUM）")
    p.add_argument("--src-weight", type=float, default=1.0,
                   help="非 GT 来源（伪标签，如 scr）的损失权重；GT 恒为 1.0。"
                        "默认 1.0 = 两批数据同等对待；设 0.3 就是给伪标签打折")
    p.add_argument("--no-gt-mask", action="store_true",
                   help="**默认用 GT 的像素级掩码**（masks/<名>.png）作为 root 真值；"
                        "加这个开关才退回 json 多边形 —— 只用于「掩码 vs 多边形」的 A/B 对照")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--patience", type=int, default=config.PATIENCE,
                   help="联合 Dice 连续 N 轮无提升则早停（应 > --lr-patience）")
    p.add_argument("--lr-patience", type=int, default=config.LR_PATIENCE,
                   help="联合 Dice 连续 N 轮无提升则 LR 减半（默认见 config.LR_PATIENCE）")
    p.add_argument("--val-size", type=int, default=None,
                   help="验证集数量：plant 模式是**植株数**（默认 config.VAL_SIZE）；"
                        "image 模式是**样本数**。都不指定时 image 用 --val-frac 自动算")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="image 模式下验证集比例（--val-size 未指定时生效，默认 0.2）")
    p.add_argument("--split-mode", default="image", choices=("plant", "image"),
                   help="train/val 划分方式：image=按单图随机（默认，忽略 VAL_PLANTS，"
                        "同一植株可能同时出现在 train/val）；plant=按植株整组")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--pos-weight", default=None,
                   help="逐通道 BCE 正样本权重，逗号分隔、顺序同 CLASS_NAMES"
                        "（如 3,10,1；默认见 config.LOSS_POS_WEIGHT）。"
                        "走命令行是为了做损失实验时不用改 config，实验值也会记进 hparams.json")
    p.add_argument("--bce-weight", default=None,
                   help="逐通道 BCE 损失权重，逗号分隔、顺序同 CLASS_NAMES"
                        "（默认见 config.LOSS_BCE_W）。例如检查 check 通道塌成常数时"
                        "可调成 1,1,0.3 不变、或单独加 Dice 权重")
    p.add_argument("--dice-weight", default=None,
                   help="逐通道 Dice 损失权重，逗号分隔、顺序同 CLASS_NAMES"
                        "（默认见 config.LOSS_DICE_W，当前 check=0）。"
                        "check 通道只有 BCE、没有 Dice 时，容易在 60~76%% 前景下塌成全 1；"
                        "做 SAM 系列时可试 1,1,1")
    p.add_argument("--data-dir", type=Path, default=config.TRAIN_DATA_DIR)
    p.add_argument("--task", default="root",
                   choices=("root", "stem", "check_background", "all"),
                   help="单独训练的通道（默认 root）。root/stem/check_background 训练"
                        "**单通道模型**（输出 1 通道）；文件夹名分别加后缀 "
                        "''、-stem、-check_background；"
                        "all=旧的三通道联合模型（输出 3 通道），用于兼容历史实验")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="模型输出根目录（默认 config.MODEL_DIR）。单通道任务的文件夹名"
                        "会加后缀：root 不加，stem 加 -stem，check_background 加 "
                        "-check_background")
    p.add_argument("--norm", default=config.NORM, choices=("group", "batch"),
                   help="归一化层：group=GroupNorm（小 batch 稳定）、batch=BatchNorm"
                        "（需 batch≥2，默认见 config.NORM）")
    p.add_argument("--arch", default="unet",
                   choices=("unet", "unet++", "attention_unet", "swin_unet",
                            "segformer", "transunet", "sam", "sam2", "sam3",
                            "medsam", "umamba", "mask2former", "vmunet",
                            "esegformer"),
                   help="骨架：unet=经典 U-Net（默认，历史 baseline）；"
                        "unet++=嵌套跳跃连接（common/unet_plusplus.py）；"
                        "attention_unet=注意力门控跳连（common/attention_unet.py）；"
                        "swin_unet=Swin Transformer U-Net（common/swin_unet.py，"
                        "固定输入尺寸、不支持 --crop、需要 timm/einops；默认加载 "
                        "ImageNet Swin-T 编码器，--swin-pretrained none 关闭）；"
                        "segformer=MiT 编码器 + 轻量解码头（common/segformer.py，"
                        "全卷积、无固定输入尺寸；默认 b2，--segformer-variant 可选 b0~b5）；"
                        "transunet=R50-ViT-B_16 + DecoderCup（common/transunet.py，"
                        "固定输入尺寸、不支持 --crop、默认开梯度检查点；"
                        "默认加载 ImageNet-21k 预训练，--transunet-pretrained none 关闭）；"
                        "sam=SAM 图像编码器（默认冻结）+ 监督解码头（common/sam_seg.py，"
                        "全卷积；默认加载 SA-1B 预训练 ViT-B，--sam-pretrained none 关闭）；"
                        "sam2=SAM 2.1 Hiera 编码器（默认冻结）+ FPN 跳跃解码头"
                        "（common/sam2_seg.py，全卷积；默认加载 Hiera-Tiny，"
                        "--sam2-pretrained none 关闭）；"
                        "sam3=SAM 3 ViT-L/14 视觉骨干（默认冻结）+ ViTDet neck + "
                        "监督解码头（common/sam3_seg.py，内部缩放到 1008x1008；"
                        "默认加载 ModelScope 的 sam3.pt，--sam3-pretrained none 关闭）；"
                        "medsam=MedSAM 医学微调的 SAM ViT-B 编码器（默认冻结）+ "
                        "监督解码头（common/sam_seg.py，按图 min-max 预处理；默认加载 "
                        "medsam_vit_b.pth，--medsam-pretrained none 关闭）；"
                        "umamba=U-Mamba_Bot 2D（common/umamba.py，9 级残差 U-Net + "
                        "最深层纯 PyTorch Mamba；不需要 mamba_ssm/CUDA 扩展；内部 "
                        "pad 到 256 的倍数）；"
                        "mask2former=Mask2Former 简化版（common/mask2former.py，"
                        "ResNet 骨干 + FPN + query-based mask 解码器；纯 PyTorch "
                        "deformable attention，不需要 detectron2；内部 pad 到 32 的倍数）；"
                        "vmunet=VM-UNet-T（common/vmunet.py，VSS 四方向 selective "
                        "scan，纯 PyTorch 分块实现，不需要 mamba_ssm/CUDA 扩展；"
                        "全卷积，内部 pad 到 32 的倍数。注意：SS2D 在 1536 下很慢，"
                        "建议先看 --vmunet-chunk 和训练尺寸说明）；"
                        "esegformer=ESegFormer（RPT 的 root 分割骨架，"
                        "common/esegformer.py；MiT + spatial_attn/CBAM + 5 级解码头，"
                        "默认 b2；--esegformer-pretrained 可加载 RPT 的 model.pdparams "
                        "转出的 PyTorch 权重；不需要 paddle）。"
                        "注意：UNet++ 显存高得多：2048/batch1 实测约 24GB 会 OOM，"
                        "3×3090（GPU0 另有 4GB 占用）建议从 ≤1536 起步；"
                        "Attention U-Net 的显存与 U-Net 接近")
    p.add_argument("--segformer-variant", default="b2",
                   choices=("b0", "b1", "b2", "b3", "b4", "b5"),
                   help="SegFormer 变体（默认 b2，约 27M 参数，与 U-Net 31M 最接近）；"
                        "b0 约 3.7M 适合小数据，b4/b5 显存需求明显更高")
    p.add_argument("--segformer-pretrained", default="none",
                   help="SegFormer 官方 mmseg 预训练权重的路径；none=从零训练（默认）。"
                        "支持 backbone.* / 裸 MiT 键名；只对 --arch segformer 生效")
    p.add_argument("--esegformer-variant", default="b2",
                   choices=("b0", "b1", "b2", "b3", "b4", "b5"),
                   help="ESegFormer 变体（默认 b2，RPT 发布的权重是 B2）")
    p.add_argument("--esegformer-pretrained", default="none",
                   help="RPT ESegFormer-PyTorch 权重路径（由 "
                        "tool/convert_esegformer_paddle.py 从 model.pdparams 转出）；"
                        "none=从零训练（默认）。相对路径按项目根目录解析")
    p.add_argument("--esegformer-freeze-backbone", action="store_true",
                   help="复现 RPT 的 freeze_exceptEAM：冻结骨干，只训 spatial_attn "
                        "和解码头。只有配合 --esegformer-pretrained 才有意义")
    p.add_argument("--esegformer-input-norm", default="pm1",
                   choices=("pm1", "none"),
                   help="模型内部输入归一化：pm1=x*2-1（默认，匹配 RPT 的 "
                        "Normalize(0.5,0.5)）；none=直接用数据管线的 [0,1]")
    p.add_argument("--esegformer-dropout", type=float, default=0.1,
                   help="ESegFormer 解码头 Dropout2d（默认 0.1，与 RPT 一致）")
    p.add_argument("--esegformer-drop-path", type=float, default=0.1,
                   help="ESegFormer 骨干 stochastic depth（默认 0.1，与 RPT 一致）")
    p.add_argument("--transunet-pretrained",
                   default=str(config.TRANSUNET_PRETRAINED_CKPT),
                   help="TransUNet 的 R50+ViT-B_16.npz 预训练权重路径（默认见 "
                        "config.TRANSUNET_PRETRAINED_CKPT）；none=从零训练。"
                        "只对 --arch transunet 生效；相对路径按项目根目录解析")
    p.add_argument("--transunet-no-checkpoint", action="store_true",
                   help="关闭 TransUNet ViT 的梯度检查点：更快，但 1024x1536 下"
                        "12 层、6144 token 的注意力激活很可能在 24GB 上 OOM")
    p.add_argument("--sam-variant", default="vit_b",
                   choices=("vit_b", "vit_l", "vit_h"),
                   help="SAM 图像编码器规模（默认 vit_b，约 90M；vit_h 约 636M，"
                        "24GB 上 1024x1536 很可能 OOM）")
    p.add_argument("--sam-pretrained", default=str(config.SAM_PRETRAINED_CKPT),
                   help="SAM 官方检查点路径（默认见 config.SAM_PRETRAINED_CKPT）；"
                        "none=从零训练。只对 --arch sam 生效；只加载图像编码器")
    p.add_argument("--sam-finetune-encoder", action="store_true",
                   help="解冻 SAM 图像编码器一起训练（默认冻结：71 张图从零微调 90M "
                        "编码器容易过拟合，显存也高得多）")
    p.add_argument("--sam2-variant", default="tiny",
                   choices=("tiny", "small", "base_plus", "large"),
                   help="SAM 2.1 Hiera 规模（默认 tiny，约 27M 编码器参数；"
                        "small/base_plus/large 更大）")
    p.add_argument("--sam2-pretrained", default=str(config.SAM2_PRETRAINED_CKPT),
                   help="SAM 2.1 官方检查点路径（默认见 config.SAM2_PRETRAINED_CKPT）；"
                        "none=从零训练。只对 --arch sam2 生效；只加载 trunk/neck")
    p.add_argument("--sam2-finetune-encoder", action="store_true",
                   help="解冻 SAM 2 Hiera 编码器一起训练（默认冻结）")
    p.add_argument("--sam3-pretrained", default=str(config.SAM3_PRETRAINED_CKPT),
                   help="SAM 3 官方 checkpoint 路径（默认见 config.SAM3_PRETRAINED_CKPT）；"
                        "none=从零训练。只对 --arch sam3 生效；只加载 ViT+neck")
    p.add_argument("--sam3-finetune-encoder", action="store_true",
                   help="解冻 SAM 3 ViT-L 编码器一起训练（默认冻结；455M 参数，"
                        "解冻后显存会大幅上升，71 张图也不建议）")
    p.add_argument("--medsam-pretrained", default=str(config.MEDSAM_PRETRAINED_CKPT),
                   help="MedSAM 官方 medsam_vit_b.pth 路径（默认见 "
                        "config.MEDSAM_PRETRAINED_CKPT）；none=从零训练。"
                        "只对 --arch medsam 生效；只加载图像编码器")
    p.add_argument("--medsam-finetune-encoder", action="store_true",
                   help="解冻 MedSAM ViT-B 编码器一起训练（默认冻结；约 90M 参数，"
                        "71 张图从零微调容易过拟合）")
    p.add_argument("--vmunet-chunk", type=int, default=0,
                   help="VM-UNet selective scan 的块长。0=按序列长度自动"
                        "（16~256，默认）；固定值可手动调。每块内做两遍 chunk 步"
                        "循环 + nc 步块间转移。只对 --arch vmunet 生效")
    p.add_argument("--vmunet-d-state", type=int, default=16,
                   help="VM-UNet SS2D 的状态维度 N（默认 16，与官方一致）；"
                        "调小可明显降低显存和扫描开销，但权重与官方不兼容")
    p.add_argument("--vmunet-drop-path", type=float, default=0.2,
                   help="VM-UNet VSS block 的 stochastic depth 概率（官方默认 0.2）")
    p.add_argument("--vmunet-no-checkpoint", action="store_true",
                   help="关闭 VM-UNet 每个 VSS block 的梯度检查点。默认开启："
                        "1536 下不检查点会保存整条 selective-scan 的中间状态，"
                        "24GB 也会 OOM")
    p.add_argument("--unet-checkpoint", action="store_true",
                   help="U-Net 编解码块启用梯度检查点：2736 + GroupNorm 在 24GB 上 "
                        "OOM 时用（用时间换显存，约慢 20~30%%；只对 --arch unet 生效；"
                        "请配 --norm group，BatchNorm 在重算时会二次更新 running stats）")
    p.add_argument("--unetpp-checkpoint", action="store_true",
                   help="UNet++ 嵌套节点启用梯度检查点：1536/batch1 显存不够时用"
                        "（用时间换显存，约慢 20~30%%；只对 --arch unet++ 生效）")
    p.add_argument("--swin-pretrained", default=str(config.SWIN_PRETRAINED_CKPT),
                   help="Swin-Unet 编码器的 ImageNet 预训练权重路径（默认见 "
                        "config.SWIN_PRETRAINED_CKPT）；none = 从零训练。"
                        "只对 --arch swin_unet 生效。相对路径按项目根目录解析")
    p.add_argument("--workers", type=int, default=config.NUM_WORKERS,
                   help="DataLoader 子进程数（0=主进程里同步加载）。服务器上设 4~8 可让"
                        "读图/增强与训练并行，避免 GPU 空等（默认见 config.NUM_WORKERS）")
    p.add_argument("--no-amp", action="store_true", help="关闭混合精度")
    p.add_argument("--gpu", type=int, default=None,
                   help="指定用哪张 GPU（默认自动挑当前最空的一张）；"
                        "也可以用环境变量 ROOT_MODEL_GPU")
    p.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return p.parse_args()


def split_by_plant(names, val_size, seed, mode="image", val_frac=0.2):
    """划分 train/val：返回 (训练名列表, 验证名列表, 验证植株列表)。

    mode="image"（默认）：按**单图**随机，验证集由 `--seed` 决定，大小
    `val_size`（样本数）或 `val_frac`（比例，默认 0.2）。同一植株的不同时点
    可能同时出现在 train/val 两侧；**忽略 config.VAL_PLANTS**。
    mode="plant"：按植株整组划分，避免同株泄漏；`val_size` 是植株数，
    config.VAL_PLANTS 非空时优先按它钉死，一个都钉不上时退回随机植株划分。
    """
    def _random_val(ks):
        rng = np.random.RandomState(seed)
        rng.shuffle(ks)
        n = min(max(val_size, 0), max(len(ks) - 1, 0))   # **至少给训练留 1 株**
        return sorted(ks[:n])

    if mode == "image":
        arr = sorted(names)
        rng = random.Random(seed)
        rng.shuffle(arr)
        if val_size is None:
            n = max(1, int(round(len(arr) * val_frac)))
        else:
            n = int(val_size)
        n = min(max(n, 0), max(len(arr) - 1, 0))
        val_names = sorted(arr[:n])
        train_names = sorted(arr[n:])
        val_plants = sorted({plant_key(n) for n in val_names})
        print(f"[划分] image 模式：按单图随机（seed={seed}），验证 "
              f"{len(val_names)} 组（{len(val_plants)} 植株，同株可能跨侧）")
        return train_names, val_names, val_plants

    if val_size is None:
        val_size = config.VAL_SIZE
    groups = {}
    for n in names:
        groups.setdefault(plant_key(n), []).append(n)
    keys = sorted(groups)
    if config.VAL_PLANTS:
        pinned = [k for k in keys if k in set(config.VAL_PLANTS)]
        missing = sorted(set(config.VAL_PLANTS) - set(pinned))
        if missing:
            print(f"[警告] config.VAL_PLANTS 里的植株不在数据集中: {missing}")
        if pinned:
            val_plants = pinned
        else:
            val_plants = _random_val(keys)
            print(f"[警告] VAL_PLANTS 一个都没对上（换了数据集？）→ 退回随机划分 "
                  f"{len(val_plants)} 植株: {val_plants}")
    else:
        val_plants = _random_val(keys)
    val_set = set(val_plants)
    val_names = sorted(n for k in val_plants for n in groups[k])
    train_names = sorted(n for k, v in groups.items() if k not in val_set for n in v)
    return train_names, val_names, val_plants


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- 单通道 / 三通道任务 ----
    task = args.task
    task_ch = None if task == "all" else TASK_TO_CH[task]
    task_name = task
    model_out_ch = N_CH if task_ch is None else 1
    out_dir = config.MODEL_DIR if args.out_dir is None else args.out_dir
    # 单通道任务的模型文件夹加后缀；root / all 不加
    folder_suffix = "" if task in ("root", "all") else f"-{task}"
    if task_ch is None:
        print("[任务] all：三通道联合模型（旧行为）")
    else:
        print(f"[任务] {task_name}：单通道模型（输出 1 通道），"
              f"模型文件夹后缀 '{folder_suffix}'")

    # 逐通道权重：命令行可给 1 个数（只作用于当前单通道任务）或 3 个数
    # （顺序同 CLASS_NAMES）。长度不对会静默广播错，所以这里严格校验。
    def _parse_channel_weights(value, default, name):
        if value is None:
            out = tuple(default)
        else:
            try:
                out = tuple(float(v) for v in value.split(","))
            except ValueError:
                sys.exit(f"[错误] {name} 解析失败: {value}（应为 1,1,1 或单个数）")
        if len(out) == N_CH:
            return out
        if len(out) == 1 and task_ch is not None:
            return tuple(out[0] if i == task_ch else default[i]
                         for i in range(N_CH))
        sys.exit(f"[错误] {name} 需要 1 个数（单通道任务）或 {N_CH} 个数"
                 f"（顺序 {config.CLASS_NAMES}），当前 {len(out)} 个: {value}")

    pos_weight = _parse_channel_weights(args.pos_weight, config.LOSS_POS_WEIGHT,
                                        "--pos-weight")
    bce_w = _parse_channel_weights(args.bce_weight, config.LOSS_BCE_W,
                                   "--bce-weight")
    # 单独训练 check 时默认打开 check 的 Dice：只给 BCE 会塌成全 1（见 train/readme）
    dice_default = list(config.LOSS_DICE_W)
    if task_ch == TASK_TO_CH["check_background"]:
        dice_default[task_ch] = 1.0
    dice_w = _parse_channel_weights(args.dice_weight, dice_default,
                                    "--dice-weight")
    if task_ch is not None:
        bce_w = (bce_w[task_ch],)
        dice_w = (dice_w[task_ch],)
        pos_weight = (pos_weight[task_ch],)
        cldice_w = (config.LOSS_CLDICE_W[task_ch],)
    else:
        cldice_w = tuple(config.LOSS_CLDICE_W)

    # ---- 多卡（DDP）：torchrun 起才有 LOCAL_RANK，否则就是原来的单卡单进程 ----
    ddp = device_mod.Dist(cpu=args.cpu)
    if ddp.enabled:
        # 多卡时**不能用自动挑卡**：每张卡固定绑一个 rank，挑了也没用
        device = ddp.init(need_gb=NEED_GB)
    else:
        # 自动挑当前最空的 GPU（不再写死 cuda:0 —— 服务器上 4 张卡常有人占着）
        device = device_mod.pick(cpu=args.cpu, gpu=getattr(args, "gpu", None),
                                 need_gb=NEED_GB)
    if device.type == "cpu":
        print("[警告] 使用 CPU 训练，速度很慢。pcc 环境支持 CUDA（RTX 5060）。")

    # ---- 数据划分：按植株整组进出（保证可复现） ----
    names = [n for n, _, _ in _pairs(args.data_dir)]
    train_names, val_names, val_plants = split_by_plant(
        names, args.val_size, args.seed, mode=args.split_mode,
        val_frac=args.val_frac)
    train_plants = sorted({plant_key(n) for n in train_names})
    print(f"数据: 共 {len(names)} 组 / {len(set(map(plant_key, names)))} 植株 | "
          f"训练 {len(train_names)} 组({len(train_plants)} 植株) | "
          f"验证 {len(val_names)} 组({len(val_plants)} 植株)")
    print(f"验证植株: {', '.join(val_plants) if val_plants else '无(不早停,保存最后轮)'}")
    t0 = time.time()
    train_ds = RootDataset(args.data_dir, names=train_names,
                           use_mask=not args.no_gt_mask,
                           max_side=args.size, augment=True, seed=args.seed,
                           crop=args.crop, crop_repeat=args.crop_repeat)
    # 验证集的预处理必须与**部署形式**一致：
    #   切片模型  -> full=True：整图、原始分辨率、不缩放，验证时逐块滑窗前向（见下面的验证段）
    #   非切片模型 -> 缩放到 --size、整图一次前向（老路径，不变）
    # 曾经给切片模型用的是「单个中心裁块」，那是个**退化**的验证：块里的 check GT 几乎
    # 全是 True，预测全 True 就能拿 Dice 0.99（实测日志里从第 21 轮起恒为 0.99），
    # 于是 select_dice / 早停 / LR 调度全在一个假信号上跑 —— 整轮实验都是盲的。
    val_ds = RootDataset(args.data_dir, names=val_names,
                         use_mask=not args.no_gt_mask,
                         max_side=args.size, augment=False, seed=args.seed,
                         crop=0, crop_repeat=1, full=bool(args.crop))
    # 切片模式下 len(ds) = 图片数 × crop_repeat，所以断言要按图片数比
    assert len(train_ds) == len(train_names) * train_ds.repeat, \
        "训练集样本数不符（名字对不上？）"
    assert len(val_ds) == len(val_names) * val_ds.repeat, \
        "验证集样本数不符（名字对不上？）"
    print(f"数据加载完成，用时 {time.time() - t0:.1f}s")

    # DDP：每张卡只看自己那一份（DistributedSampler 把数据集切成 world 份），
    # 于是"一个 epoch"仍然等于整份数据过一遍 —— 轮数的语义在单卡/多卡下一致。
    train_sampler = None
    if ddp.enabled:
        train_sampler = torch.utils.data.DistributedSampler(
            train_ds, num_replicas=ddp.world, rank=ddp.rank, shuffle=True)
        if ddp.is_main:
            print(f"[DDP] 每张卡每轮 {len(train_sampler)} 个样本"
                  f"（全量 {len(train_ds)} / {ddp.world} 卡）")
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, sampler=train_sampler,
        shuffle=train_sampler is None, drop_last=True,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0)
    # 切片模型的验证是「逐张整图滑窗」，用不上 DataLoader；而且硬建一个会让每个
    # worker 都持有一份整图（Windows 的 spawn 是真拷贝，几百 MB × workers）。
    val_loader = None
    if not args.crop:
        # 验证集也分片：每张卡算自己那份，最后把**逐样本指标**跨卡汇总
        # （见 _reduce_val —— 不能简单对平均值再平均，各卡样本数可能不等）
        val_sampler = None
        if ddp.enabled:
            val_sampler = torch.utils.data.DistributedSampler(
                val_ds, num_replicas=ddp.world, rank=ddp.rank, shuffle=False)
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=args.batch, sampler=val_sampler, shuffle=False,
            num_workers=args.workers, persistent_workers=args.workers > 0)
    if args.workers == 0 and device.type == "cuda":
        print("[提示] num_workers=0：读图与数据增强在主进程里同步做，GPU 会空等。"
              "服务器上可加 --workers 8（本机 Windows 保持 0 即可）。")

    # ---- 逐样本的来源权重（伪标签降权）----
    # GT（人工标注 + 像素掩码）恒为 1.0；其余来源乘 --src-weight。
    # 加权的**方式**很关键：把它乘进损失的逐样本权重里，让分母（wb.sum()）跟着一起变 ——
    # 那才是"这条样本对梯度的贡献打折"；只把损失值乘小、分母不动的话，会被整体稀释掉。
    src_w_lut = {it["name"]: (1.0 if (it["src"] in ("", "GT")) else args.src_weight)
                 for it in train_ds.items}
    use_src_w = any(v != 1.0 for v in src_w_lut.values())
    if use_src_w:
        n_gt = sum(1 for v in src_w_lut.values() if v == 1.0)
        print(f"[来源权重] GT {n_gt} 组 ×1.0 | 其余 {len(src_w_lut) - n_gt} 组 "
              f"×{args.src_weight}")

    # ---- 固定输入尺寸的骨架（Swin-Unet / TransUNet）----
    # 两者都不是全卷积模型（Swin 的 PatchEmbed 有 assert；TransUNet 的位置编码和
    # DecoderCup 网格在构造时固定），必须按训练时的精确 (H, W) 实例化；
    # 因此不接受 --crop（切片训练/滑窗验证会换尺寸）。
    arch_img_size = None
    if args.arch in ("swin_unet", "transunet"):
        if args.crop:
            sys.exit(f"[错误] {args.arch} 是固定尺寸模型，"
                     f"不支持 --crop 切片训练/滑窗验证")
        h1, w1 = train_ds.items[0]["img"].shape[:2]
        arch_img_size = (int(h1), int(w1))
        stride_req = 32 if args.arch == "swin_unet" else 16
        if h1 % stride_req or w1 % stride_req:
            sys.exit(f"[错误] {args.arch} 要求输入 H/W 能被 {stride_req} 整除，"
                     f"--size {args.size} 得到 {h1}×{w1}；请换一个 --size"
                     f"（如 768、1536）")
        for it in train_ds.items:
            if it["img"].shape[:2] != (h1, w1):
                sys.exit(f"[错误] {args.arch} 要求所有输入同尺寸：{it['name']} 是 "
                         f"{tuple(it['img'].shape[:2])}，首张是 {(h1, w1)}")
        print(f"[{args.arch}] 固定输入尺寸 {arch_img_size[0]}×{arch_img_size[1]}")

    # ---- 模型 ----
    if args.norm == "batch" and args.batch < 2 and not ddp.enabled:
        # DDP 下不适用：下面会把 BatchNorm 换成 SyncBatchNorm，统计量跨卡算
        # （每卡 1 张 × 3 卡 = 3 张），不是 config.NORM 里说的那种单卡 batch=1
        print(f"[警告] 归一化用 BatchNorm 但 batch={args.batch}：batch=1 时读到的统计量"
              f"与推理用的滑动平均对不上，会出现严重欠分割。请用 batch≥2 或把 "
              f"config.NORM 改成 'group'。")
    if args.arch == "unet++":
        model = UNetPlusPlus(in_ch=3, out_ch=model_out_ch, norm=args.norm,
                             use_checkpoint=args.unetpp_checkpoint)
    elif args.arch == "attention_unet":
        model = AttentionUNet(in_ch=3, out_ch=model_out_ch, norm=args.norm)
    elif args.arch == "swin_unet":
        # 延迟导入：Swin 依赖 timm/einops，没装的环境仍能跑其它骨架
        from common.swin_unet import SwinUNet2D, load_pretrained_encoder
        model = SwinUNet2D(in_ch=3, out_ch=model_out_ch, img_size=arch_img_size,
                           norm=args.norm)
        # ImageNet 预训练编码器（默认开，--swin-pretrained none 关闭）。
        # **在包 DDP / 搬上 GPU 之前加载**：各 rank 从同一份 CPU 权重出发，结果一致。
        if args.swin_pretrained and args.swin_pretrained.lower() != "none":
            ckpt_path = Path(args.swin_pretrained)
            if not ckpt_path.is_absolute():
                ckpt_path = config.PROJECT_ROOT / ckpt_path
            # 多卡时只在主进程打日志，避免 3 行一样的输出
            load_pretrained_encoder(model, ckpt_path,
                                    verbose=(not ddp.enabled or ddp.is_main))
        elif not ddp.enabled or ddp.is_main:
            print("[Swin 预训练] 已关闭（--swin-pretrained none），从零训练")
    elif args.arch == "segformer":
        # 延迟导入：依赖 timm（DropPath 等），没装 timm 的环境仍能跑其它骨架
        from common.segformer import SegFormer2D, load_pretrained_segformer
        model = SegFormer2D(in_ch=3, out_ch=model_out_ch,
                            variant=args.segformer_variant, norm=args.norm)
        # 可选预训练：官方 mmseg 权重（默认 none = 从零）
        if args.segformer_pretrained.lower() != "none":
            ckpt_path = Path(args.segformer_pretrained)
            if not ckpt_path.is_absolute():
                ckpt_path = config.PROJECT_ROOT / ckpt_path
            load_pretrained_segformer(model, ckpt_path,
                                      verbose=(not ddp.enabled or ddp.is_main))
        elif not ddp.enabled or ddp.is_main:
            print(f"[SegFormer] variant={args.segformer_variant}，从零训练"
                  f"（--segformer-pretrained 可指定官方权重）")
    elif args.arch == "esegformer":
        # RPT 的 ESegFormer：MiT + spatial_attn(CBAM) + 5 级解码头。
        # 不依赖 paddle；预训练权重需先用 tool/convert_esegformer_paddle.py 转换。
        from common.esegformer import ESegFormer2D, load_pretrained_esegformer
        model = ESegFormer2D(
            in_ch=3, out_ch=model_out_ch,
            variant=args.esegformer_variant, norm=args.norm,
            dropout=args.esegformer_dropout,
            drop_path_rate=args.esegformer_drop_path,
            input_norm=args.esegformer_input_norm,
            freeze_backbone=args.esegformer_freeze_backbone)
        if args.esegformer_pretrained.lower() != "none":
            ckpt_path = Path(args.esegformer_pretrained)
            if not ckpt_path.is_absolute():
                ckpt_path = config.PROJECT_ROOT / ckpt_path
            load_pretrained_esegformer(
                model, ckpt_path,
                verbose=(not ddp.enabled or ddp.is_main))
            if args.esegformer_freeze_backbone and (
                    not ddp.enabled or ddp.is_main):
                print("[ESegFormer] 已冻结骨干和 linear_c1..c4，只训练 "
                      "spatial_attn + linear_c5 + fuse + pred"
                      "（RPT 的 freeze_exceptEAM 策略）")
        elif args.esegformer_freeze_backbone and (
                not ddp.enabled or ddp.is_main):
            print("[ESegFormer] 警告：没有预训练权重却要求冻结骨干，"
                  "spatial_attn 之外的骨干是随机初始化；建议去掉 "
                  "--esegformer-freeze-backbone 或加载预训练")
    elif args.arch == "transunet":
        # 延迟导入：TransUNet 依赖 numpy（已装），不需要额外环境
        from common.transunet import TransUNet2D, load_pretrained_transunet
        model = TransUNet2D(in_ch=3, out_ch=model_out_ch, img_size=arch_img_size,
                            norm=args.norm,
                            use_checkpoint=not args.transunet_no_checkpoint)
        if args.transunet_pretrained.lower() != "none":
            ckpt_path = Path(args.transunet_pretrained)
            if not ckpt_path.is_absolute():
                ckpt_path = config.PROJECT_ROOT / ckpt_path
            load_pretrained_transunet(
                model, ckpt_path, verbose=(not ddp.enabled or ddp.is_main))
        elif not ddp.enabled or ddp.is_main:
            print("[TransUNet] 已关闭预训练（--transunet-pretrained none），从零训练")
    elif args.arch == "sam":
        # 延迟导入；SAM 编码器 vendored 在 common/sam_seg.py，不依赖 segment_anything 包
        from common.sam_seg import SAMSeg2D, load_pretrained_sam
        model = SAMSeg2D(in_ch=3, out_ch=model_out_ch, variant=args.sam_variant,
                         norm=args.norm, pretrained=None,
                         freeze_encoder=not args.sam_finetune_encoder)
        if args.sam_pretrained.lower() != "none":
            ckpt_path = Path(args.sam_pretrained)
            if not ckpt_path.is_absolute():
                ckpt_path = config.PROJECT_ROOT / ckpt_path
            load_pretrained_sam(model, ckpt_path,
                                verbose=(not ddp.enabled or ddp.is_main))
        elif not ddp.enabled or ddp.is_main:
            print(f"[SAM] {args.sam_variant} 从零训练"
                  f"（--sam-pretrained 可指定官方权重）")
    elif args.arch == "sam2":
        # 延迟导入；Hiera 编码器 vendored 在 common/sam2_seg.py
        from common.sam2_seg import SAM2Seg2D, load_pretrained_sam2
        model = SAM2Seg2D(in_ch=3, out_ch=model_out_ch, variant=args.sam2_variant,
                          norm=args.norm, pretrained=None,
                          freeze_encoder=not args.sam2_finetune_encoder)
        if args.sam2_pretrained.lower() != "none":
            ckpt_path = Path(args.sam2_pretrained)
            if not ckpt_path.is_absolute():
                ckpt_path = config.PROJECT_ROOT / ckpt_path
            load_pretrained_sam2(model, ckpt_path,
                                 verbose=(not ddp.enabled or ddp.is_main))
        elif not ddp.enabled or ddp.is_main:
            print(f"[SAM2] {args.sam2_variant} 从零训练"
                  f"（--sam2-pretrained 可指定官方权重）")
    elif args.arch == "sam3":
        # 延迟导入；ViT-L + ViTDet neck 已 vendored 在 common/sam3_seg.py
        from common.sam3_seg import SAM3Seg2D, load_pretrained_sam3
        model = SAM3Seg2D(in_ch=3, out_ch=model_out_ch, norm=args.norm,
                          pretrained=None,
                          freeze_encoder=not args.sam3_finetune_encoder)
        if args.sam3_pretrained.lower() != "none":
            ckpt_path = Path(args.sam3_pretrained)
            if not ckpt_path.is_absolute():
                ckpt_path = config.PROJECT_ROOT / ckpt_path
            load_pretrained_sam3(model, ckpt_path,
                                 verbose=(not ddp.enabled or ddp.is_main))
        elif not ddp.enabled or ddp.is_main:
            print("[SAM3] 从零训练（--sam3-pretrained 可指定官方权重）")
    elif args.arch == "medsam":
        # 延迟导入；MedSAM ViT-B 与 SAM 共用编码器实现，只是预训练和预处理不同
        from common.sam_seg import MedSAMSeg2D, load_pretrained_medsam
        model = MedSAMSeg2D(in_ch=3, out_ch=model_out_ch, norm=args.norm,
                            pretrained=None,
                            freeze_encoder=not args.medsam_finetune_encoder)
        if args.medsam_pretrained.lower() != "none":
            ckpt_path = Path(args.medsam_pretrained)
            if not ckpt_path.is_absolute():
                ckpt_path = config.PROJECT_ROOT / ckpt_path
            load_pretrained_medsam(model, ckpt_path,
                                   verbose=(not ddp.enabled or ddp.is_main))
        elif not ddp.enabled or ddp.is_main:
            print("[MedSAM] 从零训练（--medsam-pretrained 可指定官方权重）")
    elif args.arch == "umamba":
        # 纯 PyTorch 的 U-Mamba_Bot 2D；不需要 mamba_ssm / CUDA 扩展
        from common.umamba import UMambaBot2D
        model = UMambaBot2D(in_ch=3, out_ch=model_out_ch, norm=args.norm)
    elif args.arch == "mask2former":
        # 自包含的 Mask2Former 简化版；不需要 detectron2 / CUDA 扩展
        from common.mask2former import Mask2Former2D
        model = Mask2Former2D(in_ch=3, out_ch=model_out_ch, norm=args.norm)
    elif args.arch == "vmunet":
        # 纯 PyTorch 的 VM-UNet-T；SS2D 用分块 selective scan，不需要 mamba_ssm。
        # 结构内部全部是 LayerNorm，不跟随 --norm。
        from common.vmunet import VMUNet2D
        model = VMUNet2D(in_ch=3, out_ch=model_out_ch,
                         d_state=args.vmunet_d_state,
                         drop_path_rate=args.vmunet_drop_path,
                         use_checkpoint=not args.vmunet_no_checkpoint,
                         chunk=args.vmunet_chunk)
    else:
        model = UNet(in_ch=3, out_ch=model_out_ch, norm=args.norm,
                     use_checkpoint=args.unet_checkpoint)
    # 只在 CUDA 上换 SyncBN：CPU/gloo 下 torch 会直接报
    # "SyncBatchNorm layers only work with GPU modules"
    if ddp.enabled and args.norm == "batch" and device.type == "cuda":
        # **必须换 SyncBatchNorm**：DDP 下每张卡只算自己那批的统计量，4 卡 × batch2
        # 时每张卡只看到 2 张图 —— 正好踩在 config 里记的"batch=1 严重欠分割"的边上。
        # SyncBN 每步跨卡同步统计量，等效于"单卡跑 global batch"，与多卡前的结果可比。
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if ddp.is_main:
            print("[DDP] BatchNorm -> SyncBatchNorm（跨卡统计，等效于单卡的等效 batch）")
    model = model.to(device)
    if ddp.enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[ddp.local_rank] if device.type == "cuda" else None)
    # 存权重/取参数都要用**没包过的那份**：DDP 包装后 state_dict() 的键会多一层
    # "module." 前缀，存出来的 ckpt 用 ckpt.load_unet 加载会直接报键不匹配。
    raw_model = model.module if hasattr(model, "module") else model
    n_params = sum(p.numel() for p in raw_model.parameters())
    # 等效 batch 要把**卡数**算进去（DDP 下 --batch 是每张卡的）
    eff_batch = args.batch * args.accum * (ddp.world if ddp.enabled else 1)
    print(f"{args.arch} 参数量: {n_params / 1e6:.2f}M | "
          + (f"切片训练 块 {args.crop}×{args.crop}（原图不缩放）"
             if args.crop else f"整图缩放到长边 {args.size}")
          + f" | batch {args.batch}" + (f"×累积{args.accum}={eff_batch}"
                                       if args.accum > 1 else "")
          + f" | 输出 {model_out_ch} 通道 "
          + (f"('{task_name}',)" if task_ch is not None
             else str(config.CLASS_NAMES)))
    if args.crop and not args.workers:
        print("[警告] 切片训练下 num_workers=0 会让主进程同步切图+增强，GPU 空等。"
              "服务器上务必 --workers 8。")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=config.WEIGHT_DECAY)
    # 按验证指标自动降 LR。不用 CosineAnnealingLR(T_max=args.epochs)：T_max 是「上限轮数」
    # 而不是「实际会跑多少轮」，而早停总在上限之前触发，余弦因此永远走不到 —— 见 config.LR_PATIENCE
    # 的注释（实测全程恒定 1e-3，val 卡住不动）。
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.LR_FACTOR,
        patience=args.lr_patience, min_lr=config.MIN_LR)
    cur_lr = args.lr
    amp = (device.type == "cuda") and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    # 逐通道正样本加权，(1,C,1,1) 对应 (B,C,H,W) 的通道维（见 config.LOSS_POS_WEIGHT）
    pos_w = torch.as_tensor(pos_weight, dtype=torch.float32,
                            device=device).view(1, model_out_ch, 1, 1)
    # clDice 只对权重非零的通道计算（软骨架迭代贵，别浪费在茎/检查范围上）
    cld_chans = [c for c, w in enumerate(cldice_w)
                 if w > 0.0] if config.CLDICE_ITERS > 0 else []

    # ---- 输出目录：model/model_年月日时分，重名追加 -1/-2… ----
    ts = naming.timestamp()
    # create_unique_dir 而非 unique_path：同一分钟内启动两个训练（一张卡一个）时，
    # 「先查后建」的写法会让两个进程拿到同一个名字、其中一个直接崩；这里是原子创建。
    # 只有 rank0 建目录（4 个进程同时 create_unique_dir 会各拿到一个 -1/-2 后缀），
    # 再把名字广播给其余 rank —— 它们虽然不写文件，但 hparams 里的 model_name 要一致。
    if ddp.enabled:
        folder_name = ddp.broadcast_object(
            naming.model_folder_name(ts) + folder_suffix
            if ddp.is_main else None)
        folder = out_dir / folder_name
        if ddp.is_main:
            folder.mkdir(parents=True, exist_ok=True)
        ddp.barrier()
    else:
        folder = naming.create_unique_dir(
            out_dir, naming.model_folder_name(ts) + folder_suffix)
    ckpt_path = folder / f"{folder.name}.pth"
    log_path = folder / f"{folder.name}_log.txt"
    hparams = {k: (str(v) if isinstance(v, Path) else v)
               for k, v in vars(args).items()}
    hparams.update({"arch_img_size": list(arch_img_size) if arch_img_size else None,
                    "device": str(device),
                    "gpu": torch.cuda.get_device_name(device.index or 0)
                    if device.type == "cuda" else "cpu",
                    "ddp_world_size": ddp.world, "ddp_rank": ddp.rank,
                    "ddp_backend": ddp.backend,
                    "val_names": val_names, "train_names": train_names,
                    "val_plants": val_plants, "train_plants": train_plants,
                    "task": task,
                    "class_names": ([task_name] if task_ch is not None
                                    else list(config.CLASS_NAMES)),
                    "out_ch": model_out_ch,
                    "norm": args.norm,
                    "loss_bce_w": list(bce_w),
                    "loss_dice_w": list(dice_w),
                    "loss_pos_weight": list(pos_weight),
                    "loss_cldice_w": list(cldice_w),
                    "cldice_iters": config.CLDICE_ITERS,
                    "params_M": round(n_params / 1e6, 2),
                    "start": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model_name": folder.name})
    if ddp.is_main:
        with open(folder / "hparams.json", "w", encoding="utf-8") as f:
            json.dump(hparams, f, ensure_ascii=False, indent=2)

    def log(msg, console=True):
        """日志与打印**只有 rank0 做** —— 否则几个进程会往同一个文件里交错写好几份。"""
        if not ddp.is_main:
            return
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        if console:
            print(msg)

    log(f"===== 训练开始 {hparams['start']} =====\n"
        f"hparams: {json.dumps(hparams, ensure_ascii=False)}", console=False)
    log(f"[信息] {hparams['gpu']} | 参数量 {n_params/1e6:.2f}M | "
        f"输入 {args.size} | batch {args.batch}"
        # DDP 下 --batch 是**每张卡**的，这里把跨卡的等效 batch 也写出来，
        # 免得拿多卡的数字跟单卡历史值直接比
        + (f"×{ddp.world}卡={args.batch * ddp.world}" if ddp.enabled else "")
        + (f"×累积{args.accum}={eff_batch}" if args.accum > 1 else "")
        + f" | lr {args.lr} | "
        f"epochs {args.epochs} | 训练 {len(train_names)} 组 | "
        f"验证 {len(val_names)} 组({len(val_plants)} 植株)")
    log(f"[信息] 任务 {task_name} | "
        f"通道 {([task_name] if task_ch is not None else list(config.CLASS_NAMES))} | "
        f"BCE权重 {bce_w} | Dice权重 {dice_w} | 正样本权重 {pos_weight}")
    log(f"[信息] 学习率 {args.lr} | 平台期 {args.lr_patience} 轮不减半就 ×{config.LR_FACTOR}"
        f"（下限 {config.MIN_LR}）| 早停 {args.patience} 轮")
    log(f"[信息] clDice 拓扑损失: "
        + (f"通道 {cld_chans} 权重 {cldice_w} iters {config.CLDICE_ITERS}"
           if cld_chans else "关闭"))
    log(f"[信息] 模型目录: {folder}")

    # ---- 训练循环 ----
    # best_select = 保存判据（联合指标），best_root/best_stem = 那一轮的逐通道值（只为日志）
    best_select, best_root, best_stem, best_epoch = -1.0, -1.0, -1.0, -1
    bad_epochs = 0
    epoch, val_dice = 0, -1.0
    # 先摆一份默认值：第一轮验证之前就 Ctrl-C 的话，异常分支里要用到它们
    per_ch_dice = [-1.0] * model_out_ch
    select_dice = -1.0
    score_hist = deque(maxlen=max(config.SELECT_SMOOTH, 1))
    t_start = time.time()
    try:
        for epoch in range(1, args.epochs + 1):
            t_ep = time.time()
            # DDP：不给 sampler 播种的话，每轮的洗牌顺序**一模一样**（相当于没有 shuffle）
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            loss_sum, n_batch = 0.0, 0
            cd_sum, cd_valid, cd_degen = 0.0, 0, 0
            optimizer.zero_grad(set_to_none=True)
            n_micro = 0
            for i_batch, (x, y, batch_names, valid) in enumerate(loader):
                x, y, valid = x.to(device), y.to(device), valid.to(device)
                if task_ch is not None:
                    # 单通道任务：只监督当前通道；模型输出也是 1 通道
                    y = y[:, task_ch:task_ch + 1]
                    valid = valid[:, task_ch:task_ch + 1]
                sw = None
                if use_src_w:
                    sw = torch.tensor([src_w_lut[n] for n in batch_names],
                                      dtype=torch.float32, device=device)[:, None]
                with torch.autocast(device_type="cuda", enabled=amp):
                    out = model(x)
                    prob = torch.sigmoid(out)
                    # 逐通道 BCE（缺标注的通道权重为 0）：用 reduction="none" 再按通道
                    # 加权，避免直接对 (B,C,H,W) 求 mean 时被大面积的检查范围通道带偏。
                    # pos_weight 必须是 (1,C,1,1) 才能对 (B,C,H,W) 正确广播到逐通道
                    # （写成 (C,) 或 (C,1,1) 会广播到 batch 维，静默算错）。
                    bce = F.binary_cross_entropy_with_logits(
                        out.float(), y, pos_weight=pos_w,
                        reduction="none").mean(dim=(2, 3))
                    wb = channel_weights(bce_w, valid)
                    if sw is not None:
                        wb = wb * sw              # (B,C) × (B,1)，逐样本加权
                    loss_bce = (bce * wb).sum() / wb.sum().clamp(min=1e-6)
                    wd = channel_weights(dice_w, valid)
                    if sw is not None:
                        wd = wd * sw
                    loss_dice = ((dice_loss(prob.float(), y) * wd).sum()
                                 / wd.sum().clamp(min=1e-6))
                    loss = loss_bce + loss_dice
                    # clDice：只算权重非零的通道（软骨架迭代较贵，别浪费在茎/检查范围上）
                    if cld_chans:
                        cd, dg = cldice_loss(prob.float(), y, config.CLDICE_ITERS,
                                             cld_chans)
                        wc = channel_weights(cldice_w, valid)[:, cld_chans]
                        # 来源权重也要乘到 clDice 上（2026-10-18 补）：上面 BCE / Dice
                        # 都已乘 sw，这里漏掉的话 `--src-weight 0.3` 对伪标签的折扣
                        # 在拓扑项上完全失效 —— 一项按 0.3、一项按 1.0，等于没打对折。
                        if sw is not None:
                            wc = wc * sw
                        loss = loss + (cd * wc).sum() / wc.sum().clamp(min=1e-6)
                        # 统计只算非退化样本，否则退化的 0 会把均值拉低、掩盖问题
                        ok = ~dg
                        cd_sum += float(cd[ok].sum())
                        cd_valid += int(ok.sum())
                        cd_degen += int(dg.sum())
                # 梯度累积：把 loss 按累积步数缩放，攒够 accum 个 micro-batch 再更新一次。
                # DDP 下**中间那几步不必同步梯度** —— 默认每步都 all-reduce，累积 N 步
                # 就白通信 N-1 次（小 batch 下这个开销占比不小）。
                n_micro += 1
                need_step = n_micro >= args.accum or (i_batch + 1) == len(loader)
                ctx = (model.no_sync() if (ddp.enabled and not need_step
                                           and hasattr(model, "no_sync"))
                       else contextlib.nullcontext())
                with ctx:
                    scaler.scale(loss / args.accum).backward()
                if need_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    n_micro = 0
                loss_sum += float(loss.item())      # 记录未缩放的损失，便于跨配置比较
                n_batch += 1
            train_loss = loss_sum / max(n_batch, 1)

            # ---- 验证（轮内）：逐通道 Dice/IoU，早停/保存按联合指标（见下面 select_dice）----
            val_dice = val_iou = -1.0
            per_ch_dice = [-1.0] * model_out_ch
            if val_ds:
                model.eval()
                d_all, i_all = [], []
                if args.crop:
                    # 切片模型：**按部署形式验证** —— 原图分块滑窗、概率层拼接、
                    # 再与整图真值比。与「单个裁块」相比，这里的 check 通道能真正被量到
                    # （裁块的 GT 几乎全 True，预测全 True 就 0.99，是退化的）。
                    from common import predict as _predict
                    # DDP：val_ds 这里是直接遍历的（切片模型不走 DataLoader），
                    # 所以自己按 rank 切片 —— 不然每张卡都验全量，重复 world 倍
                    _vds = val_ds
                    if ddp.enabled:
                        _vds = torch.utils.data.Subset(
                            val_ds, list(range(ddp.rank, len(val_ds), ddp.world)))
                    with torch.no_grad():
                        for x, y, _, valid in _vds:     # 逐张：整图 5472x3648 没法 batching
                            img = (x.permute(1, 2, 0).numpy() * 255.0).round() \
                                .astype(np.uint8)
                            prob = _predict.tiled_probs(model, img, args.crop,
                                                        device=device)
                            pb = prob.numpy()[0] > 0.5
                            gt = y.numpy() > 0.5
                            vd = valid.numpy()
                            if task_ch is not None:
                                gt = gt[task_ch:task_ch + 1]
                                vd = vd[task_ch:task_ch + 1]
                            _accumulate(pb, gt, vd, d_all, i_all)
                else:
                    with torch.no_grad():
                        for x, y, _, valid in val_loader:
                            x = x.to(device)
                            with torch.autocast(device_type="cuda", enabled=amp):
                                prob = torch.sigmoid(model(x)).float()
                            pb = prob.cpu().numpy() > 0.5
                            if task_ch is not None:
                                gt = y.numpy()[:, task_ch:task_ch + 1] > 0.5
                                vd = valid.numpy()[:, task_ch:task_ch + 1]
                            else:
                                gt = y.numpy() > 0.5
                                vd = valid.numpy()
                            for i in range(len(gt)):
                                _accumulate(pb[i], gt[i], vd[i], d_all, i_all)
                if d_all:
                    if ddp.enabled:
                        # 各卡只验了自己那份，先跨卡并成全局均值（结果 = 整份验证集的 nanmean）
                        per_ch_dice = list(_reduce_val_metric(d_all, ddp))
                        per_ch_iou = list(_reduce_val_metric(i_all, ddp))
                    else:
                        with np.errstate(invalid="ignore"):
                            per_ch_dice = list(np.nanmean(np.asarray(d_all), axis=0))
                            per_ch_iou = list(np.nanmean(np.asarray(i_all), axis=0))
                    # 整个验证集都没有该通道标注时 nanmean 会返回 nan，统一记成 -1；
                    # 一律转成 Python float：numpy 标量写进 ckpt 会让 torch.load 的
                    # weights_only 模式（torch>=2.6 默认）拒绝加载
                    per_ch_dice = [float(-1.0 if np.isnan(v) else v) for v in per_ch_dice]
                    per_ch_iou = [float(-1.0 if np.isnan(v) else v) for v in per_ch_iou]
                    val_dice, val_iou = per_ch_dice[0], per_ch_iou[0]

            # ---- 选模型 / 早停 / 调 LR 的判据：select_dice ----
            # 为什么不单看根系通道（2026-09-19 实测踩到）：根系 Dice 先到峰值、茎通道后
            # 收敛，两者高峰错开一整段。只盯根系就会存下「根系峰值已到、茎还没练好」的
            # 权重；当年这条判据的直接动机是「锚定依赖茎预测，茎一塌总长/root Dice 一起
            # 被拖垮」（锚定已于 2026-10-05 删除，这条传导链不再存在，但塌茎权重本身
            # 仍与整体欠训练相关，门槛暂按原值保留）。
            #
            # **茎当准入门槛**：茎 ≥ config.STEM_MIN_DICE 就纯看根系 Dice；达不到就按缺口
            # 打折、排到后面去。不用 min()：min 只在茎**低于**根系时才保护，而茎在 root
            # 峰值附近常见的取值是 0.4~0.6 —— 高于根系却远没练好，min 会放它过去
            # （实测：root 0.4075 / stem 0.4902 被选中，测试总长误差翻倍）。
            # 门槛常数的取值依据见 config.STEM_MIN_DICE。
            # 茎在验证集里没标注时 per_ch_dice[1] == -1，退回单看根系。
            if task_ch is not None:
                # 单通道任务：直接按当前通道 Dice 选模型，不再套茎门槛
                score_now = val_dice
                stem_dice = val_dice if task_ch == 1 else -1.0
            else:
                stem_dice = per_ch_dice[1] if N_CH > 1 else -1.0
                if stem_dice >= 0.0:
                    score_now = val_dice * min(1.0, stem_dice / config.STEM_MIN_DICE)
                else:
                    score_now = val_dice
            # 滑动平均：窗口 >1 才起作用，默认 1（试过 10，没有证据支持有用，
            # 见 config.SELECT_SMOOTH 里记的教训）
            score_hist.append(score_now)
            select_dice = sum(score_hist) / len(score_hist)
            # 窗口没攒满先不比：否则头几轮的「平均」只是两三个数，会假性判优、存下废权重
            ready = len(score_hist) == score_hist.maxlen

            improved = ready and select_dice - best_select > 1e-4
            if improved:
                best_select, best_root, best_stem = select_dice, val_dice, stem_dice
                best_epoch = epoch
                # **raw_model**：DDP 包装后 model.state_dict() 的键多一层 "module." 前缀，
                # 存出来的 ckpt 用 ckpt.load_unet 加载会报键不匹配。
                # 各 rank 的 select_dice 经过 all_reduce 后完全一致，所以"哪一轮最好"
                # 所有卡都同意 —— 只在 rank0 写盘就行。
                if ddp.is_main:
                    torch.save({"state_dict": raw_model.state_dict(), "epoch": epoch,
                                "val_dice": val_dice, "select_dice": select_dice,
                                "stem_dice": stem_dice, "hparams": hparams,
                                "out_ch": model_out_ch, "norm": args.norm,
                                "arch": args.arch, "task": task,
                                "class_names": ([task_name] if task_ch is not None
                                                else list(config.CLASS_NAMES))},
                               ckpt_path)
                bad_epochs = 0
            else:
                bad_epochs += 1

            dt = time.time() - t_ep
            note = ""
            if val_ds:
                # 无验证集时指标恒为 -1，不能喂给调度器（会把 LR 一路降到下限）。
                # 喂联合指标而不是 val_dice：三者（保存 / 早停 / 降 LR）对「模型好不好」
                # 必须是同一个定义，否则会出现「早停判据说在变好、调度器判据说没变」。
                scheduler.step(select_dice)
                new_lr = optimizer.param_groups[0]["lr"]
                if new_lr < cur_lr:
                    note = (f"\n[学习率] 联合 Dice 连续 {args.lr_patience} 轮未提升："
                            f"{cur_lr:.2e} → {new_lr:.2e}")
                    cur_lr = new_lr
            if task_ch is not None:
                extra = f" {task_name}_dice={per_ch_dice[0]:.4f}"
            else:
                extra = "".join(f" {n}_dice={d:.4f}"
                                for n, d in zip(config.CLASS_NAMES, per_ch_dice))
            cd_str = ""
            if cld_chans:
                n_all = cd_valid + cd_degen
                cd_str = (f" cldice={1.0 - cd_sum / max(cd_valid, 1):.4f}"
                          if cd_valid else " cldice=nan")
                # 骨架退化 = iters 太小，此时 clDice 恒为 0 损失、梯度静默消失
                if n_all and cd_degen / n_all > 0.5 and epoch <= 2:
                    cd_str += "  [警告] 真值骨架大量为空：CLDICE_ITERS=" \
                              f"{config.CLDICE_ITERS} 太小（需 ≥ 根在模型分辨率下的最大半径），" \
                              "clDice 实际没起作用，请调大或把 LOSS_CLDICE_W 置 0"
            log(f"[Epoch {epoch:03d}/{args.epochs}] loss={train_loss:.4f} "
                f"val_dice={val_dice:.4f} val_iou={val_iou:.4f} time={dt:.1f}s"
                f" lr={cur_lr:.2e}" + cd_str
                + extra
                + (f" select_dice={select_dice:.4f}" if val_ds else "")
                + (" *best*" if improved else "") + note)

            if bad_epochs >= args.patience and epoch >= 10:
                metric_name = task_name if task_ch is not None else "联合"
                log(f"[提前停止] 连续 {args.patience} 轮{metric_name} Dice 未提升，"
                    f"停止训练。")
                break
    except KeyboardInterrupt:
        log("[中断] 收到 Ctrl-C，保存已训练到当前轮的模型权重。")
        if ddp.is_main:
            torch.save({"state_dict": raw_model.state_dict(), "epoch": epoch,
                        "val_dice": val_dice,
                        "select_dice": select_dice,
                        "stem_dice": (per_ch_dice[0] if task_ch == 1 else -1.0)
                        if task_ch is not None
                        else (per_ch_dice[1] if N_CH > 1 else -1.0),
                        "hparams": hparams,
                        "out_ch": model_out_ch, "norm": args.norm,
                        "arch": args.arch, "task": task,
                        "class_names": ([task_name] if task_ch is not None
                                        else list(config.CLASS_NAMES))},
                       ckpt_path)
    finally:
        # 进程组一定要销毁：不然 DDP 的 NCCL 通信线程会吊着，进程退不干净
        ddp.close()

    total = time.time() - t_start
    if best_epoch > 0 and task_ch is not None:
        log(f"[完成] 最佳轮次: epoch {best_epoch} | "
            f"{task_name} Dice {best_select:.4f} | 模型已保存: {ckpt_path}")
    elif best_epoch > 0:
        log(f"[完成] 最佳轮次: epoch {best_epoch} | 联合 Dice {best_select:.4f}"
            f"（根系 {best_root:.4f} / 茎 {best_stem:.4f}） | "
            f"模型已保存: {ckpt_path}")
    else:
        log(f"[完成] 模型已保存: {ckpt_path}")
    log(f"[完成] 总训练用时 {total / 60:.1f} 分钟 | 日志: {log_path}")
    if ddp.is_main:
        print(f"\n模型目录: {folder}\n日志文件: {log_path}")


def _pairs(data_dir):
    from common.dataset import discover_pairs
    return discover_pairs(data_dir)


if __name__ == "__main__":
    main()
