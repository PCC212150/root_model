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
一致），达不到才按缺口打折。防的是「根已到顶、茎还没练好」的权重：测试管线的起点
锚定依赖茎，茎塌了会把 root Dice 一起拖垮。常数依据见 `config.py`，完整来龙去脉
（含一个"这个数据规模测不出 0.02 以下差异"的教训）见 README 的「选模型判据」一节。

数据划分**按植株整组进出**（同一植株的不同时点不会分处训练/验证两侧），
`--val-size` 是验证集**植株数**。

服务器上运行：
CUDA_VISIBLE_DEVICES=0 python train.py --size  --batch  --workers 
-1 则为cpu
"""
import argparse
import json
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common import naming  # noqa: E402
from common.dataset import RootDataset, plant_key  # noqa: E402
from common.unet import UNet  # noqa: E402

N_CH = len(config.CLASS_NAMES)


def dice_loss(prob, target):
    """prob: sigmoid 后概率 (B,C,H,W), target: 0/1。返回逐样本逐通道 Dice 损失 (B,C)。"""
    eps = 1.0
    inter = (prob * target).sum(dim=(2, 3))
    den = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    return 1.0 - (2.0 * inter + eps) / (den + eps)


def _soft_erode(img):
    """软腐蚀：3x3 十字的最小池化。用 max_pool 实现（-maxpool(-x) = minpool），
    这样梯度能穿过池化——这是软骨架能当损失用的关键。"""
    p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))     # 竖直方向 3 邻域取最小
    p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))     # 水平方向 3 邻域取最小
    return torch.min(p1, p2)


def _soft_dilate(img):
    return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))


def _soft_open(img):
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


def parse_args():
    p = argparse.ArgumentParser(description="训练甘蔗根系 U-Net（三通道多标签）")
    p.add_argument("--size", type=int, default=config.MAX_SIDE, help="输入长边像素")
    p.add_argument("--crop", type=int, default=config.CROP_SIZE,
                   help="**切片训练**的块边长(px，必须是 16 的倍数)。>0 时从原图"
                        "**不缩放**随机裁 --crop × --crop 训练，让模型看到原始分辨率的"
                        "根宽（5472x3648 的图缩到 1024 时，原图 10px 的根只剩 1.9px）。"
                        "此时 --size 只写进权重、不参与预处理。"
                        "⚠️ 推理时**必须显式给 --size**：模型是全卷积的，越接近原始分辨率"
                        "越好，24G 上 --size 3648 约 18GB。默认见 config.CROP_SIZE")
    p.add_argument("--batch", type=int, default=config.BATCH_SIZE)
    p.add_argument("--accum", type=int, default=config.ACCUM,
                   help="梯度累积步数：等效 batch = --batch × --accum。显存不够时用它换等效 batch"
                        "（默认见 config.ACCUM）")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--patience", type=int, default=config.PATIENCE,
                   help="联合 Dice 连续 N 轮无提升则早停（应 > --lr-patience）")
    p.add_argument("--lr-patience", type=int, default=config.LR_PATIENCE,
                   help="联合 Dice 连续 N 轮无提升则 LR 减半（默认见 config.LR_PATIENCE）")
    p.add_argument("--val-size", type=int, default=config.VAL_SIZE,
                   help="验证集植株数（同植株的全部时点整组进同一侧）")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--pos-weight", default=None,
                   help="逐通道 BCE 正样本权重，逗号分隔、顺序同 CLASS_NAMES"
                        "（如 3,10,1；默认见 config.LOSS_POS_WEIGHT）。"
                        "走命令行是为了做损失实验时不用改 config，实验值也会记进 hparams.json")
    p.add_argument("--data-dir", type=Path, default=config.TRAIN_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=config.MODEL_DIR)
    p.add_argument("--norm", default=config.NORM, choices=("group", "batch"),
                   help="归一化层：group=GroupNorm（小 batch 稳定）、batch=BatchNorm"
                        "（需 batch≥2，默认见 config.NORM）")
    p.add_argument("--workers", type=int, default=config.NUM_WORKERS,
                   help="DataLoader 子进程数（0=主进程里同步加载）。服务器上设 4~8 可让"
                        "读图/增强与训练并行，避免 GPU 空等（默认见 config.NUM_WORKERS）")
    p.add_argument("--no-amp", action="store_true", help="关闭混合精度")
    p.add_argument("--cpu", action="store_true", help="强制使用 CPU")
    return p.parse_args()


def split_by_plant(names, val_size, seed):
    """按植株分组划分：返回 (训练名列表, 验证名列表, 验证植株列表)。

    同一植株的所有时点整组进同一侧，避免「同株不同时点」跨训练/验证造成泄漏。
    val_size = 验证集植株数；config.VAL_PLANTS 非空时优先按它钉死。
    """
    groups = {}
    for n in names:
        groups.setdefault(plant_key(n), []).append(n)
    keys = sorted(groups)
    if config.VAL_PLANTS:
        pinned = [k for k in keys if k in set(config.VAL_PLANTS)]
        missing = sorted(set(config.VAL_PLANTS) - set(pinned))
        if missing:
            print(f"[警告] config.VAL_PLANTS 里的植株不在数据集中: {missing}")
        val_plants = pinned
    else:
        rng = np.random.RandomState(seed)
        rng.shuffle(keys)
        val_plants = sorted(keys[:max(val_size, 0)])
    val_set = set(val_plants)
    val_names = sorted(n for k in val_plants for n in groups[k])
    train_names = sorted(n for k, v in groups.items() if k not in val_set for n in v)
    return train_names, val_names, val_plants


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 逐通道正样本权重：命令行优先，否则用 config。长度必须与 CLASS_NAMES 对齐，
    # 否则 pos_weight 会广播到错误的通道上（错得不明显，只是学得不对）。
    pos_weight = config.LOSS_POS_WEIGHT
    if args.pos_weight:
        try:
            pos_weight = tuple(float(v) for v in args.pos_weight.split(","))
        except ValueError:
            sys.exit(f"[错误] --pos-weight 解析失败: {args.pos_weight}（应为 3,10,1 这样的形式）")
        if len(pos_weight) != N_CH:
            sys.exit(f"[错误] --pos-weight 需要 {N_CH} 个数（顺序 {config.CLASS_NAMES}），"
                     f"当前 {len(pos_weight)} 个: {args.pos_weight}")

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available()
                          else "cuda")
    if device.type == "cpu":
        print("[警告] 使用 CPU 训练，速度很慢。pcc 环境支持 CUDA（RTX 5060）。")
    else:
        print(f"GPU: {torch.cuda.get_device_name(0)}  显存: "
              f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")

    # ---- 数据划分：按植株整组进出（保证可复现） ----
    names = [n for n, _, _ in _pairs(args.data_dir)]
    train_names, val_names, val_plants = split_by_plant(names, args.val_size,
                                                        args.seed)
    train_plants = sorted({plant_key(n) for n in train_names})
    print(f"数据: 共 {len(names)} 组 / {len(set(map(plant_key, names)))} 植株 | "
          f"训练 {len(train_names)} 组({len(train_plants)} 植株) | "
          f"验证 {len(val_names)} 组({len(val_plants)} 植株)")
    print(f"验证植株: {', '.join(val_plants) if val_plants else '无(不早停,保存最后轮)'}")
    t0 = time.time()
    train_ds = RootDataset(args.data_dir, names=train_names,
                           max_side=args.size, augment=True, seed=args.seed,
                           crop=args.crop)
    val_ds = RootDataset(args.data_dir, names=val_names,
                         max_side=args.size, augment=False, seed=args.seed,
                         crop=args.crop)
    assert len(train_ds) == len(train_names), "训练集样本数不符（名字对不上？）"
    assert len(val_ds) == len(val_names), "验证集样本数不符（名字对不上？）"
    print(f"数据加载完成，用时 {time.time() - t0:.1f}s")

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        persistent_workers=args.workers > 0)
    if args.workers == 0 and device.type == "cuda":
        print("[提示] num_workers=0：读图与数据增强在主进程里同步做，GPU 会空等。"
              "服务器上可加 --workers 8（本机 Windows 保持 0 即可）。")

    # ---- 模型 ----
    if args.norm == "batch" and args.batch < 2:
        print(f"[警告] 归一化用 BatchNorm 但 batch={args.batch}：batch=1 时读到的统计量"
              f"与推理用的滑动平均对不上，会出现严重欠分割。请用 batch≥2 或把 "
              f"config.NORM 改成 'group'。")
    model = UNet(in_ch=3, out_ch=N_CH, norm=args.norm).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    eff_batch = args.batch * args.accum
    print(f"U-Net 参数量: {n_params / 1e6:.2f}M | "
          + (f"切片训练 块 {args.crop}×{args.crop}（原图不缩放）"
             if args.crop else f"整图缩放到长边 {args.size}")
          + f" | batch {args.batch}" + (f"×累积{args.accum}={eff_batch}"
                                       if args.accum > 1 else "")
          + f" | 输出 {N_CH} 通道 {config.CLASS_NAMES}")
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
                            device=device).view(1, N_CH, 1, 1)
    # clDice 只对权重非零的通道计算（软骨架迭代贵，别浪费在茎/检查范围上）
    cld_chans = [c for c, w in enumerate(config.LOSS_CLDICE_W)
                 if w > 0.0] if config.CLDICE_ITERS > 0 else []

    # ---- 输出目录：model/model_年月日时分，重名追加 -1/-2… ----
    ts = naming.timestamp()
    # create_unique_dir 而非 unique_path：同一分钟内启动两个训练（一张卡一个）时，
    # 「先查后建」的写法会让两个进程拿到同一个名字、其中一个直接崩；这里是原子创建。
    folder = naming.create_unique_dir(args.out_dir, naming.model_folder_name(ts))
    ckpt_path = folder / f"{folder.name}.pth"
    log_path = folder / f"{folder.name}_log.txt"
    hparams = {k: (str(v) if isinstance(v, Path) else v)
               for k, v in vars(args).items()}
    hparams.update({"device": str(device), "gpu": torch.cuda.get_device_name(0)
                    if device.type == "cuda" else "cpu",
                    "val_names": val_names, "train_names": train_names,
                    "val_plants": val_plants, "train_plants": train_plants,
                    "class_names": list(config.CLASS_NAMES), "out_ch": N_CH,
                    "norm": args.norm,
                    "loss_bce_w": list(config.LOSS_BCE_W),
                    "loss_dice_w": list(config.LOSS_DICE_W),
                    "loss_pos_weight": list(pos_weight),
                    "loss_cldice_w": list(config.LOSS_CLDICE_W),
                    "cldice_iters": config.CLDICE_ITERS,
                    "params_M": round(n_params / 1e6, 2),
                    "start": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model_name": folder.name})
    with open(folder / "hparams.json", "w", encoding="utf-8") as f:
        json.dump(hparams, f, ensure_ascii=False, indent=2)

    def log(msg, console=True):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
        if console:
            print(msg)

    log(f"===== 训练开始 {hparams['start']} =====\n"
        f"hparams: {json.dumps(hparams, ensure_ascii=False)}", console=False)
    log(f"[信息] {hparams['gpu']} | 参数量 {n_params/1e6:.2f}M | "
        f"输入 {args.size} | batch {args.batch}"
        + (f"×累积{args.accum}={eff_batch}" if args.accum > 1 else "")
        + f" | lr {args.lr} | "
        f"epochs {args.epochs} | 训练 {len(train_names)} 组 | "
        f"验证 {len(val_names)} 组({len(val_plants)} 植株)")
    log(f"[信息] 通道 {config.CLASS_NAMES} | BCE权重 {config.LOSS_BCE_W} | "
        f"Dice权重 {config.LOSS_DICE_W} | 正样本权重 {pos_weight}")
    log(f"[信息] 学习率 {args.lr} | 平台期 {args.lr_patience} 轮不减半就 ×{config.LR_FACTOR}"
        f"（下限 {config.MIN_LR}）| 早停 {args.patience} 轮")
    log(f"[信息] clDice 拓扑损失: "
        + (f"通道 {cld_chans} 权重 {config.LOSS_CLDICE_W} iters {config.CLDICE_ITERS}"
           if cld_chans else "关闭"))
    log(f"[信息] 模型目录: {folder}")

    # ---- 训练循环 ----
    # best_select = 保存判据（联合指标），best_root/best_stem = 那一轮的逐通道值（只为日志）
    best_select, best_root, best_stem, best_epoch = -1.0, -1.0, -1.0, -1
    bad_epochs = 0
    epoch, val_dice = 0, -1.0
    # 先摆一份默认值：第一轮验证之前就 Ctrl-C 的话，异常分支里要用到它们
    per_ch_dice = [-1.0] * N_CH
    select_dice = -1.0
    score_hist = deque(maxlen=max(config.SELECT_SMOOTH, 1))
    t_start = time.time()
    try:
        for epoch in range(1, args.epochs + 1):
            t_ep = time.time()
            model.train()
            loss_sum, n_batch = 0.0, 0
            cd_sum, cd_valid, cd_degen = 0.0, 0, 0
            optimizer.zero_grad(set_to_none=True)
            n_micro = 0
            for i_batch, (x, y, _, valid) in enumerate(loader):
                x, y, valid = x.to(device), y.to(device), valid.to(device)
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
                    wb = channel_weights(config.LOSS_BCE_W, valid)
                    loss_bce = (bce * wb).sum() / wb.sum().clamp(min=1e-6)
                    wd = channel_weights(config.LOSS_DICE_W, valid)
                    loss_dice = ((dice_loss(prob.float(), y) * wd).sum()
                                 / wd.sum().clamp(min=1e-6))
                    loss = loss_bce + loss_dice
                    # clDice：只算权重非零的通道（软骨架迭代较贵，别浪费在茎/检查范围上）
                    if cld_chans:
                        cd, dg = cldice_loss(prob.float(), y, config.CLDICE_ITERS,
                                             cld_chans)
                        wc = channel_weights(config.LOSS_CLDICE_W, valid)[:, cld_chans]
                        loss = loss + (cd * wc).sum() / wc.sum().clamp(min=1e-6)
                        # 统计只算非退化样本，否则退化的 0 会把均值拉低、掩盖问题
                        ok = ~dg
                        cd_sum += float(cd[ok].sum())
                        cd_valid += int(ok.sum())
                        cd_degen += int(dg.sum())
                # 梯度累积：把 loss 按累积步数缩放，攒够 accum 个 micro-batch 再更新一次
                scaler.scale(loss / args.accum).backward()
                n_micro += 1
                if n_micro >= args.accum or (i_batch + 1) == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    n_micro = 0
                loss_sum += float(loss.item())      # 记录未缩放的损失，便于跨配置比较
                n_batch += 1
            train_loss = loss_sum / max(n_batch, 1)

            # ---- 验证（轮内）：逐通道 Dice/IoU，早停/保存按联合指标（见下面 select_dice）----
            val_dice = val_iou = -1.0
            per_ch_dice = [-1.0] * N_CH
            if val_ds:
                model.eval()
                d_all, i_all = [], []
                with torch.no_grad():
                    for x, y, _, valid in val_loader:
                        x = x.to(device)
                        with torch.autocast(device_type="cuda", enabled=amp):
                            prob = torch.sigmoid(model(x)).float()
                        pb = prob.cpu().numpy() > 0.5
                        gt = y.numpy() > 0.5
                        vd = valid.numpy()
                        for i in range(len(gt)):
                            ds, is_ = [], []
                            for c in range(N_CH):
                                if vd[i, c] < 0.5:      # 该图缺这个通道的标注
                                    ds.append(np.nan)
                                    is_.append(np.nan)
                                    continue
                                tp = (pb[i, c] & gt[i, c]).sum()
                                d = 2.0 * tp / (pb[i, c].sum() + gt[i, c].sum() + 1e-8)
                                iou = tp / (pb[i, c].sum() + gt[i, c].sum() - tp + 1e-8)
                                ds.append(float(d))
                                is_.append(float(iou))
                            d_all.append(ds)
                            i_all.append(is_)
                if d_all:
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
            # 权重，而测试管线的**起点锚定依赖茎预测** —— 茎一塌，根系追踪锚不上起点，
            # root Dice 反而被拖垮，看日志却以为是大分辨率不行。
            #
            # **茎当准入门槛**：茎 ≥ config.STEM_MIN_DICE 就纯看根系 Dice；达不到就按缺口
            # 打折、排到后面去。不用 min()：min 只在茎**低于**根系时才保护，而茎在 root
            # 峰值附近常见的取值是 0.4~0.6 —— 高于根系却远没练好，min 会放它过去
            # （实测：root 0.4075 / stem 0.4902 被选中，测试总长误差翻倍）。
            # 门槛常数的取值依据见 config.STEM_MIN_DICE。
            # 茎在验证集里没标注时 per_ch_dice[1] == -1，退回单看根系。
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
                torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                            "val_dice": val_dice, "select_dice": select_dice,
                            "stem_dice": stem_dice, "hparams": hparams,
                            "out_ch": N_CH, "norm": args.norm,
                            "class_names": list(config.CLASS_NAMES)},
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
                log(f"[提前停止] 连续 {args.patience} 轮联合 Dice 未提升，停止训练。")
                break
    except KeyboardInterrupt:
        log("[中断] 收到 Ctrl-C，保存已训练到当前轮的模型权重。")
        torch.save({"state_dict": model.state_dict(), "epoch": epoch,
                    "val_dice": val_dice,
                    "select_dice": select_dice,
                    "stem_dice": per_ch_dice[1] if N_CH > 1 else -1.0, "hparams": hparams,
                    "out_ch": N_CH, "norm": args.norm,
                    "class_names": list(config.CLASS_NAMES)},
                   ckpt_path)

    total = time.time() - t_start
    if best_epoch > 0:
        log(f"[完成] 最佳轮次: epoch {best_epoch} | 联合 Dice {best_select:.4f}"
            f"（根系 {best_root:.4f} / 茎 {best_stem:.4f}） | "
            f"模型已保存: {ckpt_path}")
    else:
        log(f"[完成] 模型已保存: {ckpt_path}")
    log(f"[完成] 总训练用时 {total / 60:.1f} 分钟 | 日志: {log_path}")
    print(f"\n模型目录: {folder}\n日志文件: {log_path}")


def _pairs(data_dir):
    from common.dataset import discover_pairs
    return discover_pairs(data_dir)


if __name__ == "__main__":
    main()
