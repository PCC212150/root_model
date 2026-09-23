"""模型权重的统一加载：从 state_dict 识别输出通道数，构造对应的 U-Net。

原来 inference.py / test.py / tool/tune_stats / experimental_report 各自写了一遍
`UNet(in_ch=3, out_ch=1)` + `load_state_dict`，改通道数时四处都要动，容易漏。这里统一。

通道数从 `state_dict["out.weight"]` 的形状读取（不依赖 hparams，旧权重也认得）。
"""
from pathlib import Path

import torch

import config
from common.unet import UNet

# 当前流程要求的输出通道数（见 config.CLASS_NAMES：root / stem / check）
REQUIRED_CHANNELS = 3


def load_unet(pth, device="cpu", require: int = REQUIRED_CHANNELS):
    """加载权重并构造 U-Net，返回 (model, meta)。

    require 不为 None 时校验通道数：旧版的单通道（只有根系）权重会明确报错退出，
    而不是「静默降级成只输出根系」——后者会让茎/检查范围的统计悄悄失效。
    需要跑旧权重时传 require=None，并自行忽略多出来的通道。
    """
    pth = Path(pth)
    # weights_only=False：ckpt 里除权重外还带 hparams（可能含 numpy 标量等非张量对象），
    # torch>=2.6 默认的 weights_only=True 会拒绝加载。这里的文件都是本项目自己训练产出的。
    ckpt = torch.load(pth, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise SystemExit(f"[错误] {pth.name} 不是本项目 train.py 保存的权重"
                         f"（缺少 state_dict 字段）")
    sd = ckpt["state_dict"]
    if "out.weight" not in sd:
        raise SystemExit(f"[错误] {pth.name} 的 state_dict 里没有 out.weight，"
                         f"无法判断输出通道数")
    out_ch = int(sd["out.weight"].shape[0])
    if require is not None and out_ch != require:
        raise SystemExit(
            f"[错误] 权重 {pth.name} 是 {out_ch} 通道的旧模型，当前流程需要 "
            f"{require} 通道（根系 / 茎横截面 / 检查范围）。请用改造后重新训练得到的权重。")

    # 归一化类型必须与训练时一致（BatchNorm 与 GroupNorm 的参数形状不同，装错会加载失败）
    norm = ckpt.get("norm") or (ckpt.get("hparams") or {}).get("norm") or "batch"
    model = UNet(in_ch=3, out_ch=out_ch, norm=norm)
    model.load_state_dict(sd)
    model.to(device)
    hp = ckpt.get("hparams") or {}
    meta = {
        "out_ch": out_ch,
        "norm": norm,
        "epoch": ckpt.get("epoch"),
        "val_dice": ckpt.get("val_dice"),
        "class_names": ckpt.get("class_names"),
        "hparams": hp,
        # 训练时的输入长边：推理/测试要按它来，尺度不一致会明显掉精度（实测 1024 训的模型
        # 用 2048 推理，总长误差从 4278px 涨到 9675px）
        "size": hp.get("size"),
        # 切片训练的块边长；>0 表示这个模型是**在原始分辨率的块上**训的。这时 `size`
        # 的含义变成「块边长」而不是「推理该用的长边」，两者不能混 —— 见下面
        # require_explicit_size。
        "crop": hp.get("crop") or 0,
    }
    return model, meta


def infer_tile(metas, size_arg=None) -> int:
    """推理时的**滑窗块边长**；返回 0 = 走「整图缩放」的老路径。

    切片训练的模型**必须在原始分辨率上滑窗推理**，不能把整图缩到 --size 再跑。
    实测（2026-09-23，model_202609230759，8 张测试图）两种做法的差别是决定性的：

        通道      整图缩到 2048     原始分辨率滑窗
        root      Dice 0.402        Dice 0.401     （平手，但预测面积从超检 2.5 倍回到接近真值）
        stem      两张图**整块塌成 0**  0.763 / 0.775
        check     超检到 92~99%      超检减轻

    茎塌掉的后果不只是那一个通道难看 —— **起点锚定依赖茎**，茎没了锚定就是 0 条
    （用户实测日志里 `起点已锚定到茎 0/59 条`），总长随之系统性偏短。

    块边长默认取训练时的 crop（尺度天然对齐）。传了 --size 就用它 —— 对切片模型来说
    `--size` 的含义就是块边长（与训练时 `--crop` 同义），调大只增加上下文，尺度不变。
    **不设上限校验**：块比图大时 common/predict.tiled_probs 会自动夹到短边。
    """
    crops = [m.get("crop") or 0 for m in metas]
    if not any(crops):
        return 0
    return int(size_arg) if size_arg else max(crops)


def resolve_model_dir(model_arg, root=None) -> Path:
    """把 --model 的单个值解析成模型文件夹。

    原来 inference.py 与 test.py 各写了一份，这里统一（改一处就够，不会两边不一致）。
    model_arg 可以是文件夹名（可省 model_ 前缀）或绝对路径；留空 = 取 root 下最新的。
    """
    root = Path(root) if root else config.MODEL_DIR
    if model_arg:
        cand = Path(model_arg)
        if not cand.is_absolute():
            cand = root / model_arg
            if not cand.exists():
                cand = root / f"model_{model_arg}"
        if not cand.exists():
            avail = sorted(p.name for p in root.glob("model_*") if p.is_dir())
            raise SystemExit(f"[错误] 找不到模型 {model_arg}。可用模型: {avail}")
        return cand
    dirs = [p for p in root.glob("model_*") if p.is_dir()]
    if not dirs:
        raise SystemExit(f"[错误] {root} 下没有模型，请先运行 train/train.py")
    # 按文件夹名取最新（model_YYYYMMDDHHMM 有序）；mtime 会被写进目录的测试结果文件改掉
    return max(dirs, key=lambda p: p.name)


def resolve_pths(model_arg, root=None):
    """把 --model 解析成一组权重路径，支持集成。返回 (pth 列表, 名称列表)。

    **集成写法**：`--model model_a,model_b,model_c`（逗号分隔）。留空 = 取最新那一个。
    """
    root = Path(root) if root else config.MODEL_DIR
    specs = ([s.strip() for s in str(model_arg).split(",") if s.strip()]
             if model_arg else [None])
    pths, names = [], []
    for spec in specs:
        d = resolve_model_dir(spec, root)
        p = d / f"{d.name}.pth"
        if not p.exists():
            cands = sorted(d.glob("*.pth"))
            if not cands:
                raise SystemExit(f"[错误] 模型文件夹中没有 .pth 权重: {d}")
            p = cands[-1]
        pths.append(p)
        names.append(d.name)
    return pths, names


def load_models(pths, device="cpu", require: int = REQUIRED_CHANNELS):
    """加载一个或多个权重（集成用），返回 (models, metas)。

    集成要求所有模型训练时的输入长边一致：概率图必须在同一个网格上平均。不一致会直接
    报错，而不是静默按第一个模型的尺寸跑 —— 那会让其余模型在错误的尺度上推理，结果看似
    正常实则全错（尺度必须与训练一致，实测 1024 训的模型用 2048 推理，总长误差翻倍）。
    """
    models, metas = [], []
    for p in pths:
        m, meta = load_unet(p, device, require=require)
        models.append(m)
        metas.append(meta)
    sizes = sorted({meta.get("size") for meta in metas if meta.get("size")})
    if len(sizes) > 1:
        raise SystemExit(f"[错误] 集成要求所有模型用同一个输入长边，当前是 {sizes}。"
                         f"请挑选训练 --size 相同的模型，或分开跑。")
    crops = sorted({m.get("crop") or 0 for m in metas})
    if len(crops) > 1:
        raise SystemExit(f"[错误] 不能把切片训练与整图缩放的模型混在一起集成"
                         f"（--crop 分别是 {crops}）：两者的尺度语义不同，概率图对不上。")
    return models, metas
