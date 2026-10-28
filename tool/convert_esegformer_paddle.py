"""把 RPT 的 PaddlePaddle ESegFormer-B2 权重转成 PyTorch state_dict。

RPT 的权重：`Net/RPT/model.pdparams`（Paddle 2.x 保存，2 类 root/background）。
本项目的 PyTorch 模型：`common/esegformer.py` 的 `ESegFormer2D`。

**注意：Windows 上 Paddle 与 PyTorch 的 cuDNN DLL 会冲突，不能在同一进程里同时
import。** 所以脚本分两步：先 `--stage paddle` 用 Paddle 把张量 dump 成 numpy
pickle，再 `--stage torch` 用 PyTorch 转换。`--stage both` 会自动用两个子进程
执行（默认）。

用法（本机 pcc 环境已有 paddle 3.3.0 + torch 2.12）：

    python tool/convert_esegformer_paddle.py \
      --src Net/RPT/model.pdparams \
      --out pretrained_ckpt/esegformer_b2_from_rpt.pth

转换规则：
- Paddle Linear 权重是 [in, out]，PyTorch 是 [out, in] —— 所有 2D 权重转置；
- Conv2d / LayerNorm / CBAM 的 1x1 conv 权重布局与 PyTorch 一致，直接拷贝；
- BatchNorm 的 `_mean` / `_variance` 换成 PyTorch 的 `running_mean` / `running_var`；
- `linear_fuse._conv` / `linear_fuse._batch_norm` 换成
  `decode_head.linear_fuse.conv` / `decode_head.linear_fuse.bn`；
- 预训练是 2 类，转出的 state_dict 也是 2 类；训练 3 通道时
  `load_pretrained_esegformer` 会跳过 `decode_head.linear_pred.*`。
"""
import argparse
import pickle
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def map_key(k: str) -> str:
    if k.startswith(("linear_c", "linear_fuse", "linear_pred")):
        k = "decode_head." + k
    k = k.replace("decode_head.linear_fuse._conv.",
                  "decode_head.linear_fuse.conv.")
    k = k.replace("decode_head.linear_fuse._batch_norm.",
                  "decode_head.linear_fuse.bn.")
    k = k.replace("._mean", ".running_mean").replace(
        "._variance", ".running_var")
    return k


def stage_paddle(args):
    import paddle
    src = Path(args.src)
    if not src.is_file():
        raise SystemExit(f"[错误] 找不到 {src}")
    sd = paddle.load(str(src))
    arrays = {}
    for k, v in sd.items():
        arrays[k] = v.numpy() if hasattr(v, "numpy") else v
    with open(args.arrays, "wb") as f:
        pickle.dump(arrays, f, protocol=4)
    print(f"[Paddle] {src}: {len(arrays)} 个张量 -> {args.arrays}")


def stage_torch(args):
    import numpy as np
    import torch
    from common.esegformer import ESegFormer2D

    with open(args.arrays, "rb") as f:
        paddle_sd = pickle.load(f)
    # out_ch=2 与预训练头一致；训练 3 通道时加载器会跳过分类头
    model = ESegFormer2D(in_ch=3, out_ch=2, variant=args.variant,
                         norm=args.norm, input_norm="none")
    target = model.state_dict()
    new_sd, skipped = {}, []
    for k, arr in paddle_sd.items():
        tk = map_key(k)
        if tk not in target:
            skipped.append((k, "no target"))
            continue
        if arr.ndim == 2 and target[tk].shape == tuple(arr.shape[::-1]):
            arr = arr.T
        if tuple(arr.shape) != tuple(target[tk].shape):
            skipped.append((k, f"shape {arr.shape} != {target[tk].shape}"))
            continue
        new_sd[tk] = torch.from_numpy(np.ascontiguousarray(arr)).float()

    for k, v in target.items():
        if k.endswith("num_batches_tracked") and k not in new_sd:
            new_sd[k] = torch.zeros_like(v)

    missing = [k for k in target if k not in new_sd]
    model.load_state_dict(new_sd, strict=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": new_sd}, out)
    print(f"[Torch] {out}: {len(new_sd)} 个张量 / "
          f"{sum(v.numel() for v in new_sd.values()) / 1e6:.2f}M 参数")
    if skipped:
        print(f"[跳过] {len(skipped)} 个：")
        for k, why in skipped[:10]:
            print(f"  {k}: {why}")
    if missing:
        print(f"[缺失] {len(missing)} 个目标键：")
        for k in missing[:10]:
            print(f"  {k}")
    print("[提示] 预训练头是 2 类；训练 3 通道时会自动跳过 linear_pred，"
          "其它权重照常加载。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="Net/RPT/model.pdparams")
    ap.add_argument("--out", default="pretrained_ckpt/esegformer_b2_from_rpt.pth")
    ap.add_argument("--arrays", default="pretrained_ckpt/esegformer_b2_paddle.pkl")
    ap.add_argument("--variant", default="b2", choices=("b0", "b1", "b2",
                                                        "b3", "b4", "b5"))
    ap.add_argument("--norm", default="batch", choices=("batch", "group"),
                    help="转换时用 batch，才能把 BN 的 running stats 一起存下来；"
                         "训练时用 group 加载器会跳过 running stats")
    ap.add_argument("--stage", default="both", choices=("paddle", "torch", "both"))
    args = ap.parse_args()

    # 子进程里也用绝对路径，避免依赖当前目录
    args.src = str(Path(args.src))
    args.out = str(Path(args.out))
    args.arrays = str(Path(args.arrays))

    if args.stage == "paddle":
        stage_paddle(args)
    elif args.stage == "torch":
        stage_torch(args)
    else:
        base = [sys.executable, str(Path(__file__).resolve())]
        common = ["--src", args.src, "--out", args.out, "--arrays", args.arrays,
                  "--variant", args.variant, "--norm", args.norm]
        subprocess.run(base + ["--stage", "paddle"] + common, check=True)
        subprocess.run(base + ["--stage", "torch"] + common, check=True)


if __name__ == "__main__":
    main()
