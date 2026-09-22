"""显存探针：实测量一个 (分辨率, batch, 归一化, AMP) 组合的峰值显存，用来外推。

为什么要它：想训更大分辨率时，「够不够显存」不该靠猜。显存由三部分组成 ——
    固定开销(cuDNN workspace / 缓存分配器) + 参数&优化器状态 + **激活(随像素数×batch 线性)**
其中只有激活那项是干净的线性关系，另外两项是常数。所以在本机能跑的小尺寸上量几个点、
把直线拟合出来，就能**可靠地外推到本机跑不下的尺寸**（比如服务器 24G 上的 5472）。

用法（项目根目录，pcc 环境）：
    python tool\\mem_probe\\mem_probe.py --sizes 1024,1536 --batch 1
    python tool\\mem_probe\\mem_probe.py --sizes 512,768,1024 --batch 1,2 --norm group
    python tool\\mem_probe\\mem_probe.py --sizes 512,1024 --batch 1 --no-amp
    python tool\\mem_probe\\mem_probe.py --sizes 1024 --batch 1 --checkpoint   # 加梯度检查点

输出每个组合的峰值显存 + 拟合出的 (固定开销, 每 Mpx·batch 的激活) + 外推表。
**探针只做前向+反向，不含数据加载与增强**，所以是显存的下界（但优化器状态已计入）。
"""
import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common.unet import UNet  # noqa: E402

N_CH = len(config.CLASS_NAMES)
MIB = 1024 ** 2


def parse_args():
    p = argparse.ArgumentParser(description="实测 U-Net 的峰值显存并外推")
    p.add_argument("--sizes", default="1024",
                   help="要实测的**长边**像素，逗号分隔（宽高比按数据集原图 5472x3648）")
    p.add_argument("--orig", default="5472x3648", help="数据集原图尺寸 WxH（决定宽高比）")
    p.add_argument("--batch", default="1", help="batch 大小，逗号分隔")
    p.add_argument("--norm", default="group", choices=("group", "batch"))
    p.add_argument("--base", type=int, default=64, help="U-Net 首层通道数")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--checkpoint", action="store_true",
                   help="用 torch.utils.checkpoint 包住每个 DoubleConv（省激活、慢一些）")
    p.add_argument("--steps", type=int, default=3, help="每个组合跑几步取峰值（默认 3）")
    p.add_argument("--infer", action="store_true",
                   help="只测**推理**（torch.inference_mode，无优化器/无反向）。"
                        "训练峰值和推理峰值差很多 —— 后者没有为反向保存的激活。")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def size_from_long(long_side: int, orig_w: int, orig_h: int, stride: int):
    """与 common.image_io.target_size 同一算法，保证和真实训练用到的尺寸一致。"""
    scale = long_side / float(max(orig_w, orig_h))
    w1 = max(stride, int(round(orig_w * scale / stride)) * stride)
    h1 = max(stride, int(round(orig_h * scale / stride)) * stride)
    return w1, h1


def probe(w1, h1, batch, device, norm, base, amp, ckpt, steps, infer=False):
    """在 (w1,h1)×batch 上跑 steps 步，返回 (峰值显存 MiB, 是否 OOM)。"""
    model = UNet(in_ch=3, out_ch=N_CH, base=base, norm=norm).to(device)
    if ckpt:
        # 把每个 DoubleConv 用检查点包起来：前向不存中间激活，反向时重算。
        # 代价是每个被包的块多一次前向 —— 时间 +30~40%，换来激活大幅下降。
        from torch.utils.checkpoint import checkpoint

        def wrap(mod):
            orig = mod.forward

            def fwd(x):
                return checkpoint(orig, x, use_reentrant=False)
            return fwd

        for m in model.modules():
            if type(m).__name__ == "DoubleConv":
                m.forward = wrap(m)
    opt = None if infer else torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and not infer)

    x = torch.randn(batch, 3, h1, w1, device=device)
    y = (torch.rand(batch, N_CH, h1, w1, device=device) > 0.9).float()
    out = None                    # 让下面的 del 在「第一步就 OOM」时也安全
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    try:
        for _ in range(steps):
            if infer:
                # 推理：无反向 → 不需要为 backward 保存任何激活，峰值低得多。
                with torch.inference_mode(), \
                        torch.autocast(device_type=device.type,
                                       enabled=amp and device.type == "cuda"):
                    out = model(x)
                    # 与真实推理一致：sigmoid 后取二值概率图（common/predict.py 走这条）
                    prob = torch.sigmoid(out)
                del out, prob
                continue
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                out = model(x)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(out, y)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            del out, loss
        peak = (torch.cuda.max_memory_allocated(device) / MIB
                if device.type == "cuda" else float("nan"))
        ok = True
    except Exception as e:                      # noqa: BLE001
        # 新版 torch 的显存不足抛 AcceleratorError 而不是 OutOfMemoryError，
        # 且报错是异步的、栈可能指向无关的算子 —— 所以按消息判定，别按类型。
        if "out of memory" not in str(e).lower():
            raise
        print(f"    [OOM] {type(e).__name__}: {str(e)[:160]}")
        peak, ok = float("nan"), False
    # 重新绑定成 None 就够了（引用一断张量立刻可回收），而且不用管哪个变量有没有被
    # 创建过 —— OOM 可能发生在任何一个之前。别用 `del` 列一串：循环里已经 `del out`
    # 过一次，末尾再 del 会 UnboundLocalError。
    model = opt = x = y = out = None       # noqa: F841
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    return peak, ok


def main():
    args = parse_args()
    ow, oh = (int(v) for v in args.orig.lower().split("x"))
    sizes = [int(v) for v in args.sizes.split(",") if v.strip()]
    batches = [int(v) for v in args.batch.split(",") if v.strip()]
    amp = not args.no_amp

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        total = torch.cuda.get_device_properties(device).total_memory / MIB
        print(f"设备: {name} | 总显存 {total:.0f} MiB")
    else:
        name, total = "CPU", float("nan")
        print("设备: CPU（显存数无意义，只看能否跑通）")
    print(f"配置: norm={args.norm} base={args.base} amp={amp} checkpoint={args.checkpoint}"
          f" | 宽高比来自 {ow}x{oh}\n")

    print(f"{'长边':>6}{'输入尺寸':>13}{'Mpx':>7}{'batch':>6}{'峰值MiB':>10}{'GB':>7}  状态")
    pts = []
    for ls in sizes:
        w1, h1 = size_from_long(ls, ow, oh, config.STRIDE)
        mpx = w1 * h1 / 1e6
        for b in batches:
            peak, ok = probe(w1, h1, b, device, args.norm, args.base, amp,
                             args.checkpoint, args.steps, infer=args.infer)
            print(f"{ls:>6}{w1:>7}x{h1:<5}{mpx:>7.2f}{b:>6}"
                  + (f"{peak:>10.0f}{peak / 1024:>7.2f}   ok"
                     if ok else f"{'—':>10}{'—':>7}   OOM"))
            if ok and device.type == "cuda":
                pts.append((mpx * b, peak / 1024.0))     # (Mpx·batch, GB)

    if len(pts) < 2:
        print("\n有效点不足 2 个，无法拟合。多给几个 --sizes 或降 --batch。")
        return

    # 最小二乘拟合 峰值GB = fixed + k × (Mpx·batch)
    import numpy as np
    px = np.array([p[0] for p in pts])
    gb = np.array([p[1] for p in pts])
    k, fixed = np.polyfit(px, gb, 1)
    pred = fixed + k * px
    print(f"\n=== 拟合（{len(pts)} 个点）===")
    print(f"  固定开销 ≈ {fixed:.2f} GB   （cuDNN workspace + 缓存分配器 + 参数&优化器状态）")
    print(f"  激活      ≈ {k:.3f} GB / (Mpx·batch)")
    print(f"  残差      max {np.abs(pred - gb).max():.3f} GB")
    print("\n  回代验证:")
    for (x, y), pr in zip(pts, pred):
        print(f"    {x:>7.1f} Mpx·batch  实测 {y:>6.2f} GB  拟合 {pr:>6.2f} GB")

    print(f"\n=== 外推（{name}，可用显存约 {total / 1024:.1f} GB）===")
    print(f"{'长边':>7}{'输入尺寸':>14}{'batch':>7}{'预估GB':>9}  是否装得下")
    for ls in [1024, 1536, 2048, 2736, 3648, 4096, 5472]:
        w1, h1 = size_from_long(ls, ow, oh, config.STRIDE)
        for b in (1, 2):
            need = fixed + k * (w1 * h1 / 1e6) * b
            mark = "✅" if need < total / 1024 * 0.92 else "❌"
            print(f"{ls:>7}{w1:>8}x{h1:<6}{b:>7}{need:>9.2f}  {mark}")
    print("\n注：探针不含数据加载与增强，是**下界**；实际留 8~10% 余量。"
          "\n    另一个硬约束：batch=1 必须配 --norm group（BatchNorm 会塌，见 unet.make_norm）。")


if __name__ == "__main__":
    main()
