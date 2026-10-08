"""挑 GPU：自动选**当前最空的那张**，不再无脑用 cuda:0。

由来（2026-10-08）：服务器是 4 张 3090，其中一张常被别的任务占着（实测 1 号卡被一个
Java 进程吃了 22.5G、只剩 2G）。而项目里每个入口都写的是
`torch.device("cuda")` —— 那等价于 **cuda:0**：写死 0 号卡的话，撞上别人占着就是 OOM，
或者把别人正在跑的任务挤掉。

优先级（从高到低）：

    1. `--gpu N` 或环境变量 `ROOT_MODEL_GPU=N` —— 显式指定，永远优先
    2. 自动：在**可见**的卡里挑「当前空闲显存最多」的那张
    3. 都没有足够显存（都低于 need_gb）→ 仍然挑最空的那张，但**大声告警**
       （让 OOM 自己报出来，比默默换卡好查）
    4. 没有 CUDA / 传了 cpu=True → CPU

**和 CUDA_VISIBLE_DEVICES 的关系**：那个环境变量是在**进程启动前**限定哪些卡可见，
本模块只在这之后挑 —— 两者不冲突，可以叠加（先 `CUDA_VISIBLE_DEVICES=2,3`，
再让本模块在 2/3 里挑最空的）。config.py 里那条「CUDA_VISIBLE_DEVICES 不会让它用上
第二张卡」说的是**没有本模块之前**的情况。
"""
import os

# 空闲显存低于这个值就不算「能用」（本项目 1024/batch2 实测占 4.4GB；1536 要更多）
MIN_FREE_GB = 4.0


def list_gpus() -> list:
    """[(index, name, free_gb, total_gb, util_pct or None), ...]。没 CUDA 时返回 []。"""
    import torch
    if not torch.cuda.is_available():
        return []
    out = []
    for i in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(i)
        except Exception:
            continue
        try:
            util = int(torch.cuda.utilization(i))
        except Exception:                      # NVML 不可用就只报显存
            util = None
        out.append((i, torch.cuda.get_device_name(i), free / 2**30, total / 2**30, util))
    return out


def describe(devs=None) -> str:
    """把每张卡的占用情况拼成一段文本（写日志用，事后能知道当时跑的哪张）。"""
    devs = list_gpus() if devs is None else devs
    if not devs:
        return "（没有可见的 CUDA 设备）"
    parts = []
    for i, name, free, total, util in devs:
        s = f"cuda:{i} {name} 空闲 {free:.1f}/{total:.1f}GB"
        if util is not None:
            s += f" 占用率 {util}%"
        parts.append(s)
    return "；".join(parts)


def pick(cpu: bool = False, gpu=None, need_gb: float = None, verbose: bool = True):
    """返回 torch.device。gpu 为整数 = 显式指定那张；None = 自动挑最空的。"""
    import torch
    if cpu:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        if verbose:
            print("[提示] 没有可用的 CUDA，改用 CPU")
        return torch.device("cpu")

    devs = list_gpus()
    if not devs:
        return torch.device("cpu")
    if verbose:
        print("可见 GPU：" + describe(devs))

    # ---- 1. 显式指定（命令行 --gpu 优先于环境变量）----
    want = gpu
    if want is None and os.environ.get("ROOT_MODEL_GPU"):
        want = os.environ["ROOT_MODEL_GPU"]
    if want is not None:
        try:
            idx = int(want)
        except (TypeError, ValueError):
            raise SystemExit(f"[错误] GPU 编号必须是整数，收到 {want!r}"
                             f"（用 --gpu N 或环境变量 ROOT_MODEL_GPU）")
        hit = [d for d in devs if d[0] == idx]
        if not hit:
            raise SystemExit(f"[错误] 指定了 cuda:{idx}，但当前可见的只有 "
                             f"{[d[0] for d in devs]}（可用的几张：{describe(devs)}）；"
                             f"想限定可见范围请用 CUDA_VISIBLE_DEVICES")
        if verbose:
            print(f"[选卡] 按指定用 cuda:{idx}（空闲 {hit[0][2]:.1f} GB）")
        return torch.device(f"cuda:{idx}")

    # ---- 2. 自动挑空闲显存最多的 ----
    best = max(devs, key=lambda d: d[2])
    if verbose:
        need = MIN_FREE_GB if need_gb is None else need_gb
        print(f"[选卡] 自动选 cuda:{best[0]}（空闲 {best[2]:.1f} GB，"
              f"{len(devs)} 张卡里最空的一张）")
        if best[2] < need:
            busy = [d for d in devs if d[2] < need]
            print(f"[警告] 最空的这张也只有 {best[2]:.1f} GB，低于需要的 {need:.1f} GB"
                  + (f"；另有 {len(busy)} 张卡空闲不足" if busy else "")
                  + "。跑下去可能会 OOM —— 等别的任务腾出来，或用 --gpu 指定别的卡。")
        if best[4] is not None and best[4] > 80:
            print(f"[警告] cuda:{best[0]} 的占用率已经 {best[4]}%：显存够但算力在被人抢，"
                  f"训练会变慢。想换一张用 --gpu N。")
    return torch.device(f"cuda:{best[0]}")
