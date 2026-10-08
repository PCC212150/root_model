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


class Dist:
    """DDP 上下文：**没走 torchrun 时 world=1、rank=0，所有分支退化成单卡单进程**。

    于是 train.py 里只有一条训练循环，单卡和多卡共用 —— 不需要维护两份代码。

    单卡：`python train/train.py --batch 8`
    多卡：`torchrun --nproc_per_node=4 train/train.py --batch 2`
          （--batch 是**每张卡**的；4 卡 × 2 = 等效 batch 8）

    判据是 `LOCAL_RANK` 这个环境变量：torchrun 一定会设它，手工 `python xxx.py` 一定没有。
    不用 RANK/WORLD_SIZE —— 那两个变量用户自己也常设，会误判。
    """

    def __init__(self, cpu: bool = False):
        import os
        self.enabled = "LOCAL_RANK" in os.environ
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = int(os.environ.get("RANK", 0))
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        self._cpu = cpu
        self.backend = None

    # ---- 初始化 / 收尾 ----
    def init(self, verbose=True, need_gb: float = None):
        """建进程组。返回本 rank 该用的 torch.device。"""
        import torch
        import torch.distributed as dist
        if not self.enabled:
            return None
        if not dist.is_available():
            raise SystemExit("[错误] 这个 torch 没有编译分布式支持，跑不了 DDP")
        cuda = torch.cuda.is_available() and not self._cpu
        if cuda and dist.is_nccl_available():
            self.backend = "nccl"
        elif cuda:
            # Windows 上 torch 没有编译 NCCL（NCCL 是 Linux 专有的），只剩 gloo；
            # 而 gloo **不支持 CUDA 张量**的集合通信 —— 硬跑会在第一次 all-reduce 时
            # 报一句很难懂的 "No backend type associated with device type cuda"。
            # 与其让人对着那句报错查半天，不如在这里说清楚。
            raise SystemExit(
                "[错误] 这个平台没有 NCCL（Windows 就是），多卡 DDP 跑不了 GPU。\n"
                "       正式的多卡训练请在 Linux 上跑；"
                "本机想验证逻辑可以加 --cpu（gloo + CPU，慢但流程一样）。")
        else:
            self.backend = "gloo"
        # init_method 默认 env://（torchrun 的标准做法）。留个口子给
        # `file://` —— 有些环境（Windows 上这个 torch 就没编 libuv、TCPStore 建不起来）
        # 用不了 TCP 集合点，file:// 不碰它，本地也能跑 2 个 rank 做验证。
        init_method = os.environ.get("ROOT_MODEL_DDP_INIT", "env://")
        dist.init_process_group(self.backend, init_method=init_method,
                                rank=self.rank, world_size=self.world)
        dev = torch.device(f"cuda:{self.local_rank}" if cuda else "cpu")
        if cuda:
            torch.cuda.set_device(self.local_rank)
        if verbose:
            # **DDP 下没有"自动挑卡"这回事**：每个 rank 绑死一张，挑不了。
            # 而分到哪张取决于 `CUDA_VISIBLE_DEVICES`（不设就是 0,1,2,...）——
            # 正好切到别人占着的卡上时，报一句"这张卡剩多少"比让它 OOM 好查得多。
            mem = ""
            if cuda:
                try:
                    free, total = torch.cuda.mem_get_info(self.local_rank)
                    mem = f" | 显存空闲 {free / 2**30:.1f}/{total / 2**30:.1f} GB"
                    if need_gb is not None and free / 2**30 < need_gb:
                        mem += (f"  ← **低于需要的 {need_gb:.1f} GB**："
                                f"这张卡上多半有别的任务，换个组合重来"
                                f"（用 CUDA_VISIBLE_DEVICES 挑）")
                except Exception:
                    pass
            print(f"[DDP] rank {self.rank}/{self.world}（本机第 {self.local_rank} 张卡）"
                  f" | 后端 {self.backend} | 设备 {dev}{mem}", flush=True)
        # **非 rank0 的 stdout 静音**：项目里到处是 print（数据加载、警告、提示），
        # 4 个进程会各打一份、在控制台上交错成一片。只留 rank0 的。
        # **stderr 不动** —— 别的 rank 崩了，traceback 照样看得见。
        if self.rank != 0:
            import sys as _sys
            _sys.stdout.flush()
            _sys.stdout = open(os.devnull, "w")
        return dev

    def close(self):
        import torch.distributed as dist
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()

    # ---- 集合通信 ----
    def allreduce_sum(self, tensor):
        import torch.distributed as dist
        if self.enabled:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def broadcast_object(self, obj, src=0):
        if not self.enabled:
            return obj
        import torch.distributed as dist
        box = [obj]
        dist.broadcast_object_list(box, src=src)
        return box[0]

    def barrier(self):
        import torch.distributed as dist
        if self.enabled:
            dist.barrier()

    @property
    def is_main(self):
        """只有 rank 0 打印/写文件/存权重 —— 否则 4 张卡会往同一个日志里写四份。"""
        return self.rank == 0


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
