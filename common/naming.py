"""命名辅助：时间戳名称 + 名称冲突去重（追加 -1、-2…）+ 植株识别（遵循项目规范）。

**本模块不依赖 torch**（只用标准库）—— 划分/装配这类纯文件名操作的工具要能在
没装 torch 的机器上跑（本地 Windows 就没装），所以别往这儿加 numpy/torch 依赖。
"""
from datetime import datetime
from pathlib import Path


def timestamp(fmt: str = "%Y%m%d%H%M") -> str:
    """当前时间字符串，如 202609091135（年月日时分）。"""
    return datetime.now().strftime(fmt)


def plant_key(name: str) -> str:
    """图片名 -> 植株标识，用于「整株进出」的数据划分。

    'plant_ S062-1_20251116ST' -> 'S062-1'
    'root_C001-1_20241229CK'   -> 'C001-1'

    **2026-10-15 从 common/dataset.py 搬来这里**：逻辑一行没改，只是 dataset.py 顶层
    import torch，于是「按文件名判植株」这件纯字符串的事被绑上了 torch —— 本地没装
    torch 的装配/划分工具就调不了它，只能各自抄一份正则（那正是两边规则漂移的老路）。
    dataset.py 现在从这里 import 再导出，`from common.dataset import plant_key` 照常可用。
    """
    s = str(name).replace(" ", "")  #去掉空格
    for pre in ("plant_", "root_"):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    head, sep, tail = s.rpartition("_")
    # 末段是「日期+批次」这类记号时去掉（如 20251116ST / 20241229CK）
    if sep and len(tail) >= 6 and any(ch.isdigit() for ch in tail):
        s = head
    return s


def unique_path(path) -> Path:
    """若 path 已存在，则在末尾追加 -1、-2 … 后返回（文件/文件夹均适用）。

    例：model_202609091135 已存在 -> model_202609091135-1
        result.txt 已存在       -> result-1.txt
    """
    path = Path(path)
    if not path.exists():
        return path
    k = 1
    while True:
        cand = path.with_name(f"{path.stem}-{k}{path.suffix}")
        if not cand.exists():
            return cand
        k += 1


def create_unique_dir(parent, name) -> Path:
    """并发安全地创建唯一目录（重名追加 -1、-2 …），返回**实际创建**的路径。

    与 unique_path 的区别在并发：unique_path 是「先查存在、再由调用方 mkdir」，两个训练
    同时启动时会双双查到「不存在」拿到同一个名字，其中一个 mkdir(exist_ok=False) 抛
    FileExistsError 直接崩。这里把「判断」和「创建」并成一次 mkdir —— mkdir 本身是原子的，
    撞名就换下一个候选重试。**一张卡跑 1536、另一张跑 1024 同时启动时必须用它。**
    """
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    for k in range(1000):
        cand = parent / (name if k == 0 else f"{name}-{k}")
        try:
            cand.mkdir(exist_ok=False)
            return cand
        except FileExistsError:
            continue
    raise SystemExit(f"[错误] {parent} 下 {name} 的重名副本已超过 1000 个，请先清理")


def create_unique_file(parent, name) -> Path:
    """并发安全地创建一个空文件（重名追加 -1、-2 …），返回**实际创建**的路径。

    与 create_unique_dir 同理：unique_path 是「先查存在、再 open(w)」，两个进程同时
    启动会双双选中同一个文件名，一个把另一个的结果**静默覆盖**掉（比崩溃更糟）。
    open(mode="x") 是原子的，撞名就换下一个候选重试。

    **调用方拿到路径后自己写内容**（本函数只负责占位）。
    """
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    for k in range(1000):
        cand = parent / (name if k == 0 else f"{Path(name).stem}-{k}{Path(name).suffix}")
        try:
            with open(cand, "x", encoding="utf-8"):
                return cand
        except FileExistsError:
            continue
    raise SystemExit(f"[错误] {parent} 下 {name} 的重名副本已超过 1000 个，请先清理")


def model_folder_name(ts: str) -> str:
    """模型文件夹名，如 model_202609091135。"""
    return f"model_{ts}"
