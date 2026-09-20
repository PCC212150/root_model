"""【已废弃】预测折线的父子关系推断：把「一堆折线」分成 主根 / 侧根。

2026-09-14 起本项目**不再区分一级/二级根**（只统计根系总数与总长），
inference.py / test.py / tool/tune_stats 都已不再调用本模块。文件保留是因为
experimental_reports/experimental_report_20260911_归档/实验报告.md 里「主根/侧根口径与实测精度」
的结论引用它，删除代码容易、删除结论难。新流程请直接用 common.skeleton_stats 的统计量。

----- 以下为原说明 -----

预测折线的父子关系推断：把「一堆折线」分成 主根 / 侧根。

口径与标注一致（见 readme「标注格式」）：**折线的某个端点落在另一条折线上**（也就是长在
分叉点上）→ 这条折线是侧根，那条是它的父根；两端都自由的折线 = 主根。标注里子根的起点就
精确等于父根折线上的一个控制点，这里是它的预测侧对应物。

为什么放在折线级做后处理，而不是改 skeleton_stats 的分链逻辑：分链/分叉点配对/计数归一
是互相耦合、按 16 张图调过参的（见 config.py 注释），动它回归成本高；折线级几何判据同样
能定位分叉点，而且可以整体退让 —— **判定不出来就一律当主根，不丢根、不改长度**。

实测注意（决定默认容差）：折线是按 ~50px 抽稀的，分叉点常常落在父根两个控制点之间，
弦到真点的距离可达 ~12px，所以 attach_tol 默认取 max(6, 0.25*spacing) 而不是几像素。
"""
from dataclasses import dataclass, field

import numpy as np

# 折线端点与其宿主折线的切向夹角小于这个余弦阈值时，视为「同一条根的续接」而不是分叉
# （实测标注里的侧根与父根夹角中位数 68°，只有 3.4% 小于 25°，所以 25° 基本不会误判真侧根）
PARALLEL_COS = 0.906


@dataclass
class Attachment:
    """一条折线的挂载信息：哪一端、挂在哪条折线的哪个位置。"""
    polyline: int = -1        # 子折线索引
    end: int = 0              # 0=首点端，1=尾点端
    parent: int = -1          # 父折线索引
    seg: int = 0              # 挂载点在父折线第 seg 段（P[seg] -> P[seg+1]）
    t: float = 0.0            # 段内参数 0~1
    point: tuple = (0.0, 0.0)  # 投影点（落在父折线上，导出时把子根首点吸附到这里）
    dist: float = 0.0         # 子根该端到投影点的距离
    angle: float = 0.0        # 与父根切向的夹角(度)，越小越像「同一条根」


@dataclass
class Hierarchy:
    """infer_hierarchy 的结果，与输入折线一一对应。"""
    parent: list = field(default_factory=list)   # 父折线索引，None = 主根
    attach: list = field(default_factory=list)   # Attachment 或 None
    depth: list = field(default_factory=list)    # 1=主根 2=侧根 3=更深
    stats: dict = field(default_factory=dict)    # 自检数字（见 hierarchy_diagnostics）

    def __len__(self) -> int:
        return len(self.parent)

    def is_primary(self, i: int) -> bool:
        return self.parent[i] is None


def _plen(P) -> float:
    P = np.asarray(P, dtype=np.float64)
    if len(P) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


def _end_dir(P, end: int, arc: float = 20.0) -> np.ndarray:
    """端点处「朝折线内部」的单位方向：按 ~arc 像素弧长取弦。

    单段方向在抽稀后可能只有几像素、噪声大，所以按弧长取弦（与标注的抽稀间距同量级）。
    """
    P = np.asarray(P, dtype=np.float64)
    seq = P if end == 0 else P[::-1]
    if len(seq) < 2:
        return np.zeros(2)
    acc, k = 0.0, 1
    for k in range(1, len(seq)):
        acc += float(np.linalg.norm(seq[k] - seq[k - 1]))
        if acc >= arc or k >= len(seq) - 1:
            break
    v = seq[k] - seq[0]
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(2)


def _project(p, Q) -> tuple:
    """点 p 投到折线 Q 上，返回 (距离, 段号, 段内参数 t, 投影点)。

    逐段投影一次性向量化（这个函数在「每个端点 × 每条其它折线」上调用，
    参数扫描时要跑几十万次，Python 逐段循环是瓶颈）。
    """
    p = np.asarray(p, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if len(Q) < 2:
        return float("inf"), 0, 0.0, np.zeros(2)
    A, B = Q[:-1], Q[1:]
    AB = B - A
    t = np.clip(((p - A) * AB).sum(axis=1)
                / np.maximum((AB * AB).sum(axis=1), 1e-9), 0.0, 1.0)
    pts = A + t[:, None] * AB
    d = np.linalg.norm(p - pts, axis=1)
    k = int(d.argmin())
    return float(d[k]), k, float(t[k]), pts[k]


def _tangent(Q, seg: int, t: float, half: float = 25.0) -> np.ndarray:
    """折线 Q 在投影片段位置处的切向（沿弧长向前后各取 ~half 像素的弦）。"""
    Q = np.asarray(Q, dtype=np.float64)
    q = Q[seg] + t * (Q[seg + 1] - Q[seg])

    back, acc, i = q, t * float(np.linalg.norm(Q[seg + 1] - Q[seg])), seg
    while i > 0 and acc < half:
        back = Q[i]
        acc += float(np.linalg.norm(Q[i] - Q[i - 1]))
        i -= 1

    fwd, acc, i = q, (1.0 - t) * float(np.linalg.norm(Q[seg + 1] - Q[seg])), seg + 1
    while i < len(Q) - 1 and acc < half:
        acc += float(np.linalg.norm(Q[i + 1] - Q[i]))
        i += 1
    fwd = Q[i]

    v = fwd - back
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.zeros(2)


def infer_hierarchy(polylines, spacing: float = 50.0, attach_tol: float = None,
                    end_margin: float = None, parallel_cos: float = PARALLEL_COS) -> Hierarchy:
    """推断折线之间的父子关系。

    polylines: [[(x, y), ...], ...]（如 skeleton_stats.extract_root_paths 的输出）。
    spacing:    折线抽稀间距，用来缩放两个容差的默认值（与 controlpointseparation 一致）。
    attach_tol: 端点离父根折线多远算「长在上面」，默认 max(6, 0.25*spacing)。
    end_margin: 不许挂在父根自己两端 end_margin 以内（否则会把断口当成侧根），默认 0.5*spacing。
    parallel_cos: 与父根切向夹角余弦超过它就认为「同一条根」（续接），不作为侧根。

    返回 Hierarchy；判定不出来的一律 parent=None（当主根），保证不丢根、总长守恒。
    """
    n = len(polylines)
    if attach_tol is None:
        attach_tol = max(6.0, 0.25 * spacing)
    if end_margin is None:
        end_margin = 0.5 * spacing

    lengths = [_plen(P) for P in polylines]
    ends = []          # [(i, end, 点, 朝内方向)]
    for i, P in enumerate(polylines):
        if len(P) < 2:
            continue
        P = np.asarray(P, dtype=np.float64)
        for e in (0, 1):
            ends.append((i, e, P[0] if e == 0 else P[-1], _end_dir(P, e)))

    # ---- 每个端点找宿主：距离 <= attach_tol，且不落在宿主自己的端头附近 ----
    cand = {}          # (i, end) -> Attachment
    diag = {"ambiguous": 0, "rejected_by_end_margin": 0, "rejected_by_parallel": 0,
            "rejected_by_length": 0}
    for (i, e, pt, d) in ends:
        best = None
        for j, Q in enumerate(polylines):
            if j == i or len(Q) < 2:
                continue
            if lengths[j] < 2 * end_margin:      # 太短的折线不能当父根（长度守恒用）
                diag["rejected_by_length"] += 1
                continue
            dist, seg, t, q = _project(pt, Q)
            if dist > attach_tol:
                continue
            # 投影点沿父根的弧长，太靠端头就当没挂（避免把「断口」认成侧根）
            acc = t * float(np.linalg.norm(np.asarray(Q[seg + 1]) - np.asarray(Q[seg])))
            for k in range(seg):
                acc += float(np.linalg.norm(np.asarray(Q[k + 1]) - np.asarray(Q[k])))
            if acc < end_margin or acc > lengths[j] - end_margin:
                diag["rejected_by_end_margin"] += 1
                continue
            tp = _tangent(Q, seg, t)
            cos = abs(float(np.dot(tp, d)))
            angle = float(np.degrees(np.arccos(np.clip(cos, 0.0, 1.0))))
            if cos >= parallel_cos:              # 与父根共线 = 同一条根的续接，不是侧根
                diag["rejected_by_parallel"] += 1
                continue
            better = (best is None or dist < best.dist - 3.0
                      or (abs(dist - best.dist) <= 3.0 and lengths[j] > lengths[best.parent]))
            if better:
                best = Attachment(polyline=i, end=e, parent=j, seg=seg, t=t,
                                  point=(float(q[0]), float(q[1])), dist=dist, angle=angle)
        if best is not None:
            # 同一端点有多个近邻候选时记一笔，便于调容差时看歧义程度
            near = [j for j, Q in enumerate(polylines)
                    if j != i and len(Q) >= 2 and _project(pt, Q)[0] <= attach_tol]
            if len(near) > 1:
                diag["ambiguous"] += 1
            cand[(i, e)] = best

    # ---- 组织成父子链：首点端优先（与标注「子根首点=分叉点」一致），成环则断开 ----
    parent = [None] * n
    attach = [None] * n
    cycles_broken = 0
    for i in range(n):
        for e in (0, 1):
            a = cand.get((i, e))
            if a is not None and attach[i] is None:
                attach[i], parent[i] = a, a.parent
    for i in range(n):                            # 断环：parent 指针不能成环
        seen, cur = {i}, parent[i]
        while cur is not None:
            if cur in seen:
                attach[i], parent[i] = None, None
                cycles_broken += 1
                break
            seen.add(cur)
            cur = parent[cur]

    # ---- 深度：从主根 BFS ----
    depth = [0] * n
    children = {i: [] for i in range(n)}
    for i in range(n):
        if parent[i] is not None:
            children[parent[i]].append(i)
    queue = [i for i in range(n) if parent[i] is None]
    for i in queue:
        depth[i] = 1
    head = 0
    while head < len(queue):
        u = queue[head]
        head += 1
        for v in children[u]:
            depth[v] = depth[u] + 1
            queue.append(v)

    stats = {
        "primary": sum(1 for i in range(n) if parent[i] is None),
        "secondary": sum(1 for i in range(n) if parent[i] is not None),
        "depth_max": max(depth) if depth else 0,
        "cycles_broken": cycles_broken,
        **diag,
    }
    return Hierarchy(parent=parent, attach=attach, depth=depth, stats=stats)


def hierarchy_summary(polylines, hier: Hierarchy) -> dict:
    """把分层结果折算成统计量：主根/侧根的条数与长度（长度守恒，便于断言）。"""
    lengths = [_plen(P) for P in polylines]
    prim = [i for i in range(len(polylines)) if hier.parent[i] is None]
    sec = [i for i in range(len(polylines)) if hier.parent[i] is not None]
    return {
        "primary_count": len(prim),
        "secondary_count": len(sec),
        "primary_length": float(sum(lengths[i] for i in prim)),
        "secondary_length": float(sum(lengths[i] for i in sec)),
        "total_length": float(sum(lengths)),
    }


def gt_split(roots) -> tuple:
    """标注侧分组：返回 (主根列表, 侧根列表)。

    判据与标注的嵌套口径一致 —— label 不是 primary，或 root_id 层级多于一层（如 "2.1.1"）；
    现有标注里两者完全一致（嵌套的 root 都标 secondary）。rsml_parse 已经把嵌套展开成平表，
    所以这里不用改解析器，只看 label 与 ID 层级即可。
    """
    prim, sec = [], []
    for r in roots:
        if r.label == "primary" and r.root_id.count(".") <= 1:
            prim.append(r)
        else:
            sec.append(r)
    return prim, sec


def gt_summary(roots) -> dict:
    """标注侧的 主根/侧根 条数与长度（与 hierarchy_summary 同键名，便于并排比较）。"""
    prim, sec = gt_split(roots)
    return {
        "primary_count": len(prim),
        "secondary_count": len(sec),
        "primary_length": float(sum(r.length for r in prim)),
        "secondary_length": float(sum(r.length for r in sec)),
        "total_length": float(sum(r.length for r in roots)),
    }
