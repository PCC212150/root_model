"""预测掩码 -> 骨架化 -> 剪枝 -> 交叉点续接配链，估算 根数量 / 各根长度 / 总长度。

流程：
1. 轻度腐蚀（disk=1，默认 1 次）：消除 5px 画线/预测线的厚度伪影；
2. 二值掩码骨架化（skimage），在原图分辨率上进行；
3. 剪枝：移除长度短于 spur(像素) 的末梢（噪声）；
4. 链收缩：度数=2 的骨架像素折叠成带权链边（直连1、对角 sqrt(2)），
   只保留 叶端(度1)/分叉点(度>=3)；邻近分叉点(<=6px)合并成单节点，
   消除线条交叉处厚度产生的簇状伪分叉；
5. 分叉点续接：在每个分叉点把"方向最连贯"的两条臂配成一对
   （十字交叉的两条根在交点处走向连续，配对即把交叉的根"穿过去"还原整根）；
   不成对的多余臂 = 该根的起点/终止端；
6. 从 叶端/未配对臂 起步沿配对关系串成完整轨迹：每条轨迹 = 1 条根，
   长度 = 路径上各链边长之和（像素欧氏，与 RSML 口径一致）；
   轨迹同时保留像素序列，可按弧长抽稀导出为 RSML 折线（见 extract_root_paths）。

说明：无先验的近似拆分（相切/粘连时走向可能误配），误差在 test.py 汇总对比
中体现；阈值参数化便于调优。

实现上分成两段，便于参数扫描时复用：[掩码 -> 骨架 -> 邻接表]（_skeleton_adj，
与阈值无关，一张图只算一次）与 [邻接表 -> 剪枝 -> 分链]（_strands_from_adj，
每组阈值算一次，内部拷贝邻接表、不改入参）。
"""
import numpy as np
from PIL import Image
from skimage.morphology import disk, erosion, skeletonize

_EPS = 1e-9
# 骨架化前的裁切留边：erosion 在数组边界按 reflect 处理，贴着物体边界裁会把反射出的
# 假前景算进来；留几像素纯背景即可，代价可忽略（见 _skeleton_adj）。
_CROP_MARGIN = 4


def _step_len(a, b) -> float:
    return 1.4142135623730951 if (a[0] != b[0] and a[1] != b[1]) else 1.0


def _unit(dy, dx):
    n = (dy * dy + dx * dx) ** 0.5
    return (dy / n, dx / n) if n > _EPS else (0.0, 1.0)


def _prune(adj, spur_s):
    """从叶端向内剥除长度 < spur_s 的末梢（原地修改 adj）。"""
    pruned = True
    while pruned:
        pruned = False
        for leaf in [p for p in adj if len(adj[p]) == 1]:
            if leaf not in adj:
                continue
            path = [leaf]
            cur, prev, length = leaf, None, 0.0
            while True:
                nbrs = [n for n in adj[cur] if n != prev]
                if not nbrs:
                    break
                nxt = nbrs[0]
                length += adj[cur][nxt]
                if length >= spur_s or len(adj.get(nxt, ())) > 2:
                    break
                path.append(nxt)
                prev, cur = cur, nxt
            if length < spur_s:
                for p in path:
                    if p not in adj:
                        continue
                    for q in adj[p]:
                        if q in adj:
                            adj[q].pop(p, None)
                    del adj[p]
                pruned = True
    for p in [p for p in adj if not adj[p]]:
        del adj[p]


def _components(adj):
    seen, comps = set(), []
    for start in adj:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            p = stack.pop()
            comp.append(p)
            for q in adj[p]:
                if q not in seen:
                    seen.add(q)
                    stack.append(q)
        comps.append(comp)
    return comps


def _merge_junctions(nodes, adj):
    """把邻近(<=6px)的分叉点归并为一个代表点，返回 原节点->代表点 映射。

    只合并分叉点(度>=3)，绝不合并叶端（相邻两条根的端部可能只有几像素远）。
    """
    juncs = [p for p in nodes if len(adj[p]) >= 3]
    parent = {p: p for p in nodes}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, a in enumerate(juncs):
        for b in juncs[i + 1:]:
            if abs(a[0] - b[0]) <= 6 and abs(a[1] - b[1]) <= 6:
                union(a, b)
    return {p: find(p) for p in nodes}


def _strands_of_component(comp_nodes, adj, spur_s, min_len, normalize=True):
    """对一个连通块拆根。

    返回 [[长度(像素), 像素轨迹[(y,x), ...], 端点元信息], ...]；轨迹首点是一个根端。
    元信息与 analyze_mask_ex 的 strand_meta 同构：长度 < 2 的条目会让上层取 t[2] 越界，
    所以每条分支都必须带元信息（纯环分支的元信息两个端点都是 "loop"）。

    normalize=True 时执行末尾的「计数归一」（把过碎的短轨迹按端点最近拼接回现有轨迹）；
    False 时原样返回全部轨迹。侧根只有 1 个自由端，而归一按 ceil(叶端数/2) 限数，
    会把侧根并掉近一半，所以需要评估侧根时必须能关掉它（参数扫描里的一个维度）。
    """
    # ---- 全部为度2节点 -> 纯环，整块 1 条根 ----
    if all(len(adj[p]) == 2 for p in comp_nodes):
        p0 = comp_nodes[0]
        prev, cur = p0, next(iter(adj[p0]))
        total = 0.0
        px = [p0]
        while cur != p0:
            total += adj[prev][cur]
            px.append(cur)
            a, b = tuple(adj[cur])
            nxt = a if a != prev else b
            prev, cur = cur, nxt
        total += adj[prev][cur]
        meta = {"start_kind": "loop", "start_node": p0,
                "end_kind": "loop", "end_node": p0}
        return [[total, px, meta]] if total >= min_len else []

    # ---- 链收缩：节点 = 叶端/分叉点；边 = 带像素序列的链 ----
    node_set = {p for p in comp_nodes if len(adj[p]) != 2}
    # 同一条链若从两端各走一遍会登记成两条，故只从"字典序较小"端点登记
    edges = []  # (a, b, w, d_a, d_b, chain)  chain: a -> b 的骨架像素(含两端)
    for p in node_set:
        for nxt0 in adj[p]:
            prev, cur = p, nxt0
            w = adj[p][nxt0]
            chain = [p, nxt0]
            while len(adj[cur]) == 2:
                a, b = tuple(adj[cur])
                nxt = a if a != prev else b
                w += adj[cur][nxt]
                prev, cur = cur, nxt
                chain.append(cur)
            end = cur
            if end == p or end < p:
                continue  # 自环或由另一端登记
            da = _unit(chain[1][0] - p[0], chain[1][1] - p[1])       # p 端方向
            db = _unit(chain[-2][0] - end[0], chain[-2][1] - end[1])  # end 端方向
            edges.append((p, end, w, da, db, chain))

    # ---- 邻近分叉点合并 ----
    rep_of = _merge_junctions(node_set, adj)
    arms = {}  # 代表节点 -> [{end, w, dout, px}]
    for (a, b, w, da, db, chain) in edges:
        ra, rb = rep_of[a], rep_of[b]
        if ra == rb:
            continue  # 簇内自环，长度极小，忽略
        arms.setdefault(ra, []).append(
            {"end": rb, "w": w, "dout": da, "px": chain})
        arms.setdefault(rb, []).append(
            {"end": ra, "w": w, "dout": db, "px": chain[::-1]})

    # ---- 分叉点续接配对 ----
    # 第一轮：只配"方向连贯"(接近直通)的对，避免把垂直粘连误连；
    # 第二轮：剩余臂两两按最连贯方向补配（剩 0/1 条为止），
    #         保证轨迹能贯通到真正的根端，计数贴近 叶端数/2。
    paired = {}
    for j, jarms in arms.items():
        if len(jarms) < 3:
            continue
        idx = list(range(len(jarms)))
        use = {}

        def best_pair(idx):
            best = None
            for ai, i in enumerate(idx):
                for aj in range(ai + 1, len(idx)):
                    j2 = idx[aj]
                    d, e = jarms[i]["dout"], jarms[j2]["dout"]
                    cost = 1.0 + d[0] * e[0] + d[1] * e[1]
                    if best is None or cost < best[0]:
                        best = (cost, i, j2)
            return best

        while len(idx) >= 2:
            cand = best_pair(idx)
            if cand is None or cand[0] > 0.8:
                break  # 第一轮：非直通不再配
            _, bi, bj = cand
            use[bi], use[bj] = bj, bi
            idx.remove(bi)
            idx.remove(bj)
        while len(idx) >= 2:  # 第二轮：无条件补配到 0/1 条
            _, bi, bj = best_pair(idx)
            use[bi], use[bj] = bj, bi
            idx.remove(bi)
            idx.remove(bj)
        if use:
            paired[j] = use

    # ---- 沿配对串轨迹 ----
    seen_arms = set()
    results = []  # [[长度, 像素轨迹, 端点元信息], ...]

    def _node_kind(node):
        """节点类型：叶端(1 条臂) / 分叉点(其余)。判断侧根要用它。"""
        return "leaf" if len(arms.get(node, ())) == 1 else "junction"

    def traverse(start_node, start_k):
        """从某臂起步串一条轨迹，返回 (长度, 像素轨迹, 终点, 终点类型)；遇到已消费臂返回 None。"""
        length = 0.0
        px = []
        node, k = start_node, start_k
        while True:
            if (node, k) in seen_arms:
                return (length, px, node, _node_kind(node)) if length > 0 else None
            if node not in arms or k >= len(arms[node]):
                return None
            seen_arms.add((node, k))
            arm = arms[node][k]
            seg = arm["px"]
            px.extend(seg if not px else seg[1:])
            length += arm["w"]
            n2 = arm["end"]
            k2 = None
            for i2, a2 in enumerate(arms.get(n2, ())):
                if a2["end"] == node:
                    k2 = i2
                    break
            if k2 is None:
                return length, px, n2, _node_kind(n2)
            juse = paired.get(n2)
            if juse and k2 in juse:
                seen_arms.add((n2, k2))  # 到达侧臂已消费
                node, k = n2, juse[k2]
                continue
            seen_arms.add((n2, k2))
            return length, px, n2, _node_kind(n2)

    starts = []   # [(节点, 臂序号, 起点类型), ...]
    for p, parms in arms.items():
        if len(parms) == 1:
            starts.append((p, 0, "leaf"))  # 叶端
    for p, parms in arms.items():
        if len(parms) >= 3:
            juse = paired.get(p, {})
            for k in range(len(parms)):
                if k not in juse:
                    starts.append((p, k, "junction"))  # 未配对的起点臂
    started_set = {(p, k) for p, k, _ in starts}
    for (p, k, kind) in starts:
        r = traverse(p, k)
        if r is not None and r[0] >= min_len:
            results.append([r[0], r[1], {"start_kind": kind, "start_node": p,
                                         "end_kind": r[3], "end_node": r[2]}])
    # 兜底：剩余未消费臂（环等）也串起来，避免丢长度
    for p, parms in arms.items():
        for k in range(len(parms)):
            if (p, k) not in seen_arms and (p, k) not in started_set:
                r = traverse(p, k)
                if r is not None and r[0] >= min_len:
                    results.append([r[0], r[1], {"start_kind": "loop", "start_node": p,
                                                 "end_kind": r[3], "end_node": r[2]}])

    results.sort(key=lambda t: t[0], reverse=True)
    # 计数归一：本块根数 = ceil(叶端数/2)（每根两端在图上分开、互不粘连时严格成立，
    # GT 统计验证 23 根 -> 23)。轨迹多于该值时，把最短轨迹按"端点最近"原则
    # 拼接回现有轨迹（既保住总长，又让每条折线几何上连续完整）。
    # 副作用：侧根只有 1 个自由端，这一步会把侧根并掉，故须能通过 normalize=False 关掉。
    n_leaves = sum(1 for p, v in arms.items() if len(v) == 1)
    if normalize and n_leaves >= 2:
        k = (n_leaves + 1) // 2
        while len(results) > k:
            piece = results.pop()  # 最短的一条
            ppts = piece[1]
            p_ends = (ppts[0], ppts[-1])
            best = None
            for idx, r in enumerate(results):
                pts = r[1]
                for end_i in (0, 1):
                    for piece_end in (0, 1):
                        e1, e2 = pts[0 if end_i == 0 else -1], p_ends[piece_end]
                        d = (e1[0] - e2[0]) ** 2 + (e1[1] - e2[1]) ** 2
                        if best is None or d < best[0]:
                            best = (d, idx, end_i, piece_end)
            _, idx, end_i, piece_end = best
            L, pts, meta = results[idx]
            pL, ppts, pmeta = piece
            if end_i == 1:  # 接到 pts 尾部：S 起点保留，终点取 piece 没被拼上的那一端
                seg = ppts if piece_end == 0 else ppts[::-1]
                tail = pmeta["end_kind"], pmeta["end_node"]
                if piece_end == 1:
                    tail = pmeta["start_kind"], pmeta["start_node"]
                results[idx] = [L + pL, pts + seg,
                                {"start_kind": meta["start_kind"],
                                 "start_node": meta["start_node"],
                                 "end_kind": tail[0], "end_node": tail[1]}]
            else:           # 接到 pts 头部：S 终点保留，起点取 piece 没被拼上的那一端
                seg = ppts if piece_end == 1 else ppts[::-1]
                head = pmeta["end_kind"], pmeta["end_node"]
                if piece_end == 1:
                    head = pmeta["start_kind"], pmeta["start_node"]
                results[idx] = [L + pL, seg + pts,
                                {"start_kind": head[0], "start_node": head[1],
                                 "end_kind": meta["end_kind"],
                                 "end_node": meta["end_node"]}]
        results.sort(key=lambda t: t[0], reverse=True)
    return results


def stem_anchor_tolerance(stem_mask, factor: float, min_px: float, max_px: float) -> float:
    """锚定阈值(px)：与茎的等效半径成正比（泡沫环的厚度跟茎粗细同量级）。

    等效半径 r = sqrt(面积/π)，阈值取 factor*r，卡在 [min_px, max_px] 之间
    （下限防止小茎时阈值太小锚不上，上限防止茎预测异常时把远处的碎段也拉过来）。
    """
    area = float(np.count_nonzero(stem_mask))
    r_eq = (area / np.pi) ** 0.5 if area > 0 else 0.0
    return float(min(max_px, max(min_px, factor * r_eq)))


def stem_anchor_gain(ends, tree, max_dist: float):
    """一条折线的两个端点里，离茎最近的那一端需要补多长；超过阈值返回 None。

    ends: [(x, y), (x, y)]，原图坐标（与 analyze_mask_ex 的 paths 同口径）。
    返回 (补的长度, 该端点的下标, 茎上最近的落点(x, y))。
    """
    dists = []
    for (x, y) in ends:
        d, idx = tree.query([x, y])
        dists.append((float(d), idx))
    k = 0 if dists[0][0] <= dists[1][0] else 1
    d, idx = dists[k]
    if d > max_dist:
        return None
    sx, sy = tree.data[idx]
    return d, k, (float(sx), float(sy))


def anchor_paths_to_stem(paths, lengths, stem_mask, factor: float = 6.0,
                         min_px: float = 250.0, max_px: float = 600.0,
                         metas=None) -> tuple:
    """把每条折线的起点锚定到茎边界，返回 (新折线, 新长度, 已锚定条数, 新元信息)。

    metas 为 None 时第 4 项也是 None。

    茎外那圈黑色泡沫/海绵环在图像上不是根（模型判成背景是对的），标注却是从茎边开始
    画的折线 —— 也就是**那一段本来就存在，只是被挡住看不见**。这一步把预测折线的起点
    沿直线补到茎上，使每条根都从茎发出，长度也计入补回的这一段。

    **注意（2026-09-22 更正）**：这里原来写的是「与标注口径一致」，**那句是错的**。
    实测真值折线的端点距茎中位 27~490px（跨图差 18 倍），也就是标注起点**并不**在茎边。
    所以这一步**单边加长了预测**，与真值不同口径 —— 拿它和未锚定的真值比会系统性高估。
    要两侧可比，真值侧必须走 anchor_roots_to_stem（见下面那个函数）。

    折线会按「起点在茎上」重新定向：锚定的那一端被放到首位，并把茎上的最近点插为首点。
    两端都离茎超过阈值的折线原样保留（当作独立根计入，不丢信息）。
    """
    if stem_mask is None or not stem_mask.any() or not paths:
        return list(paths), list(lengths), 0, metas
    from scipy.spatial import cKDTree
    ys, xs = np.nonzero(stem_mask)
    tree = cKDTree(np.column_stack([xs, ys]))
    max_dist = stem_anchor_tolerance(stem_mask, factor, min_px, max_px)

    out_paths, out_lengths, out_metas, n_anchored = [], [], [], 0
    for i, P in enumerate(paths):
        is_meta = metas is not None
        if len(P) < 2:
            gain = None
        else:
            gain = stem_anchor_gain([P[0], P[-1]], tree, max_dist)
        if gain is None:
            out_paths.append(P)
            out_lengths.append(lengths[i])
            if is_meta:
                out_metas.append(metas[i])
            continue
        d, k, anchor = gain
        newP = list(P)
        if k == 0:
            newP.insert(0, anchor)
        else:                      # 起点在尾部 -> 整条反转，让起点落在首位
            newP.append(anchor)
            newP.reverse()
        out_paths.append(newP)
        out_lengths.append(lengths[i] + d)
        n_anchored += 1
        if is_meta:
            m = dict(metas[i])
            if k == 1:
                m["start_kind"], m["end_kind"] = m.get("end_kind"), m.get("start_kind")
                m["start_node"], m["end_node"] = m.get("end_node"), m.get("start_node")
            m["anchored"] = True
            out_metas.append(m)
    return out_paths, out_lengths, n_anchored, (out_metas if metas is not None else None)


def continuation_flags(roots, max_gap: float = 150.0, max_angle: float = 30.0) -> list:
    """标记哪些 RSML 折线是「上一条的续接」（**交叉处断开重画**留下的碎片）。

    用户的标注习惯（2026-09-22 确认）：根系交叉之后，看不出后续是哪个根，所以
    在交叉处断开、交叉过后另起一条重新画。于是**一条物理根 = 多条折线 = 多个 ID**。
    实测 plant_ S068-4_20251126ST 的 114 个 ID 里有 103 个是续接。

    判据：存在另一条折线 j，使 `end(j) → start(i)` 的距离 <= max_gap，且方向连续
    （夹角 < max_angle）。两个条件缺一不可 —— 只看距离会把「都从茎边发出、起点挨得近」
    的无关根误判；只看方向会把「恰好平行」的误判。

    **阈值是启发式的、没有干净解**：实测缺口距离分布是 20~400px 连续、无双峰
    （中位 111px），因为缺口宽度 = 压在上面那根根的宽度 + 标注时的随手留白，跨图不一样。
    所以这个函数**只用来抑制锚定**（宁可漏锚，也不要给中段凭空加几百像素），
    不要拿它当「根数」的口径用 —— 根数在这份数据上不可靠，见 tool/chain_diag/readme.md。
    """
    n = len(roots)
    flags = [False] * n
    pts = [np.asarray(r.points, dtype=np.float64) for r in roots]
    for i in range(n):
        if len(pts[i]) < 2:
            continue
        d_in = pts[i][0] - pts[i][1]
        nrm = float(np.linalg.norm(d_in))
        if nrm < _EPS:
            continue
        d_in = d_in / nrm
        for j in range(n):
            if i == j or len(pts[j]) < 2:
                continue
            gap = float(np.linalg.norm(pts[j][-1] - pts[i][0]))
            if gap > max_gap:
                continue
            d_out = pts[j][-1] - pts[j][-2]
            nrm = float(np.linalg.norm(d_out))
            if nrm < _EPS:
                continue
            cos = float(np.dot(d_out / nrm, d_in))
            if np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))) < max_angle:
                flags[i] = True
                break
    return flags


def anchor_roots_to_stem(roots, stem_mask, factor: float = 6.0,
                         min_px: float = 250.0, max_px: float = 600.0,
                         max_gap: float = 150.0, max_angle: float = 30.0) -> tuple:
    """把 **RSML 真值折线**按「起点锚定到茎」的口径补长，返回 (总长, 锚定条数)。

    **当前 `test.py` 走的是另一条路线**（真值掩码 → analyze_mask_anchored），因为
    那条路与预测侧是**逐字同一个函数**，口径不可能漂。本函数是**折线路线**的备选实现，
    保留用于交叉校验：两条路线在 11 张测试图上给出 +8.9%（折线）vs +7.5%（掩码），
    差 1.4 个百分点，互相印证。**要改锚定口径时，两条都跑一遍看是否仍然一致。**


    为什么真值也要锚：实测（2026-09-22，tool/chain_diag）证明

        GT 折线端点距茎中位 27~490px（跨图差 18 倍），锚定却用固定阈值
        clamp(6×r_eq, 250, 600)=600px —— 也就是**标注起点并不在茎边**。
        于是同一条流水线跑真值掩码 vs 模型掩码，真值 −3.4%、模型 +7.5%，
        差的这 10.9 个百分点全是锚定，不是模型。

    所以「用锚定」这个决定要求**两侧同口径**：预测补的那段，真值也得补。
    不补的话任何误差数字都混了口径差。

    走的是与预测侧**同一个** anchor_paths_to_stem，只是输入换成 RSML 折线，
    所以两边不会漂。roots 只要有 .points 与 .length 即可（鸭子类型）。

    **续接片段不锚**（见 continuation_flags）：标注在交叉处断开重画，中段碎片的起点
    在交叉点而不是茎上，把它们也锚过去等于凭空加几百像素。实测不抑制的话
    plant_ S068-4_20251126ST 会被推到 +88%（21228 → 39925），抑制后回到合理量级。
    """
    lengths = [float(r.length) for r in roots]
    if stem_mask is None or not stem_mask.any() or not roots:
        return float(sum(lengths)), 0
    cont = continuation_flags(roots, max_gap=max_gap, max_angle=max_angle)
    paths, keep, skipped = [], [], 0.0
    for r, c, L in zip(roots, cont, lengths):
        if c:                     # 续接片段：原样计入，不锚
            skipped += L
            continue
        paths.append(list(r.points))
        keep.append(L)
    if not paths:
        return float(sum(lengths)), 0
    _, out_lengths, n, _ = anchor_paths_to_stem(
        paths, keep, stem_mask, factor=factor, min_px=min_px, max_px=max_px)
    return float(sum(out_lengths)) + skipped, n


def _decimate(pts, spacing):
    """像素轨迹 -> 折线点 [(x, y), ...]：每约 spacing 像素取一点（同 RSML 控制点间距）。"""
    line = [pts[0][::-1]]
    acc = 0.0
    for prev, cur in zip(pts[:-1], pts[1:]):
        acc += _step_len(prev, cur)
        if acc >= spacing:
            line.append(cur[::-1])
            acc = 0.0
    end = pts[-1][::-1]
    if line[-1] != end:
        line.append(end)
    if len(line) < 2:
        line.append(end)
    return [(int(x), int(y)) for (x, y) in line]


def _bbox_with_margin(mask: np.ndarray, margin: int = _CROP_MARGIN):
    """掩码非零像素的外接框（四周各留 margin 像素，并夹到图内）。"""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    y0 = max(0, int(rows[0]) - margin)
    x0 = max(0, int(cols[0]) - margin)
    y1 = min(mask.shape[0], int(rows[-1]) + 1 + margin)
    x1 = min(mask.shape[1], int(cols[-1]) + 1 + margin)
    return y0, x0, y1, x1


def _skeleton_adj(mask: np.ndarray, erode_iters: int = 1):
    """掩码 -> 轻度腐蚀 -> 骨架化 -> 带权 8 邻域邻接表（坐标是**原图**分辨率下的）。

    返回 {像素(y, x): {邻居: 步长}}；掩码为空 / 腐蚀后为空 / 无骨架像素时返回 None。
    只做「掩码 -> 图」这一步，与 spur/min_len 无关，所以参数扫描时同一张图可以只算一次、
    多组阈值复用（_strands_from_adj 自己会拷贝，不会改到这里）。

    **先裁到掩码外接框再算**：框外全是 0，腐蚀/骨架化的结果与整幅图逐位相同
    （2026-09-17 在两张测试图上逐一比对过骨架像素集合），但代价按框面积算 ——
    实测框只占全图 42%，腐蚀 185→68ms、骨架化 679→301ms，单张省约 0.5s。
    坐标在返回前加回偏移，调用方拿到的仍是原图坐标，无需感知裁切。
    """
    if mask is None or mask.ndim != 2 or not mask.any():
        return None
    y0, x0, y1, x1 = _bbox_with_margin(mask)
    m = mask[y0:y1, x0:x1]
    for _ in range(erode_iters):  # 消除线宽厚度伪影
        m = erosion(m, footprint=disk(1))
    if not m.any():
        return None

    skel = skeletonize(m)
    ys, xs = np.nonzero(skel)
    pts = {(int(y) + y0, int(x) + x0) for y, x in zip(ys.tolist(), xs.tolist())}
    if not pts:
        return None

    # 带权 8 邻域邻接表
    adj = {p: {} for p in pts}
    for (y, x) in pts:
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                q = (y + dy, x + dx)
                if q in pts:
                    w = _step_len((y, x), q)
                    adj[(y, x)][q] = w
                    adj[q][(y, x)] = w
    return adj


def _strands_from_adj(adj, spur: float = 30.0, min_len: float = 20.0,
                      normalize: bool = True) -> list:
    """邻接表 -> 剪枝 -> 分链，返回 [[长度(像素), 像素轨迹[(y,x), ...]], ...]（长度降序）。

    输入 adj 不会被修改（先拷贝再剪枝），便于参数扫描复用同一张图的骨架。
    """
    work = {p: dict(nb) for p, nb in adj.items()}
    _prune(work, spur)
    if not work:
        return []
    entries = []
    for comp in _components(work):
        entries.extend(_strands_of_component(comp, work, spur, min_len, normalize))
    entries.sort(key=lambda t: t[0], reverse=True)
    return entries


def analyze_mask_ex(mask: np.ndarray, spur: float = 30.0, min_len: float = 20.0,
                    erode_iters: int = 1, with_paths: bool = False,
                    spacing: float = 50.0, normalize_count: bool = True) -> dict:
    """mask: (h, w) bool 原图分辨率二值掩码。

    返回 {"count", "lengths"(降序), "total"}；with_paths=True 时附加
    "paths": [[(x, y), ...], ...]（与 lengths 一一对应、同序的抽稀折线）。
    normalize_count=False 关闭「计数归一」（见 _strands_of_component），
    用于评估侧根数（默认为 True，与历史结果一致）。
    """
    empty = {"count": 0, "lengths": [], "total": 0.0}
    adj = _skeleton_adj(mask, erode_iters)
    if adj is None:
        return {**empty, "paths": []} if with_paths else empty

    entries = _strands_from_adj(adj, spur, min_len, normalize_count)
    if not entries:
        return {**empty, "paths": []} if with_paths else empty
    lengths = [t[0] for t in entries]
    out = {"count": len(lengths), "lengths": lengths,
           "total": float(sum(lengths))}
    if with_paths:
        out["paths"] = [_decimate(t[1], spacing) for t in entries]
        # 每条折线的端点类型（叶端 / 分叉点）：侧根 = 起点在分叉点、终点在叶端且不长的折线
        out["strand_meta"] = [t[2] for t in entries]
    return out


def analyze_mask_anchored(mask, stem_mask=None, spur: float = 30.0,
                          min_len: float = 20.0, erode_iters: int = 1,
                          spacing: float = 50.0, normalize_count: bool = True,
                          factor: float = 6.0, min_px: float = 250.0,
                          max_px: float = 600.0) -> dict:
    """analyze_mask_ex + 「起点锚定到茎」：统计量全部是锚定后的口径。

    给 stem_mask 时，每条折线的起点会被补到茎边界并计入长度（见 anchor_paths_to_stem）。
    返回的字典与 analyze_mask_ex 同构，另加 "anchored_count"（锚定成功的条数）。
    """
    st = analyze_mask_ex(mask, spur=spur, min_len=min_len, erode_iters=erode_iters,
                         with_paths=True, spacing=spacing,
                         normalize_count=normalize_count)
    st["anchored_count"] = 0
    if stem_mask is None or not st["paths"]:
        return st
    paths, lengths, n, metas = anchor_paths_to_stem(
        st["paths"], st["lengths"], stem_mask,
        factor=factor, min_px=min_px, max_px=max_px,
        metas=st.get("strand_meta"))
    st["paths"], st["lengths"] = paths, lengths
    st["total"] = float(sum(lengths))
    st["strand_meta"] = metas
    st["anchored_count"] = n
    return st


def anchor_gain_for_trace(trace, tree, max_dist: float) -> float:
    """给「像素轨迹」(skeleton_stats 内部的 (y, x) 口径) 算需要补的锚定长度。

    参数扫描（tool/tune_stats）复用邻接表时用，保证扫描口径与部署一致。
    """
    ends = [(trace[0][1], trace[0][0]), (trace[-1][1], trace[-1][0])]
    g = stem_anchor_gain(ends, tree, max_dist)
    return g[0] if g is not None else 0.0


def analyze_mask(mask: np.ndarray, spur: float = 30.0, min_len: float = 20.0,
                 erode_iters: int = 1, normalize_count: bool = True) -> dict:
    """兼容入口：只要统计量（count / lengths / total）。"""
    return analyze_mask_ex(mask, spur, min_len, erode_iters, with_paths=False,
                           normalize_count=normalize_count)


def extract_root_paths(mask: np.ndarray, spur: float = 30.0, min_len: float = 20.0,
                       erode_iters: int = 1, spacing: float = 50.0,
                       normalize_count: bool = True):
    """返回逐根折线 [[(x, y), ...], ...]（按长度降序，与 analyze_mask 的长度一一对应）。

    spacing: 折线点抽稀间距(px)，默认 50，与标注 RSML 的
    controlpointseparation="50" 一致。可直接喂给 common.rsml_export.write_rsml。
    """
    return analyze_mask_ex(mask, spur, min_len, erode_iters,
                           with_paths=True, spacing=spacing,
                           normalize_count=normalize_count)["paths"]
