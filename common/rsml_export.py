"""把预测出的根系折线导出为 RSML 文件。

⚠️ **本模块目前没有调用点**（2026-09-30 起）。根系标注统一进 labelme json 之后，
推理侧也改成只出 labelme json（见 common/labelme_export.py），RSML 那条
（RootNav / rsml-visualizer）不再有人用。保留本模块是为了「要恢复时不至于重写」——
在 inference.py 里调一次 write_rsml 即可。

2026-09-14 起本项目不再区分一级/二级根，inference.py 当初只走「扁平」写法
（每条折线一个 plant、全部 primary）。下面的「嵌套」写法（hierarchy=...）与
check_nested_rsml 同样保留但已无调用点，仅作历史对照。

两种写法：
1. **扁平**（不带 hierarchy，向后兼容旧行为）：每条折线 = 一个 plant 下的一条 primary 根；
2. **嵌套**（带 hierarchy，与标注口径一致）：每条主根一个 plant，其下 `<root ID="i.1" label="primary">`，
   长在它上面的侧根写成嵌套的 `<root ID="i.1.1" label="secondary">`（更深的 `i.1.1.1` 类推）。
   子根首点与父根折线上的挂载点重合（标注就是这么存的：293/293 条子根首点 = 父根某控制点），
   所以导出时把挂载点插进父根的 `<point>` 列表 —— 点在父根直线上，几何不变。

结构（与标注一致）：
    <rsml ...>
      <metadata>version/unit/resolution/last-modified/software/user/file-key</metadata>
      <scene>
        <plant ID="1" label="sugarcane">
          <annotations>annotation</annotations>
          <root ID="1.1" label="primary">
            <geometry>
              <rootnavspline controlpointseparation="50" tension="0.5">
                <point x="1638" y="786" />
                ...
              </rootnavspline>
            </geometry>
            <root ID="1.1.1" label="secondary">
              <geometry>...</geometry>
            </root>
          </root>
        </plant>
"""
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

_NS = {"xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
       "xmlns:xsd": "http://www.w3.org/2001/XMLSchema"}


def _attach_point(att, snap_tol):
    """返回 (挂载点, 是否吸附)。

    snap_tol=None 表示「吸附全部已判定的挂载」—— 这些挂载本来就在 attach_tol（默认 12.5px）内
    通过了几何检验，而分叉点物理上就在父根中线上，吸附过去更接近真实；给具体数值则超过就不吸附。
    """
    if att is None:
        return None, False
    return att.point, (snap_tol is None or att.dist <= snap_tol)


def _oriented(pts, att, snap_tol: float):
    """把折线摆成「首点=挂载点」的方向，能吸附则把首点换成分叉点（落在父根折线上）。

    返回 (点列, 是否反转过)；反转会让段号整体镜像，调用方要据此换算子根的挂载段号
    （见 _flip_seg），否则孙辈的挂载点会插错位置。
    """
    P = [(float(x), float(y)) for x, y in pts]
    q, ok = _attach_point(att, snap_tol)
    if q is None or not ok:
        return P, False
    if att.end == 0:
        P[0] = q
        return P, False
    P = P[::-1]
    P[0] = q
    return P, True


def _flip_seg(seg: int, t: float, n: int) -> tuple:
    """折线反转后，原来的第 seg 段、段内 t 变成第 n-2-seg 段、段内 1-t。"""
    return n - 2 - seg, 1.0 - t


def _insert_mount_points(points, mounts):
    """把若干挂载点插进父根折线（mounts: [(seg, t, 点), ...]）。

    挂载点一律按**原始**折线插值算出（这样它与子根首点是同一个浮点数，取整后逐位一致），
    再按段号一次性装配，避免「插入一个点后后面段的索引整体错位」的经典错误。
    同一位置 1px 内的多条子根共用一个控制点，与已有控制点重合的也不重复插。
    """
    src = [(float(x), float(y)) for x, y in points]
    by_seg = {}
    for seg, t, q in mounts:
        if 0 <= seg < len(src) - 1:
            a, b = src[seg], src[seg + 1]
            qq = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))
        else:                       # 兜底：段号越界就用投影点本身
            qq = (float(q[0]), float(q[1]))
        by_seg.setdefault(seg, []).append((t, qq))

    out = []
    used = list(src)
    for i, p in enumerate(src):
        out.append(p)
        for _, qq in sorted(by_seg.get(i, ())):
            # 只有「同一个分叉点」才复用（0.25px 内）；仅相距零点几像素的两条子根各插各的
            if any((qq[0] - u[0]) ** 2 + (qq[1] - u[1]) ** 2 <= 0.0625 for u in used):
                continue
            out.append(qq)
            used.append(qq)
    return out


def write_rsml(path, file_key: str, polylines, software: str = "U-Net inference",
               user: str = "PCC", plant_label: str = "sugarcane",
               controlpoint_separation: int = 50, tension: str = "0.5",
               hierarchy=None, snap_tol=None) -> Path:
    """把若干折线写成 RSML 文件。

    polylines: [[(x, y), ...], ...]，来自 skeleton_stats.extract_root_paths。
    hierarchy: common.root_hierarchy.Hierarchy（可选）。传了 = 按主根/侧根嵌套输出；
               不传 = 每条折线各写一个 plant（旧行为，逐字节不变）。
    snap_tol: 子根首点吸附到挂载点的距离上限，None = 全部已判定的挂载都吸附（见 _attach_point）。

    返回写入的路径。
    """
    rsml = ET.Element("rsml", _NS)
    md = ET.SubElement(rsml, "metadata")
    ET.SubElement(md, "version").text = "1.0"
    ET.SubElement(md, "unit").text = "pixel"
    ET.SubElement(md, "resolution").text = "xxx dpi"
    ET.SubElement(md, "last-modified").text = datetime.now().strftime(
        "%Y/%m/%d %H:%M:%S")
    ET.SubElement(md, "software").text = software
    ET.SubElement(md, "user").text = user
    ET.SubElement(md, "file-key").text = file_key
    scene = ET.SubElement(rsml, "scene")

    def add_spline(root_el, pts):
        geo = ET.SubElement(root_el, "geometry")
        spline = ET.SubElement(geo, "rootnavspline", {
            "controlpointseparation": str(controlpoint_separation),
            "tension": tension})
        for (x, y) in pts:
            ET.SubElement(spline, "point", {"x": str(int(round(x))),
                                            "y": str(int(round(y)))})

    if hierarchy is None:
        # ---- 旧行为：每条折线 = 一个 plant 下的一条 primary 根 ----
        for i, pts in enumerate(polylines, 1):
            if len(pts) < 2:
                continue  # 少于 2 个点无法构成折线
            plant = ET.SubElement(scene, "plant",
                                  {"ID": str(i), "label": plant_label})
            ET.SubElement(plant, "annotations").text = "annotation"
            root = ET.SubElement(plant, "root",
                                 {"ID": f"{i}.1", "label": "primary"})
            add_spline(root, pts)
    else:
        # ---- 嵌套：主根 i.1，侧根挂在父根下（ID 逐层加一段） ----
        children = {}
        for i, att in enumerate(hierarchy.attach):
            if att is not None:
                children.setdefault(att.parent, []).append(i)
        for k in children:  # 子根按挂载位置排序（父根首点端为起点方向）
            children[k].sort(key=lambda j: (hierarchy.attach[j].seg,
                                            hierarchy.attach[j].t))

        def emit(parent_el, idx, root_id, label):
            """写一条根：先几何、再递归写子根（元素顺序与标注一致）。

            本根的子根挂载点插进本根折线（点在直线上，几何不变），子根本身由递归写。
            """
            pts, flipped = _oriented(polylines[idx], hierarchy.attach[idx], snap_tol)
            mounts = [(hierarchy.attach[j].seg, hierarchy.attach[j].t,
                       hierarchy.attach[j].point) for j in children.get(idx, ())]
            if flipped:   # 本根被反向写了，子根的挂载段号要跟着镜像
                mounts = [(*_flip_seg(seg, t, len(pts)), q) for seg, t, q in mounts]
            root = ET.SubElement(parent_el, "root", {"ID": root_id, "label": label})
            add_spline(root, _insert_mount_points(pts, mounts))
            for m, j in enumerate(children.get(idx, ()), 1):
                emit(root, j, f"{root_id}.{m}", "secondary")

        plant_no = 0
        for i in range(len(polylines)):
            if hierarchy.parent[i] is not None or len(polylines[i]) < 2:
                continue  # 只从主根起笔（侧根跟在父根下面写）
            plant_no += 1
            plant = ET.SubElement(scene, "plant",
                                  {"ID": str(plant_no), "label": plant_label})
            ET.SubElement(plant, "annotations").text = "annotation"
            emit(plant, i, f"{plant_no}.1", "primary")

    ET.indent(rsml, space="  ")
    path = Path(path)
    ET.ElementTree(rsml).write(path, encoding="utf-8", xml_declaration=True)
    return path


def check_nested_rsml(path) -> dict:
    """回读自检：数层级、查「子根首点是否精确落在父根某个控制点上」。

    返回 {"plants", "roots", "primary", "secondary", "children",
          "children_first_point_on_parent", "ok"}；ok=False 说明嵌套结构没写对。
    """
    from common.rsml_parse import parse_rsml
    roots = parse_rsml(path)
    by_id = {r.root_id: r for r in roots}
    children = 0
    hit = 0
    for r in roots:
        if "." in r.root_id:
            parent_id = r.root_id.rsplit(".", 1)[0]
            p = by_id.get(parent_id)
            if p is None:
                continue
            children += 1
            if r.points and any(abs(r.points[0][0] - q[0]) < 0.5
                                and abs(r.points[0][1] - q[1]) < 0.5
                                for q in p.points):
                hit += 1
    primary = sum(1 for r in roots if r.label == "primary")
    secondary = sum(1 for r in roots if r.label != "primary")
    return {"plants": len({r.root_id.split(".")[0] for r in roots}),
            "roots": len(roots), "primary": primary, "secondary": secondary,
            "children": children, "children_first_point_on_parent": hit,
            "ok": (children == 0 or hit == children)}
