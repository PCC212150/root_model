"""labelme 标注（labels/other/*.json）解析：根系折线 + 茎横截面 + 检查范围。

json 结构（labelme 6.x）：
    {"version","flags","shapes":[{label,shape_type,points,...}],
     "imagePath","imageData","imageHeight","imageWidth"}

本项目用三类 label：
    root              根系折线（linestrip）—— **2026-09-30 起并入本文件**
    stem              甘蔗茎的横截面（polygon）
    check_background  框选检查的范围（rectangle，两点轴对齐）

历史：根系原先单独存在 `.rsml`（RootNav 的 XML）里，2026-09-30 起统一进本 json，
三通道一份文件。`root` 折线要按 `rsml_parse.parse_rsml` 的同款口径处理
（**少于 2 个点的丢弃**），否则同一份标注在新旧两种格式下会算出不同的掩码 ——
见 [tool/merge_annot](../tool/merge_annot/readme.md)。

解析结果统一为**原图坐标**下的矢量（点列 / 外接矩形），画掩码时按目标尺寸换算
（见 common/gt_mask.py），这样同一份标注可以按任意输入分辨率绘制。

注意：labelme 默认会把整张图 base64 塞进 imageData（本项目 29/31 个文件都有，单个最大 28MB），
解析后必须立刻丢弃，绝不能把整个 dict 缓存下来。
"""
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

STEM_LABEL = "stem"
CHECK_LABEL = "check_background"
ROOT_LABEL = "root"
# labelme 里画折线用 linestrip；line 也接住 —— 都是「一串点连起来」，
# 与 rsml 的 <point> 序列语义相同，没必要因为画图工具的选择不同就丢掉标注。
# **polygon 是另一回事**（2026-10-06 起）：它表示"根的**真实轮廓**"，掩码要按
# **填充**画（见 dataset.build_target_masks），不是描一条 10px 宽的线。
ROOT_SHAPE_TYPES = ("linestrip", "line", "polygon")
ROOT_FILLED_TYPES = ("polygon",)

# 「root 的多边形与折线同名并存」的告警只提醒一次（见 parse_other 里的说明）
_warned_mixed_root = False


@dataclass
class OtherLabels:
    """一张图的 labelme 标注（原图坐标）。"""

    path: Path = None
    stems: list = field(default_factory=list)       # [[(x, y), ...], ...] 茎多边形
    check_rect: tuple = None                        # (x0, y0, x1, y1) 检查范围外接矩形
    roots: list = field(default_factory=list)       # [[(x, y), ...], ...] 根系折线（>=2 点）
    root_polygons: list = field(default_factory=list)  # 与 roots 逐条对应：True=轮廓多边形(填充)
    info: dict = field(default_factory=dict)        # 自检信息（shape 数 / 面积占比 / 告警）

    @property
    def ok(self) -> bool:
        """两类标注都解析到了才为 True（缺任一类的图，训练时会屏蔽对应通道的损失）。

        **只看 stem / check，不含 roots**：这个属性管的是「这两条通道的损失要不要屏蔽」，
        而**零根是合法的负样本**（真值里就是「这张图没有根」，见 plant_S003-3），
        不能拿「roots 为空」判成缺标注。
        """
        return bool(self.stems) and self.check_rect is not None


def _finite_points(points) -> list:
    """过滤非有限坐标（NaN/inf），返回 [(x, y), ...]。"""
    out = []
    for p in points or []:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            out.append((x, y))
    return out


def _polygon_area(points) -> float:
    """鞋带公式算多边形面积（用于面积占比告警，不做精确统计）。"""
    if len(points) < 3:
        return 0.0
    s = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def parse_other(json_path, image_size=None, verbose: bool = True) -> OtherLabels:
    """解析 labelme json，返回 OtherLabels（原图坐标）。

    image_size: (w, h) 磁盘上原图的实际尺寸；给了就校验与 json 里记录的一致
                （标注画在别的尺寸上会导致掩码整体错位，必须报错而不是静默继续）。
    verbose: 打印异常/未知 label 的告警（每张图每类只打一次）。
    """
    json_path = Path(json_path)
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    data.pop("imageData", None)          # 594MB 的坑：解析后立刻丢弃

    lab = OtherLabels(path=json_path)
    warns = []

    if image_size is not None:
        w, h = int(image_size[0]), int(image_size[1])
        jw, jh = data.get("imageWidth"), data.get("imageHeight")
        if jw is not None and jh is not None and (int(jw), int(jh)) != (w, h):
            raise ValueError(
                f"{json_path.name}: 标注尺寸 {int(jw)}x{int(jh)} 与图片实际尺寸 "
                f"{w}x{h} 不一致，掩码会整体错位。请重新导出/导出后再标注。")

    shapes = data.get("shapes") or []
    labels_seen = []
    root_lines, root_polys = [], []      # root 的两种形状先分开收，循环后再定（见下）
    for s in shapes:
        label = (s.get("label") or "").strip()
        labels_seen.append(label)
        pts = _finite_points(s.get("points"))
        if len(pts) != len(s.get("points") or []):
            warns.append(f"有 {len(s.get('points') or []) - len(pts)} 个非有限坐标点被丢弃")

        if label == ROOT_LABEL:
            # 与 rsml_parse.py:52 同口径：少于 2 个点无法构成折线，丢弃。
            # 两格式必须一致，否则同一份标注换个格式就会算出不同的掩码。
            if len(pts) >= 2:
                st = (s.get("shape_type") or "").strip()
                if st not in ROOT_SHAPE_TYPES:
                    warns.append(f"root 的 shape_type={st!r} 不是折线，已按折线读取")
                (root_polys if st in ROOT_FILLED_TYPES else root_lines).append(pts)
            else:
                warns.append(f"root 只有 {len(pts)} 个点，忽略")
        elif label == STEM_LABEL:
            if len(pts) >= 3:
                lab.stems.append(pts)
            else:
                warns.append(f"stem 只有 {len(pts)} 个点，忽略")
        elif label == CHECK_LABEL:
            if not pts:
                warns.append("check_background 没有有效点，忽略")
                continue
            # 2 点/4 点/旋转矩形一律取外接矩形（当前标注全是 2 点轴对齐）
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            box = (min(xs), min(ys), max(xs), max(ys))
            lab.check_rect = box if lab.check_rect is None else _union(lab.check_rect, box)
        else:
            warns.append(f"未知 label {label!r}，已忽略")

    # root 的两种形状**同名并存**时的取舍（2026-10-06，配合 labelme_export 的导出）：
    # 多边形是"真实轮廓"（权威），折线只是它的中心线 —— 两个都收会把一根数成两根，
    # 直接污染根数与掩码。所以**有多边形就只用多边形**（老的折线数据集没有多边形，不受影响）。
    if root_polys:
        if root_lines:
            # **整个进程只提醒一次**（2026-10-08 改）：标注工具产出的 json **每张都是**
            # 「多边形 + 折线」这个形态（折线记根长、多边形是掩码真值，两者本就该并存），
            # 一份 11 张的 GT 会刷 11 行一模一样的告警，噪音盖过别的信息。
            # 这条提醒本身保留 —— 它是"一根不会被数成两根"的可见证据。
            global _warned_mixed_root
            if not _warned_mixed_root:
                _warned_mixed_root = True
                warns.append(f"root 同时有 {len(root_polys)} 个多边形和 {len(root_lines)} 条折线："
                             f"**只取多边形**（折线是多边形的中心线，两个都算会重复计根）。"
                             f"同类文件不再重复提醒")
        lab.roots = root_polys
        lab.root_polygons = [True] * len(root_polys)
    else:
        lab.roots = root_lines
        lab.root_polygons = [False] * len(root_lines)

    lab.info = {
        "n_shapes": len(shapes),
        "labels": labels_seen,
        "n_root": len(lab.roots),
        "n_stem": len(lab.stems),
        "check_rect": lab.check_rect,
        "warns": warns,
    }
    if image_size is not None:
        w, h = int(image_size[0]), int(image_size[1])
        if lab.stems:
            area = sum(_polygon_area(p) for p in lab.stems)
            lab.info["stem_area_ratio"] = area / float(w * h)
        if lab.check_rect:
            x0, y0, x1, y1 = lab.check_rect
            lab.info["check_area_ratio"] = max(0.0, (x1 - x0)) * max(0.0, (y1 - y0)) / float(w * h)

    if verbose and warns:
        print(f"[标注告警] {json_path.name}: " + "；".join(sorted(set(warns))))
    return lab


def _union(a, b) -> tuple:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
