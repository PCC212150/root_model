"""把预测出的根系（折线 + 轮廓多边形）导出为 labelme json —— 与标注**同一种格式**。

用途：把预测结果直接叠在标注上对比。根系标注 2026-09-30 起就在 labelme json 里
（`label="root"` + `shape_type="linestrip"`，见 [tool/merge_annot](../tool/merge_annot/readme.md)），
本模块产出同样的结构，所以 labelme 能直接打开，也能和标注 json 并排比对。

与 `rsml_export.write_rsml` 的关系：**这是替代关系**。根系的两种表示曾同时输出，
2026-09-30 起只出 labelme json —— 标注全在 labelme json 里了，`.rsml` 那条
（RootNav / rsml-visualizer）已经没人用，多写一份是纯开销。
`rsml_export.py` 仍留着（无调用点），要恢复只需在 inference.py 里再调一次。

结构与 labelme 6.x 完全一致（键序、`imageData: null`、`flags: {}` 都照抄），
所以导出的文件混进标注里不会有任何格式差异。

**坐标系**：折线一律是**原图坐标**（`skeleton_stats.analyze_mask_ex` 的 `paths`
就是原图系），与标注同一个坐标系，不需要任何换算。

**根系的两种形状**（2026-10-06 起同时支持）：折线 = 骨架中心线（统计用）；
多边形 = 掩码轮廓（`mask_to_polygons`），供 labelme 里改成新的多边形标注。
两者**同名 `root`**，解析侧有规则「有 root 多边形就忽略 root 折线」。

**怎么在 labelme 里打开**：json 里的 `imagePath` 写的是**原图文件名**（不是 overlay），
labelme 按它去 json 同级目录找图。所以要建一个同时有图与 json 的临时目录，
并且**把 json 改个名**（如 `<名>_pred.json`）—— 导出的文件名与标注 json 同名，
直接拷到数据集目录里会**覆盖掉标注**。

> ⚠️ **不要写进 `datasets/`**：那是训练数据。预测结果混进去会污染划数据集
> （`tool/separate_dataset` 按文件夹里所有文件分组）和对照工具（`tool/check_convert`
> 会把它当一组标注来读）。看结果另建目录。
"""
import json
from pathlib import Path

import numpy as np

ROOT_LABEL = "root"
CHECK_LABEL = "check_background"
STEM_LABEL = "stem"
ROOT_SHAPE_TYPE = "linestrip"
ROOT_POLY_TYPE = "polygon"        # 根系轮廓（2026-10-06 起；与折线同名 root 并存）

# labelme 单条形状的确切键序（多一个少一个都会和既有标注文件不一致）
SHAPE_KEYS = ("label", "points", "group_id", "description",
              "shape_type", "flags", "mask")


def make_shape(label: str, shape_type: str, points) -> dict:
    """构造一条 labelme 形状。

    坐标一律 `float()` 强转：`json.dumps` 遇到 `np.int64` / `np.float32` 会抛 `TypeError`
    （它们不是 Python 内建类型），而折线是一路从 numpy 里算出来的。
    """
    return {
        "label": label,
        "points": [[float(x), float(y)] for x, y in points],
        "group_id": None,
        "description": "",
        "shape_type": shape_type,
        "flags": {},
        "mask": None,
    }


def mask_to_polygons(mask, approx_px: float = 2.0, min_area: float = 400.0,
                     max_points: int = 400) -> list:
    """二值掩码 → labelme 用的多边形点列 [[(x, y), ...], ...]（原图坐标）。

    只取**外轮廓**（`RETR_EXTERNAL`）：根/茎都是细长结构，孔洞没有意义，而内轮廓
    会让 labelme 里的形状变成一堆碎片。`approxPolyDP` 把锯齿压成折线
    （`approx_px` 是允许的偏差，单位像素），点数再按等距抽稀到 `max_points` 以内
    —— labelme 拖一个几千点的多边形会卡。

    `min_area` 用来丢掉碎片：比它小的连通块不导出（根掩码碎成几十块时，
    这些都倒出来会淹掉真正要看的东西）。
    """
    import cv2
    m = mask.astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        ap = cv2.approxPolyDP(c, approx_px, True) if len(c) > 4 else c
        pts = ap.reshape(-1, 2)
        if len(pts) < 3:
            continue
        if len(pts) > max_points:              # 等距抽稀，保住首尾
            idx = np.linspace(0, len(pts) - 1, max_points).round().astype(int)
            pts = pts[idx]
        out.append([(float(x), float(y)) for x, y in pts])
    return out


def write_labelme_json(path, image_height: int, image_width: int, polylines=(),
                       image_path: str = None, check_box=None,
                       stem_polygons=(), root_polygons=(), version: str = "6.0.0") -> Path:
    """把预测折线写成 labelme json，返回写入的路径。

    polylines:      [[(x, y), ...], ...] 预测根系折线，**原图坐标**。
    image_path:     写进 `imagePath` 的值 —— **必须是磁盘上那张原图的真实文件名**，
                    因为 labelme 靠它回找图片。默认按输出文件名 + `.jpg` 猜。
    check_box:      (x0, y0, x1, y1) 原图坐标的 ROI，写成 `check_background` rectangle；
                    没启用检查范围时传 None（那张形状就不写）。
    stem_polygons:  想连茎一起导出就传多边形点列。默认不传 —— 茎的轮廓要从掩码里
                    提取（`5472x3648` 上几百 ms/张），而茎是标注里已有的东西，
                    导出预测的意义在根系。
    root_polygons:  根系**轮廓**多边形（见 `mask_to_polygons`）。与 `polylines`
                    （中心线）**同名 `root`、同时并存** —— 这是有意的：labelme 里改完
                    可以直接当标注用；而解析侧有规则「图里有 root 多边形就忽略 root
                    折线」（见 common/labelme.py），不会把一根数成两根。

    形状顺序照标注文件：root 折线 → root 多边形 → check_background → stem。
    """
    shapes = []
    for pts in polylines:
        if len(pts) >= 2:                      # 少于 2 点构不成折线，与 write_rsml 同口径
            shapes.append(make_shape(ROOT_LABEL, ROOT_SHAPE_TYPE, pts))
    for poly in root_polygons:
        if len(poly) >= 3:
            shapes.append(make_shape(ROOT_LABEL, ROOT_POLY_TYPE, poly))
    if check_box is not None:
        x0, y0, x1, y1 = check_box
        shapes.append(make_shape(CHECK_LABEL, "rectangle",
                                 [(x0, y0), (x1, y1)]))
    for poly in stem_polygons:
        if len(poly) >= 3:                     # 多边形至少要 3 点
            shapes.append(make_shape(STEM_LABEL, "polygon", poly))

    path = Path(path)
    data = {
        "version": version,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_path if image_path is not None else path.with_suffix(".jpg").name,
        "imageData": None,
        "imageHeight": int(image_height),
        "imageWidth": int(image_width),
    }
    # **不写 newline=""** —— 这里要的正是「跟 labelme 一模一样」：labelme 用的是普通
    # 文本模式 `json.dump`，所以在 Windows 上落盘是 CRLF、在 Linux 上是 LF。
    # 本项目自己的标注 json 就是 CRLF（labelme 写的）。工具里其他地方用 newline=""
    # 是为了**就地改文件时不碰换行**，而这里是新建文件，照抄 labelme 的行为才对。
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path
