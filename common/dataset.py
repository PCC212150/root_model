"""PyTorch Dataset：图片 + 三通道真值（根系折线 / 茎 / 检查范围）。

数据布局（train/test 同构）——**两种都支持，自动识别**：

    扁平（2026-09-30 起，当前用法）：
        <data_dir>/<名>.jpg|png     原图
        <data_dir>/<名>.json        茎横截面 + 检查范围 + 根系折线（三通道一份文件）
        <data_dir>/<名>.rsml        根系折线（**旧格式**，可选）

    分目录（旧布局，仍然能读）：
        <data_dir>/images/<名>.jpg
        <data_dir>/labels/other/<名>.json
        <data_dir>/labels/roots/<名>.rsml

根系折线有两种存放格式，`load_annot()` 统一读取：
    · 新格式：json 里 `label="root"` 的 linestrip，三通道一份文件；
    · 旧格式：单独的 `.rsml`（RootNav 的 XML）。
新格式优先；**退回旧格式时会大声告警** —— 静默回退会让「格式迁移是否生效」无从判断
（新旧两条路读出来的东西一模一样，指标自然也一样），
见 [tool/merge_annot](../tool/merge_annot/readme.md)、[tool/check_convert](../tool/check_convert/readme.md)。

目标张量 (3, H, W) 的通道顺序见 config.CLASS_NAMES：
    0 = root   根系（折线画线，线宽按比例换算）
    1 = stem   茎横截面（labelme polygon 填充）
    2 = check  检查范围（labelme rectangle 填充；缺标注时置全 True = 不做限制）

三张掩码都在**模型输入分辨率**上直接绘制（坐标先按比例换算），比「原图分辨率画好再
缩放」快约 200 倍，且细线不会被最近邻降采样漏掉（见 common/gt_mask.py 说明）。
"""
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import config
from common import gt_mask, image_io
from common.labelme import OtherLabels, parse_other
from common.rsml_parse import Root, parse_rsml

# 通道序号（与 config.CLASS_NAMES 一致）
CH_ROOT, CH_STEM, CH_CHECK = 0, 1, 2


def _images_dir(data_dir) -> Path:
    """图片目录：新布局 <data_dir>/images；旧布局（图片与标注同目录）回退到 data_dir 本身。"""
    d = Path(data_dir)
    sub = d / "images"
    return sub if sub.is_dir() else d


def _roots_dir(data_dir) -> Path:
    """旧格式的根系标注目录：`labels/roots`（兼容旧命名 `labels/root`）。

    **没有 labels/ 目录时回退到 data_dir 本身** —— 2026-09-30 起数据集改成扁平布局
    （图片与标注同层，见模块 docstring），此时 `<名>.rsml` 就放在图片旁边。
    """
    from config import ROOTS_LABEL_SUBDIR, ROOTS_LABEL_SUBDIR_ALT
    d = Path(data_dir)
    for name in (ROOTS_LABEL_SUBDIR, ROOTS_LABEL_SUBDIR_ALT):
        sub = d / name
        if sub.is_dir():
            return sub
    return d


def _other_dir(data_dir) -> Path:
    """labelme 标注目录（新格式下根系也在这里）。

    同样回退到 data_dir 本身：扁平布局下 `<名>.json` 与图片同层。
    """
    from config import OTHER_LABEL_SUBDIR
    d = Path(data_dir)
    sub = d / OTHER_LABEL_SUBDIR
    return sub if sub.is_dir() else d


# 数据集支持**按来源分子目录**（2026-10-08 起）：
#
#     datasets/root/train/
#         GT/     人工标的多边形 + 像素掩码（masks/）—— 真值
#         scr/    老数据：折线膨胀出来的伪标签（只有 json 的 10px 折线）
#
# 每个来源子目录都是一份**独立完整**的扁平数据集（图 + json + 自己的 masks/），
# 所以 `load_annot` 传进去的 data_dir 就是子目录本身，找 masks/<名>.png 自然是对的。
# 好处：两种真值口径（掩码 vs 折线）**各自走各自的路**，不需要在代码里分支；
# 训练侧还能按来源给损失加权（伪标签降权，见 train.py 的 --src-weight）。
#
# 下面这几个名字是**结构目录**，不是来源 —— 扫描时要跳过：
# masks/overlay 是标注工具的产物目录，images/labels 是老的嵌套布局。
SOURCE_DIRS_RESERVED = {"masks", "overlay", "images", "labels", "cache"}


def source_dirs(data_dir) -> list:
    """[(来源名, 该来源的目录), ...]。

    没有来源子目录时返回 `[("", data_dir)]` —— 也就是**原来那种一层的布局照常能用**，
    老数据集一个字节都不用动。
    """
    d = Path(data_dir)
    if not d.is_dir():
        return []
    subs = []
    for p in sorted(d.iterdir()):
        if not p.is_dir() or p.name in SOURCE_DIRS_RESERVED:
            continue
        if any(q.is_file() and q.suffix.lower() in config.IMAGE_EXTS
               for q in p.iterdir()):
            subs.append((p.name, p))
    return subs or [("", d)]


def source_of(data_dir, img_path) -> str:
    """这张图属于哪个来源：`<data_dir>/<来源>/图.jpg` -> "<来源>"；直接在 data_dir 下 -> ""。"""
    p = Path(img_path).parent
    d = Path(data_dir)
    if p != d and p.parent == d and p.name not in SOURCE_DIRS_RESERVED:
        return p.name
    return ""


def find_other(data_dir, stem):
    """找该图的 labelme 标注（茎/检查范围/新格式的根系）；没有返回 None。

    **顶层找不到就去各来源子目录里找** —— 见 `sample_dir`。
    """
    p = _other_dir(sample_dir(data_dir, stem)) / f"{stem}.json"
    return p if p.exists() else None


def sample_dir(data_dir, stem) -> Path:
    """这个样本（stem）**实际住在哪个目录**：顶层，或某个来源子目录（GT/scr/...）。

    2026-10-08 加。由来：`discover_pairs` 改成会扫来源子目录之后，
    `load_annot` 还只在顶层找 `<data_dir>/<名>.json` —— 于是每张图都被判成
    "没有 json"，88 张全报"缺 stem/check 标注"（训练时那两个通道的损失会被整体屏蔽）。
    样本住哪儿，标注和 `masks/<名>.png` 就在哪儿的旁边，所以统一从这里解析。
    """
    d = Path(data_dir)
    for base in [d] + [b for _s, b in source_dirs(d) if b != d]:
        if ((_other_dir(base) / f"{stem}.json").exists()
                or (_roots_dir(base) / f"{stem}.rsml").exists()):
            return base
    return d


def discover_pairs(data_dir, image_exts=None) -> list:
    """返回 [(stem, 图片路径, 标注路径), ...]：图片要有**至少一个**标注文件才配对。

    配对条件（2026-09-30 放宽）：`labels/other/<名>.json` 或 `labels/roots/<名>.rsml`
    **有一个就算**。第三项返回实际存在的那个（优先 json），但**它只是「有标注」的证据**——
    根系到底从哪读由 `load_annot()` 决定。

    这里刻意**不去读 json 内容**：判断「json 里有没有 root 折线」要解析整个文件
    （转换后单个 ~90KB），而本函数被 train / test / tune_stats / chain_diag 四条路径调用，
    每次都扫一遍不值当。停在 `stat()` 级别即可，真正的判定在 `load_annot` 里做一次。

    注意语义变化：过去「有图片、无 rsml」的图会被**静默跳过**；现在只要还有 json 就算数，
    只是那个 json 若没有 root 折线，root 通道会是空的（`_warn_empty_root` 会就此告警）。
    """
    if image_exts is None:
        from config import IMAGE_EXTS
        image_exts = IMAGE_EXTS
    data_dir = Path(data_dir)
    pairs = []
    # **顶层 + 各来源子目录**都扫（见 source_dirs 的说明）。顶层是空的也没关系
    # （只有子目录时它扫出 0 个，不影响）。
    for _src, base in source_dirs(data_dir):
        img_dir = _images_dir(base)
        if not img_dir.is_dir():
            continue
        roots_dir = _roots_dir(base)
        other_dir = _other_dir(base)
        for img_path in sorted(p for p in img_dir.iterdir()
                               if p.suffix.lower() in image_exts):
            json_path = other_dir / f"{img_path.stem}.json"
            rsml_path = roots_dir / f"{img_path.stem}.rsml"
            if json_path.exists():
                pairs.append((img_path.stem, img_path, json_path))
            elif rsml_path.exists():
                pairs.append((img_path.stem, img_path, rsml_path))
    return pairs


# 像素级掩码的子目录名。**必须和 create_datasets/root/annotate_io.py 里的
# MASK_SUBDIR 一致** —— 标注工具往那儿写，这里读。
MASK_SUBDIR = "masks"


@dataclass
class Annot:
    """一个样本的全部标注。`roots` 的来源见 `source`。"""

    stem: str
    orig_size: tuple = None                 # (w0, h0) 磁盘上图片的实际尺寸
    data_dir: object = None                 # 数据集目录（找 masks/<名>.png 用）
    roots: list = field(default_factory=list)   # list[Root]（json/rsml 两条路产出同型对象）
    lab: OtherLabels = None                 # 茎 / 检查范围（没有 json 时是空壳）
    source: str = "json"                    # "json"=新格式 | "rsml"=旧格式（会大声告警）
    json_path: Path = None
    rsml_path: Path = None


_warned_legacy = set()


def _warn_legacy(stem, json_path, rsml_path, n_roots):
    """旧格式仍生效时的告警。同一个文件只吵一次。

    **这不是「贴心的提示」，是让格式迁移可验证的前提。** 静默回退的话，
    「转换前后指标逐位相同」会因为两边读的是同一份 rsml 而永远成立——
    转换哪怕一个字节都没改，验证也会通过。
    """
    key = str(rsml_path)
    if key in _warned_legacy:
        return
    _warned_legacy.add(key)
    why = ("labels/other 里没有对应 json" if json_path is None
           else f"{json_path.name} 里没有 root 折线")
    print(f"[警告] 旧格式仍在生效：{stem} —— {why}，"
          f"正从 {Path(rsml_path).name} 读 {n_roots} 条根系折线。\n"
          f"        新格式要求 root 也写进 labels/other/<名>.json（labelme linestrip）。\n"
          f"        转换： python tool\\merge_annot\\merge_annot.py --dir \"<数据集目录>\" -r --dry-run",
          flush=True)


def load_annot(data_dir, stem, orig_size=None, verbose=True) -> Annot:
    """读取一个样本的全部标注：**json（含 root 折线）优先，退回 .rsml**。

    orig_size=(w, h) 会拿去校验 json 里记的尺寸与磁盘图片是否一致——
    标注画在别的尺寸上会让掩码整体错位，那种情况必须报错而不是静默继续。

    返回的 `source` 标注了根系实际来自哪条路，调用方（尤其是验证脚本）应当据此判断
    格式迁移是否真的生效。
    """
    # 样本可能住在来源子目录里（GT/scr）；标注与 masks/<名>.png 都在它旁边，
    # 所以这里把 data_dir 换成**它实际住的目录**（见 sample_dir）。
    data_dir = sample_dir(data_dir, stem)
    json_path = find_other(data_dir, stem)
    rsml_path = _roots_dir(data_dir) / f"{stem}.rsml"
    rsml_path = rsml_path if rsml_path.exists() else None
    lab = (parse_other(json_path, image_size=orig_size, verbose=verbose)
           if json_path is not None else OtherLabels(path=None, info={}))

    if lab.roots:
        return Annot(stem, orig_size, data_dir, [Root(points=p) for p in lab.roots],
                     lab, "json", json_path, rsml_path)
    if rsml_path is not None:
        roots = parse_rsml(rsml_path)
        if verbose:
            _warn_legacy(stem, json_path, rsml_path, len(roots))
        return Annot(stem, orig_size, data_dir, roots, lab, "rsml", json_path, rsml_path)
    # 有 json 但没有 root 折线、也没有 rsml：照常当「这张图没有根」的负样本，
    # 由 _warn_empty_root 去提醒（它可能是合法的负样本，也可能是漏标）。
    return Annot(stem, orig_size, data_dir, [], lab, "json", json_path, None)


def plant_key(name: str) -> str:
    """图片名 -> 植株标识，用于「整株进出」的数据划分。

    'plant_ S062-1_20251116ST' -> 'S062-1'
    'root_C001-1_20241229CK'   -> 'C001-1'
    """
    s = str(name).replace(" ", "")
    for pre in ("plant_", "root_"):
        if s.startswith(pre):
            s = s[len(pre):]
            break
    head, sep, tail = s.rpartition("_")
    # 末段是「日期+批次」这类记号时去掉（如 20251116ST / 20241229CK）
    if sep and len(tail) >= 6 and any(ch.isdigit() for ch in tail):
        s = head
    return s


def _corner_fill(img: np.ndarray, patch: int = 32) -> tuple:
    """取图像四角小块的均值颜色，作为旋转增强的填充色（避免黑边假象）。"""
    h, w = img.shape[:2]
    corners = [img[0:patch, 0:patch], img[0:patch, w - patch:w],
               img[h - patch:h, 0:patch], img[h - patch:h, w - patch:w]]
    mean = np.mean(np.concatenate(corners).reshape(-1, 3), axis=0)
    return (int(mean[0]), int(mean[1]), int(mean[2]))


def _affine_matrix(angle: float, scale: float, w: int, h: int) -> tuple:
    """绕画布中心「先缩放、再旋转」的仿射矩阵（PIL AFFINE 的输出->输入方向）。"""
    rad = math.radians(angle)
    cos, sin = math.cos(rad), math.sin(rad)
    cx, cy = w / 2.0, h / 2.0
    a, b = cos / scale, sin / scale
    d, e = -sin / scale, cos / scale
    return (a, b, cx - a * cx - b * cy, d, e, cy - d * cx - e * cy)


def _affine_rgb(arr: np.ndarray, angle: float, scale: float, fill: tuple) -> np.ndarray:
    h, w = arr.shape[:2]
    im = Image.fromarray(arr).transform(
        (w, h), Image.AFFINE, _affine_matrix(angle, scale, w, h),
        resample=Image.Resampling.BILINEAR, fillcolor=fill)
    return np.asarray(im, dtype=np.uint8)


def _affine_mask(mask: np.ndarray, angle: float, scale: float) -> np.ndarray:
    h, w = mask.shape[:2]
    im = Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).transform(
        (w, h), Image.AFFINE, _affine_matrix(angle, scale, w, h),
        resample=Image.Resampling.NEAREST, fillcolor=0)
    return np.asarray(im, dtype=np.uint8) > 127


def _jitter(img: np.ndarray) -> np.ndarray:
    """亮度/对比度/饱和度抖动（模拟不同批次的光照与白平衡差异）。"""
    out = img.astype(np.float32)
    b = random.uniform(*config.AUG_BRIGHTNESS)
    c = random.uniform(*config.AUG_CONTRAST)
    out = (out * b - 127.5) * c + 127.5
    out = np.clip(out, 0, 255).astype(np.uint8)
    s = random.uniform(*config.AUG_SATURATION)
    if abs(s - 1.0) > 0.01:
        from PIL import ImageEnhance
        out = np.asarray(ImageEnhance.Color(Image.fromarray(out)).enhance(s),
                         dtype=np.uint8)
    return out


_warned_empty_root = set()


def _warn_empty_root(annot):
    """同一个样本只吵一次（数据集构造 + 评测会对同一张图问好几遍）。

    **只是提醒，不改变行为**：这种图照常按「无根」负样本参与训练。
    """
    key = (str(annot.json_path or ""), str(annot.rsml_path or ""))
    if key in _warned_empty_root:
        return
    _warned_empty_root.add(key)
    src = (Path(annot.rsml_path).name if annot.source == "rsml" and annot.rsml_path
           else Path(annot.json_path).name if annot.json_path else annot.stem)
    print(f"[提示] {src} 里没有任何根系折线，将按「这张图没有根」参与训练。\n"
          f"        如果确实没有根（合法负样本，如 plant_S003-3），忽略本条即可；\n"
          f"        如果是**漏标**（画了没保存/忘了画），请补标或把这张图移出数据集 ——"
          f"那种情况下会把模型教坏。")


def load_root_mask_png(annot, target_size):
    """读标注工具存的**像素级 root 掩码**（`<data_dir>/masks/<名>.png`，黑底白条）。

    返回 bool (h1, w1)；没有这个文件就返回 None（调用方保留多边形那条路的结果）。

    只提供 **root**：茎和检查框本来就是模型预测的、人没动过，继续走 json 那条路。
    root 是人一笔笔改出来的，值得逐像素保真 —— 比"把多边形在目标分辨率上重新栅格化"
    准在两点：少了轮廓近似的量化损失；也不会有"漏画折线的根被当成背景"这种假负样本
    （实测这批图 root 面积 +1.6%）。

    **缩放口径**：原图 -> 目标尺度用**面积平均再取过半**（PIL BOX + `>127`），不是最近邻
    —— 5472 缩到 1024 时根只有约 1.9px 宽，最近邻会把细根整段抽没。
    """
    from PIL import Image
    if getattr(annot, "data_dir", None) is None:
        return None
    p = Path(annot.data_dir) / MASK_SUBDIR / f"{annot.stem}.png"
    if not p.exists():
        return None
    with Image.open(p) as im:
        arr = np.asarray(im.convert("L"), dtype=np.uint8)
    w1, h1 = int(target_size[0]), int(target_size[1])
    if (arr.shape[1], arr.shape[0]) != (w1, h1):
        arr = np.asarray(Image.fromarray(arr).resize((w1, h1), Image.BOX),
                         dtype=np.uint8)
    return arr > 127


def build_target_masks(annot, target_size, mask_width, use_mask: bool = True):
    """画三通道真值掩码，返回 (masks[h,w,3] bool, chan_valid[3] float)。

    annot: `load_annot()` 的返回值；原图尺寸从 `annot.orig_size` 取 ——
           刻意不做成参数，免得调用方传一个跟图片对不上的尺寸进来。

    test.py 评测时复用这同一份实现，保证「训练真值」与「评测真值」口径一致。

    缺 labelme 标注时：stem 通道置空、check 通道置全 True（统计不被截断），
    并把对应通道标为「无效」——训练时该通道的损失会被屏蔽，避免模型学成「这里没有框」。
    """
    w1, h1 = target_size
    orig_size = annot.orig_size
    masks = np.zeros((h1, w1, 3), dtype=bool)
    valid = np.zeros(3, dtype=np.float32)

    line_w = gt_mask.target_line_width(mask_width, orig_size, target_size)
    # root 两种画法，**逐条判断、可以混用**（2026-10-06 起支持多边形轮廓）：
    #   polygon （新）→ **填充**：根的看得见轮廓，宽度是标注时量出来的真实宽度；
    #   linestrip/line（老）→ 描一条 mask_width 宽的线（10px@原图尺度）。
    # 这两套是**两把尺子**（前景宽度不同 → Dice/IoU/面积不可跨口径比），
    # 但根数、骨架总长的口径不变。
    flags = getattr(annot.lab, "root_polygons", None) or []
    line_pts, fill_pts = [], []
    for i, r in enumerate(annot.roots):
        if len(r.points) < 2:
            continue
        pts = gt_mask.scale_points(r.points, orig_size, target_size)
        if i < len(flags) and flags[i] and len(pts) >= 3:
            fill_pts.append(pts)
        else:
            line_pts.append(pts)
    if line_pts:
        masks[:, :, CH_ROOT] = gt_mask.draw_polylines_at(line_pts, target_size, line_w)
    if fill_pts:
        masks[:, :, CH_ROOT] |= gt_mask.draw_polygons_at(fill_pts, target_size)

    # 根系标注是配对前提，但**文件存在 ≠ 里面画了东西**：RSMLGenerator 里没标就保存会留下
    # 一个没有 <geometry> 的空壳（如 560 字节、0 个控制点）；新格式下则是一个
    # 没有 root 形状的 json。
    # 这种图**照常当「无根」负样本参与训练**（valid 保持 1）—— 实测数据集里的
    # plant_S003-3_20251116ST 就是真·没有根的合法样本，用户确认过。
    # 但仍要告警一次：它也可能是「忘记画就保存」的漏标，那种情况下会把模型教坏，得让人看见。
    valid[CH_ROOT] = 1.0
    # 空 root 的告警挪到最后统一发（下面可能被像素级掩码覆盖，那时以掩码为准）

    lab = annot.lab
    if lab.stems:
        masks[:, :, CH_STEM] = gt_mask.draw_polygons_at(
            [gt_mask.scale_points(p, orig_size, target_size) for p in lab.stems],
            target_size)
        valid[CH_STEM] = 1.0
    if lab.check_rect is not None:
        (x0, y0), (x1, y1) = gt_mask.scale_points(
            [(lab.check_rect[0], lab.check_rect[1]),
             (lab.check_rect[2], lab.check_rect[3])], orig_size, target_size)
        masks[:, :, CH_CHECK] = gt_mask.draw_rects_at([(x0, y0, x1, y1)],
                                                      target_size)
        valid[CH_CHECK] = 1.0
    if valid[CH_CHECK] == 0:                   # 没有检查框：不限制统计范围
        masks[:, :, CH_CHECK] = True

    # **最后一步：如果有像素级 root 掩码，用它替换 root 通道**（茎/检查框保持上面
    # 多边形那条路的结果 —— 它们本来就是模型预测的，人没动过）。
    # 放在最后覆盖，是为了让"有没有掩码"只影响 root 一个通道，别的逻辑一行都不用分叉。
    # ⚠️ 掩码和多边形是**两把尺子**（实测 root 面积差约 1.6%）：同一批数据要么都有
    #    掩码、要么都没有，混着用等于一半图一个口径。
    # use_mask=False 时**忽略像素掩码**，一律走多边形/折线那条路 ——
    # 用来做「掩码 vs 多边形」的 A/B 对照（同一批图、同一份标注，只换真值口径）。
    root_px = load_root_mask_png(annot, target_size) if use_mask else None
    if root_px is not None:
        masks[:, :, CH_ROOT] = root_px
        valid[CH_ROOT] = 1.0
    if not masks[:, :, CH_ROOT].any():
        _warn_empty_root(annot)
    return masks, valid


def _annot_bbox(masks):
    """**根系 / 茎**的并集包围盒 (x0, y0, x1, y1)（右/下开区间）；全空返回 None。

    **不含 check 通道**：缺 labelme 标注时 check 通道被置成全 True，把它算进来
    包围盒就等于整图，「把裁块中心放在标注上」的偏置会彻底失效。
    """
    core = masks[:, :, CH_ROOT] | masks[:, :, CH_STEM]
    if not core.any():
        return None
    rows = np.flatnonzero(core.any(axis=1))
    cols = np.flatnonzero(core.any(axis=0))
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


class RootDataset(Dataset):
    """逐项返回 (img[3,H,W] float32 0~1, gt[3,H,W] float32 0/1, name, chan_valid[3])。

    构造时完成解码/画掩码/缩放（较慢），训练时仅做轻量增强。
    chan_valid 标记该图哪些通道有真值（缺 labelme 标注时 stem/check 为 0），
    训练侧据此屏蔽对应通道的损失。

    **两种预处理模式**，由 `crop` 决定：

        crop=0（默认）  整图缩放到长边 max_side 再训练。快、省内存，但 5472x3648 的图
                        缩到 1024 时原图 10px 宽的根只剩 1.9px —— 细根被抹掉。
        crop=N>0        **切片训练**：原图**不缩放**，每次随机裁 N×N 的块。
                        同样 1280 个像素，块里的根就是原图的 10px。代价是原图与
                        全分辨率掩码都要常驻内存（5472x3648 约 120MB/张）。

    为什么不做「整图原始分辨率训练」：实测（tool/mem_probe）5472 batch1 要 116GB，
    加了梯度检查点也还有 84GB —— 24G 卡差 4.8 倍，不可能。切片是唯一能拿到
    「原始分辨率下的根宽」的路径。见 config.CROP_SIZE 的注释。
    """

    def __init__(self, data_dir, names=None, max_side=1024, stride=16,
                 mask_width=5, augment=False, seed=0, crop=0, crop_repeat=1,
                 full=False, use_mask=True):
        self.augment = augment
        self.use_mask = bool(use_mask)
        self.data_dir = Path(data_dir)
        self.crop = int(crop or 0)
        # full=True：**整图、原始分辨率、不缩放也不裁**。给「验证切片模型」用 ——
        # 切片模型的部署形式是「原图分块跑再拼」，验证必须在同一形式下做，
        # 否则量到的是另一个分布（实测拿单个中心裁块验证时，check 通道的 GT 几乎全 True，
        # 预测全 True 就能拿 Dice 0.99，而整图上只有 0.79 —— 指标完全失效）。
        self.full = bool(full)
        self.stride = int(stride)
        # 切片模式下每张原图在一个 epoch 里抽多少个块。**这不是可有可无的调优项**：
        # 不重复的话一个 epoch 只有「图片数」个样本（本项目 23 个，batch2 才 11 步），
        # 而 LR 平台期(25)与早停(120)都是按**轮**计的 —— 一个 epoch 才 4.5 秒的话，
        # LR 几分钟就降到下限、早停十几分钟就触发，等于根本没训。抽 8 个块让一个
        # epoch 的步数与整图模式同量级。见 config.CROP_REPEAT。
        # **只在 augment 时重复**：验证集用的是确定性裁块（裁在标注包围盒中心），
        # 重复抽 8 次拿到的是同一块 —— 白白把验证时间乘以 8，指标一点没变。
        self.repeat = max(1, int(crop_repeat)) if (self.crop and self.augment) else 1
        if self.crop:
            if self.crop <= 0 or self.crop % self.stride:
                raise ValueError(f"crop 必须是 {self.stride} 的正整数倍（U-Net 要下采样 4 次"
                                 f"），收到 {crop}")
        pairs = discover_pairs(self.data_dir)
        if names is not None:
            wanted = set(names)
            pairs = [p for p in pairs if p[0] in wanted]
            if len(pairs) != len(wanted):
                missing = sorted(wanted - {p[0] for p in pairs})
                print(f"[警告] 有 {len(missing)} 个名字在数据集中找不到配对: {missing[:5]}")
        self.names = [p[0] for p in pairs]
        # 每个样本的「来源」（GT / scr / ""）—— 训练侧按它给损失加权（伪标签降权）
        srcs = [source_of(self.data_dir, p[1]) for p in pairs]

        self.items = []
        n_no_other = 0
        n_legacy = 0
        bytes_full = 0
        for name, img_path, _annot_path in pairs:
            img = image_io.load_rgb(img_path)
            h0, w0 = img.shape[:2]
            annot = load_annot(self.data_dir, name, (w0, h0))
            n_legacy += annot.source == "rsml"
            if self.crop or self.full:
                if self.crop and self.crop > min(w0, h0):
                    raise ValueError(f"crop={self.crop} 比 {name} 的短边({min(w0, h0)})还大，"
                                     f"裁不出块来")
                # 掩码在**原图分辨率**上画（orig=target），裁块/整图时坐标天然对齐
                masks, valid = build_target_masks(annot, (w0, h0), mask_width,
                                                  use_mask=self.use_mask)
                bytes_full += img.nbytes + masks.nbytes
                item = {"name": name, "img": img, "masks": masks, "valid": valid,
                        "fill": _corner_fill(img),
                        "bbox": _annot_bbox(masks) if self.crop else None}
            else:
                w1, h1 = image_io.target_size(w0, h0, max_side, stride)
                masks, valid = build_target_masks(annot, (w1, h1), mask_width,
                                                  use_mask=self.use_mask)
                item = {"name": name,
                        "img": image_io.resize_rgb(img, w1, h1),   # (h1,w1,3) uint8
                        "masks": masks,                            # (h1,w1,3) bool
                        "valid": valid,                            # (3,) float32
                        "fill": _corner_fill(img), "bbox": None}
            if valid[CH_STEM] == 0 or valid[CH_CHECK] == 0:
                n_no_other += 1
            item["src"] = srcs[len(self.items)]
            self.items.append(item)
        if self.crop:
            print(f"[切片训练] 原图不缩放，每轮随机裁 {self.crop}×{self.crop}，"
                  f"每图抽 {self.repeat} 块 → 一个 epoch {len(self.items) * self.repeat} 个样本；"
                  f"{len(self.items)} 张原图+全分辨率掩码常驻内存 ≈ "
                  f"{bytes_full / 2**30:.2f} GB")
        elif self.full:
            print(f"[整图验证] 原图不缩放、不裁块，{len(self.items)} 张常驻内存 ≈ "
                  f"{bytes_full / 2**30:.2f} GB")
        if n_no_other:
            print(f"[警告] {n_no_other} 张图缺 stem/check 标注（labels/other 里没有对应 "
                  f"json），训练时这两个通道的损失会被屏蔽。")
        # 多来源时把构成打出来（GT / scr 各多少组）—— 混着训的时候这是最该先看见的一行
        n_src = {}
        for it in self.items:
            n_src[it["src"] or "(无来源)"] = n_src.get(it["src"] or "(无来源)", 0) + 1
        if len(n_src) > 1:
            print("[来源] " + " | ".join(f"{k}: {v} 组" for k, v in sorted(n_src.items())))
        if n_legacy:
            print(f"[警告] {n_legacy} / {len(self.items)} 张图仍用**旧格式**（根系在 .rsml 里）。\n"
                  f"        建议跑一次 tool\\merge_annot 把根系并进 json —— 新旧混用时"
                  f"「有 json 却没有根」的样本最容易被漏掉。")

    def __len__(self):
        return len(self.items) * self.repeat

    def _crop_window(self, it):
        """返回裁块左上角 (x0, y0)。训练时随机、验证时固定（保证 val 指标跨轮可比）。"""
        img = it["img"]
        h, w = img.shape[:2]
        n = self.crop

        def clamp(v, hi):
            return int(max(0, min(hi, v)))

        if not self.augment:
            # 验证集：**确定性**裁在标注包围盒中心。每轮裁的位置一样，val Dice 才能
            # 跨轮比较；随机裁会让 val 曲线抖到没法用来早停/选模型。
            bb = it["bbox"]
            cx, cy = ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2) if bb else (w / 2, h / 2)
            return clamp(cx - n / 2, w - n), clamp(cy - n / 2, h - n)
        if it["bbox"] is None or random.random() < config.CROP_BG_PROB:
            # 纯随机位置：留这个口子是为了让模型也见到整块背景，
            # 否则推理时容易在空白水域产生假阳性。
            return random.randint(0, w - n), random.randint(0, h - n)
        # 把块中心放在标注包围盒（外扩 n/2）内随机取 —— 保证裁块一定碰到标注，
        # 不然 5472x3648 上纯随机会有相当比例的块整块落在空白处，白跑一轮。
        bx0, by0, bx1, by1 = it["bbox"]
        cx = random.uniform(bx0 - n / 2, bx1 + n / 2)
        cy = random.uniform(by0 - n / 2, by1 + n / 2)
        return clamp(cx - n / 2, w - n), clamp(cy - n / 2, h - n)

    def __getitem__(self, idx):
        # repeat>1 时同一张原图在一个 epoch 里出现 repeat 次（各抽一个不同的块）。
        # 用整除而不是取模：保证每张图**恰好**被抽 repeat 次，epoch 的语义不乱。
        it = self.items[idx // self.repeat]
        if self.crop:
            x0, y0 = self._crop_window(it)
            n = self.crop
            # numpy 切片是视图，代价接近 0；全分辨率掩码因此可以直接常驻内存
            img = np.ascontiguousarray(it["img"][y0:y0 + n, x0:x0 + n])
            m = np.ascontiguousarray(it["masks"][y0:y0 + n, x0:x0 + n])
        else:
            img = it["img"]
            m = it["masks"]
        if self.augment:
            if random.random() < 0.5:
                img = np.flip(img, axis=1)
                m = np.flip(m, axis=1)
            # 旋转 + 尺度抖动一起做（同一个仿射矩阵作用于图像与三张掩码，保证对齐）
            angle = random.uniform(-config.AUG_ROTATE_DEG, config.AUG_ROTATE_DEG)
            scale = random.uniform(*config.AUG_SCALE)
            if abs(angle) > 0.3 or abs(scale - 1.0) > 0.01:
                img = _affine_rgb(img, angle, scale, it["fill"])
                m = np.stack([_affine_mask(m[:, :, c], angle, scale)
                              for c in range(m.shape[2])], axis=2)
            img = _jitter(img)
        img = np.ascontiguousarray(img)
        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        y = torch.from_numpy(np.ascontiguousarray(
            m.transpose(2, 0, 1), dtype=np.float32))          # (3,H,W)
        return x, y, it["name"], torch.from_numpy(it["valid"].copy())
