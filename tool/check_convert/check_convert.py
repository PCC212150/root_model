"""对照工具：证明「rsml → 合并进 json」这一步没有改变任何真值。

用途有两个：

1. **抓指纹 / 比对**（本阶段主力）。转换前后各跑一次，逐样本比：
   折线条数、点数、坐标序列哈希、三通道掩码哈希、图片与 json 记的尺寸。
   差异会报到「第几条折线的第几个点、两侧各是什么」，不是干巴巴一句「哈希不同」。

2. **断言来源**。加载器是「json 有 root 就用 json、否则退回 rsml」——
   **这个回退会让上面那套比对变成空转**：回退时新路径*就是*旧路径，比对当然通过，
   而转换其实一个字节都没改。所以：
     · `--compare` 在转换后若发现还有样本走 rsml，**直接判失败**；
     · `--provenance` 更进一步，直接问加载器「你这张图实际用的是哪个来源」。

用法（转换前后都用 `--roots-from auto`）：
    # 转换前抓基线
    python check_convert.py --dir "D:\\数据集总表\\root" -r --capture base.json

    # 转换后比对（同时会检查「有没有漏转的」）
    python check_convert.py --dir "D:\\数据集总表\\root" -r --compare base.json

    `auto` 的含义是「json 里有 root 就用 json，否则退回 rsml」——转换前那 73 组会
    自动落到 rsml，转换后落到 json，正好是我们要比的两侧。而且比对结束时会检查
    **转换后是否还有样本落在 rsml**，有就判失败（那就是空转）。
    `--roots-from rsml` 只在想强制「一律看旧格式」时用；注意那些只有 json 没有
    rsml 的样本（比如 `抽出来的图片/C` 里已经转好的 2 个）会因读不出根系而报错。

    # 加载器支持新格式之后：断言 73 组全都走 json
    python check_convert.py --dir "datasets\\root\\train" --dir "datasets\\root\\test" --provenance

    # 注意：本工具只认**单一** label，现在只服务 root_model 的数据集。
    # plant_model 的折线分 shoot（茎）/leaf（叶）两类、加载器也不是 load_annot，
    # 那边用 plant_model\\tool\\check_lines（2026-10-05 起）。

为什么自己读文件、不走项目的加载代码：
    `common/dataset.py` 的加载器正是被验证的对象之一。拿它来算「转换前」，
    等于用被测对象给自己出题。所以这里只借三样**不受本次改动影响**的东西：
    `common.rsml_parse.parse_rsml`（旧格式的权威读法）、
    `common.gt_mask` 的画线原语、`config.py` 的常量。
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common import gt_mask  # noqa: E402
from common.rsml_parse import parse_rsml  # noqa: E402

ROOT_LABEL = "root"


def read_raw(path):
    """按 utf-8 读文本，**不做换行转换**。

    不要写成 `Path.read_text(newline=...)` —— 那个参数 **Python 3.13 才有**，
    而项目跑在 3.12 的 `pcc` 环境里（3.12 上会 `TypeError`）。
    """
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="比对 rsml 与合并后 json 的根系真值是否一致",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dir", required=True, action="append",
                   help="目标文件夹（可重复）。既认扁平布局，也认 labels/other + labels/roots")
    p.add_argument("-r", "--recursive", action="store_true", help="递归处理子文件夹")
    p.add_argument("--roots-from", choices=("rsml", "auto"), default="auto",
                   help="折线来源：auto（默认）=json 里有 root 就用 json，否则退回 .rsml；"
                        "rsml=一律读 .rsml（只有 json 的样本会因此读不出根系）")
    p.add_argument("--capture", default=None, help="把逐样本指纹写到这个 json")
    p.add_argument("--compare", default=None, help="与之前抓的指纹比对")
    p.add_argument("--label", default=ROOT_LABEL,
                   help=f"json 里折线的 label（默认 {ROOT_LABEL!r}；"
                        f"plant_model 数据集用 --label plant）")
    p.add_argument("--no-mask", action="store_true",
                   help="跳过掩码哈希（快得多；坐标逐位相同则掩码必然相同）")
    p.add_argument("--provenance", action="store_true",
                   help="断言加载器实际用的是 json 来源（需要 common.dataset.load_annot）")
    return p.parse_args(argv)


# ---------------------------------------------------------------- 找文件

def _pick(dirs, stem, exts):
    """在若干候选目录里找同名文件；找到 0 个或多个都返回 None（不猜）。"""
    hits = []
    for d in dirs:
        if d is None or not d.is_dir():
            continue
        hits += [p for p in d.iterdir()
                 if p.is_file() and p.stem == stem and p.suffix.lower() in exts]
    return hits[0] if len(hits) == 1 else None


def siblings(json_path: Path, stem: str):
    """返回 (rsml 路径, 图片路径)，找不到给 None。

    扁平布局：三件套同在 json 那一层。
    项目布局（`<split>/{images,labels/other,labels/roots}`）：
    json 在 `labels/other`，rsml 在 `labels/roots`，图片在与 `labels/` **平级**的 `images/`
    —— 也就是 json 往上数三层，别写成 `labels/images`。
    """
    p = json_path.parent
    rsml = _pick([p, p.parent / "roots", p.parent / "root"], stem, {".rsml"})
    img = _pick([p, p.parent.parent / "images"], stem, config.IMAGE_EXTS)
    return rsml, img


# ---------------------------------------------------------------- 读标注

def roots_from_rsml(path):
    """旧格式的权威读法：直接用项目自己的 parse_rsml。"""
    return [[(float(x), float(y)) for x, y in r.points]
            for r in parse_rsml(path) if len(r.points) >= 2]


def roots_from_json(path, label=ROOT_LABEL):
    """新格式：shapes 里 label==label 的折线。<2 点丢弃，与 rsml_parse.py:52 同口径。"""
    data = json.loads(read_raw(path))
    out = []
    for s in data.get("shapes") or []:
        if (s.get("label") or "").strip() != label:
            continue
        try:
            pts = [(float(p[0]), float(p[1])) for p in s.get("points") or []]
        except (TypeError, ValueError, IndexError):
            continue
        if len(pts) >= 2:
            out.append(pts)
    return out


# ---------------------------------------------------------------- 指纹

def coords_hash(polys):
    """把整批坐标按序打包成 double 再哈希 —— 逐位敏感，不是近似比较。"""
    h = hashlib.sha256()
    for pts in polys:
        for x, y in pts:
            h.update(struct.pack("<dd", float(x), float(y)))
    return h.hexdigest()


def image_size(path):
    """只读文件头拿尺寸，不解码像素。"""
    from PIL import Image
    with Image.open(path) as im:
        return [int(im.size[0]), int(im.size[1])]          # (w, h)


def mask_hash(polys, size):
    import numpy as np
    w = gt_mask.target_line_width(config.MASK_LINE_WIDTH, size, size)
    m = gt_mask.draw_polylines_at(polys, size, w)
    return hashlib.sha256(np.packbits(m).tobytes()).hexdigest(), int(m.sum())


def fingerprint(json_path: Path, roots_from, no_mask, label=ROOT_LABEL):
    """返回一个样本的指纹 dict。读不了就给 {"error": ...}。"""
    stem = json_path.stem
    rsml, img = siblings(json_path, stem)
    try:
        data = json.loads(read_raw(json_path))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        return {"error": f"json 读不了: {e}"}

    census = {}
    for s in data.get("shapes") or []:
        k = (s.get("label") or "").strip() or "(空)"
        census[k] = census.get(k, 0) + 1

    if roots_from == "auto" and census.get(label):
        polys, src = roots_from_json(json_path, label), "json"
    elif rsml is not None:
        polys, src = roots_from_rsml(rsml), "rsml"
    elif roots_from == "auto":
        # 新格式下「这张图没有根」就表现为 json 里没有 root 形状 —— 那是**合法的真值**
        # （如 plant_S003-3，用户确认过），不是「读不出来」。旧 rsml 被归档之后就只剩
        # 这一条路可走，所以必须把它当成 0 条根，而不是报错。
        polys, src = [], "json"
    else:
        return {"error": "指定了 --roots-from rsml，但没有同名 rsml"}

    flat = [p for pts in polys for p in pts]
    ent = {
        "source": src,
        "rel": json_path.name,
        "rsml": rsml.name if rsml else None,
        "image": img.name if img else None,
        "n_polylines": len(polys),
        "n_points": len(flat),
        "coords_sha256": coords_hash(polys),
        "head": [[round(x, 4), round(y, 4)] for x, y in flat[:3]],
        # 折线那一类（root/plant）**故意**要从 0 变成 N，所以不能参与「相等」比较。
        # 剩下的 stem / check 计数必须一字不变。
        "n_shapes": {k: v for k, v in census.items() if k != label},
        "n_root_shapes": census.get(label, 0),
        "top_keys": list(data.keys()),
        "json_dims": [data.get("imageWidth"), data.get("imageHeight")],
        # 完整体折线留一份：比对时用它定位「第一处不同是哪个点」，
        # 也让指纹本身成为自足的证据（datasets/ 不入 git，这是唯一的留存）。
        "polylines": [[[float(x), float(y)] for x, y in pts] for pts in polys],
    }
    if img is not None:
        ent["img_dims"] = image_size(img)
        if not no_mask:
            ent["mask_sha256"], ent["root_px"] = mask_hash(
                polys, (ent["img_dims"][0], ent["img_dims"][1]))
    return ent


def collect(args):
    man = {}
    for d in args.dir:
        root = Path(d)
        if not root.is_dir():
            sys.exit(f"[错误] 文件夹不存在: {root}")
        files = sorted(root.rglob("*.json") if args.recursive else root.glob("*.json"))
        for f in files:
            ent = fingerprint(f, args.roots_from, args.no_mask, args.label)
            if f.stem in man:
                print(f"[警告] 名字重复，后者覆盖前者: {f}")
            ent["rel"] = (str(f.parent.relative_to(root) / f.name)
                          if f.parent != root else f.name)
            man[f.stem] = ent
    return man


# ---------------------------------------------------------------- 比对

def first_diff(a, b):
    """返回第一处不同的 (折线序号, 点序号, a 值, b 值)；完全相同返回 None。

    坐标哈希对不上时靠它定位 —— 「哈希不同」本身没法查。
    """
    if len(a) != len(b):
        return ("折线总数", "", len(a), len(b))
    for i, (pa, pb) in enumerate(zip(a, b)):
        if len(pa) != len(pb):
            return (i, "点数", len(pa), len(pb))
        for j, (qa, qb) in enumerate(zip(pa, pb)):
            if qa != qb:
                return (i, j, qa, qb)
    return None


# 这几个字段不参与「字段级差异」报告：
#   rel       随抓取目录变，不是数据
#   source    rsml → json 的变化是**预期**的，且已由 check_sources 专门把关
#   rsml      只记「折线是从哪个 rsml 读的」；转换后根系改从 json 读，这个文件就不再参与，
#             归档掉之后变成 None 是**预期**的（真实的数据差异会体现在 coords/mask 上）
#   polylines 坐标单列出来用 first_diff 定位，不整块 diff
#   n_root_shapes  本来就该从 0 变成 N，只作信息报告
SKIP = {"rel", "source", "rsml", "polylines", "n_root_shapes"}


def compare(before, after):
    n_bad = 0

    flipped = sorted(k for k in set(before) & set(after)
                     if not before[k].get("n_root_shapes")
                     and after[k].get("n_root_shapes"))
    if flipped:
        print(f"新增 root 的形状: {len(flipped)} 组（这是本次转换的目的，不是差异）")
        n_gain = sum(after[k]["n_root_shapes"] for k in flipped)
        print(f"  合计新增 {n_gain} 条折线，例: {flipped[0]} "
              f"0 → {after[flipped[0]]['n_root_shapes']} 条")

    only_a = sorted(set(before) - set(after))
    only_b = sorted(set(after) - set(before))
    if only_a:
        n_bad += 1
        print(f"❌ 只在转换前里有 {len(only_a)} 组（转换后不见了）:")
        for k in only_a:
            print(f"     {k}  ← {before[k].get('rel')}")
    if only_b:
        n_bad += 1
        print(f"❌ 只在转换后里有 {len(only_b)} 组:")
        for k in only_b:
            print(f"     {k}  ← {after[k].get('rel')}")

    for k in sorted(set(before) & set(after)):
        a, b = before[k], after[k]
        diffs = [f for f in set(a) | set(b) if f not in SKIP and a.get(f) != b.get(f)]
        if not diffs:
            continue
        n_bad += 1
        print(f"❌ {k}  ← {a.get('rel')}")
        for f in sorted(diffs):
            print(f"     {f}:  {a.get(f)!r}  →  {b.get(f)!r}")
        if "coords_sha256" in diffs:
            loc = first_diff(a.get("polylines") or [], b.get("polylines") or [])
            if loc:
                print(f"     第一处坐标不同: 第 {loc[0]} 条折线 / 第 {loc[1]} 点:"
                      f"  {loc[2]!r}  →  {loc[3]!r}")
    return n_bad


def check_sources(before, after, expect_json):
    """★ 防空转的关键一条：本该转成 json、却仍走 rsml 的样本，说明这一步没生效。

    没有这条，上面那套「逐位相同」在转换完全没跑的情况下也会通过 —— 因为
    加载器退回 rsml 时，新路径就是旧路径，比对双方根本是同一份数据。

    只盯「**本来有根要转**」的样本：像 plant_S003-3 那种 rsml 里就是 0 条根的
    合法负样本，没东西可插，json 里当然也没有 root 形状 —— 它不是漏转，是没得转。
    判据用「转换前的折线数 > 0」，而不是「json 里有没有 root」。
    """
    if not expect_json:
        return 0
    should = {k for k, e in before.items() if (e.get("n_polylines") or 0) > 0}
    stuck = sorted(k for k in should & set(after) if after[k].get("source") != "json")
    n_ok = len(should) - len(stuck)
    if not stuck:
        print(f"来源: 该转的 {len(should)} 组全部转成了 json ✅"
              f"（另有 {len(after) - len(should)} 组本来就无根可转）")
        return 0
    print(f"❌ 转换后有 {len(stuck)} 组本该转成 json、却仍从 rsml 读根 ——"
          f"这些样本的「一致」不能算数:")
    for k in stuck[:20]:
        print(f"     {k}  ← {after[k].get('rel')}")
    if len(stuck) > 20:
        print(f"     … 另有 {len(stuck) - 20} 组")
    print(f"   （其余 {n_ok} 组已转成 json）")
    return 1


def check_provenance(args):
    """直接问加载器：这张图实际用的是哪个来源。"""
    try:
        from common.dataset import load_annot          # noqa: PLC0415
    except ImportError:
        sys.exit("[跳过] 还没有 common.dataset.load_annot —— 新格式支持尚未落地。")
    bad, n_empty, n = [], [], 0
    for d in args.dir:
        root = Path(d)
        imgs = (sorted(root.rglob("*")) if args.recursive else sorted(root.iterdir()))
        for img in imgs:
            if not img.is_file() or img.suffix.lower() not in config.IMAGE_EXTS:
                continue
            n += 1
            ent = load_annot(root, img.stem, tuple(image_size(img)), verbose=False)
            if ent.source == "json":
                continue
            # 退回 rsml 但**一条根都没有** = 本来就没东西可转（合法的「这张图没有根」
            # 负样本，如 plant_S003-3），不是漏转。判据是「rsml 里有没有根」，
            # 不是「json 里有没有 root」——后者对这类样本永远为假。
            (n_empty if not ent.roots else bad).append(
                img.stem if not ent.roots else f"{img.stem}（{ent.source}，{len(ent.roots)} 条根）")
    print(f"来源断言: {n - len(bad) - len(n_empty)}/{n} 走 json"
          f"（另有 {len(n_empty)} 组本来就无根可转）")
    if bad:
        print(f"❌ {len(bad)} 组本该转成 json、却仍从 rsml 读根 ——"
              f"此时任何「指标逐位相同」都证明不了转换生效:")
        for s in bad[:20]:
            print(f"     {s}")
        return 1
    print("✅ 该转的全部走了 json 来源")
    return 0


def main():
    # 同 merge_annot：GBK 控制台打印 ❌/✅ 会 UnicodeEncodeError，允许替换字符
    if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parse_args()
    if not (args.capture or args.compare or args.provenance):
        sys.exit("要指定 --capture / --compare / --provenance 至少一个（用法见文件头）")

    rc = 0
    if args.provenance:
        rc = check_provenance(args) or rc

    if args.capture:
        man = collect(args)
        Path(args.capture).write_text(
            json.dumps(man, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8")
        src = {}
        for e in man.values():
            k = e.get("source") or "(读不出根系)"
            src[k] = src.get(k, 0) + 1
        print(f"已抓指纹 {len(man)} 组 → {args.capture}")
        print("  来源: " + " | ".join(f"{k} {v}" for k, v in sorted(src.items())))
        n_err = src.get("(读不出根系)", 0)
        if n_err:
            print(f"  ⚠️ {n_err} 组读不出根系（见指纹文件里的 error 字段）")

    if args.compare:
        before = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        after = collect(args)
        n_bad = compare(before, after)
        n_bad += check_sources(before, after, expect_json=(args.roots_from == "auto"))
        if n_bad:
            print(f"\n❌ {n_bad} 处问题")
            rc = 1
        else:
            print(f"\n✅ {len(before)} 组逐项相同"
                  f"（折线数 / 点数 / 坐标 / 掩码 / 尺寸 / 顶层键）")

    sys.exit(rc)


if __name__ == "__main__":
    main()
