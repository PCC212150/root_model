"""把标注工具的产出一键装配成 `datasets/root/{train,test}/{GT,scr}/`。

**分类口径**（2026-10-15 用户定，与训练侧的两条真值路一一对应）：
    有多边形 + 有掩码  ->  GT    真值：root 轮廓多边形 + 标注工具导出的像素级掩码
    只有折线（无掩码） ->  scr   伪标签：10px 折线栅格化（config.MASK_LINE_WIDTH）

后者走 `common/dataset.build_target_masks` 的「linestrip → 描线」那条路，前者走
「masks/<名>.png 替换 root 通道」那条路 —— 所以这个分类**不是标签好坏的分档，是
真值口径的分档**：分错了就是一张图一个尺子（config 里记过：5px→10px 尺子自己就让
Dice +0.12）。

**划分口径**：
    scr **全部进 train** —— 测试集只能有 GT（用户定的硬约束）。
    GT 按**单图**随机抽 --test-ratio 进 test/GT（默认 1/3），其余进 train/GT。
    ⚠️ 按单图抽 = 同株泄漏：同一植株的若干时点会分居两侧。项目平时的规矩是「整株
    进出」（common.naming.plant_key），这是**有意的例外**（与 2026-10-08 那次同口径）。
    工具会把跨侧的植株逐株点名并写进 split.txt —— 不点名的话，下一个人会把 test/GT
    上的分数当成泛化性能。

用法：
    python assemble_dataset.py --dir "C:\\Users\\21215\\Desktop\\数据集总表\\root\\新" --dry-run
    python assemble_dataset.py --dir "C:\\Users\\21215\\Desktop\\数据集总表\\root\\新" \\
        --out "D:\\python projects\\Deep_learning_model_for_sugarcane\\root_model\\datasets\\root"

只依赖标准库 + common/labelme（不 import torch），所以没装 torch 的本地机器也能跑。
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common import naming  # noqa: E402
from common.labelme import parse_other  # noqa: E402
from common.naming import plant_key  # noqa: E402

GT, SCR = "GT", "scr"
KINDS = (GT, SCR)
SUBSETS = ("train", "test")

# 像素级掩码的子目录。**必须与 common/dataset.py 的 MASK_SUBDIR、标注工具的
# MASK_SUBDIR 三处一致** —— 标注工具往这儿写、训练代码从这儿读，这里决定搬不搬。
MASK_SUBDIR = "masks"

# 顺手的垃圾文件，不参与装配
JUNK_FILES = {".ds_store", "thumbs.db", "desktop.ini", ".gitkeep"}

DEFAULT_TEST_RATIO = 1 / 3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="把标注工具产出装配成 datasets/root/{train,test}/{GT,scr}/"
                    "（GT=有多边形+掩码，scr=只有折线；test 只收 GT）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            '  python assemble_dataset.py --dir "C:\\...\\数据集总表\\root\\新" --dry-run\n'
            '  python assemble_dataset.py --dir "C:\\...\\数据集总表\\root\\新"\n'
            "\n"
            "输出目录非空时会拒绝执行（避免新旧混在一起）：\n"
            "重装数据集的标准动作是**先把 datasets\\root 改名备份，再跑本工具**。\n"
        ),
    )
    parser.add_argument("--dir", required=True, help="源文件夹（平铺：<名>.jpg/.json + masks/）")
    parser.add_argument("--out", default=None,
                        help=f"输出目录，默认 {config.ROOT_DATA_DIR}")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO,
                        help=f"GT 里抽多大比例进 test（**按单图抽**），默认 {DEFAULT_TEST_RATIO:.4g}")
    parser.add_argument("--seed", type=int, default=config.SEED,
                        help=f"随机种子，默认 {config.SEED}")
    parser.add_argument("--move", action="store_true", help="移动文件（默认复制，源数据保留）")
    parser.add_argument("--force", action="store_true",
                        help="反常组（多边形与掩码对不上、缺图/缺 json）不再拦，按规则降级处理")
    parser.add_argument("--dry-run", action="store_true", help="只预览，不写任何文件")
    return parser.parse_args(argv)


def transfer(path, target, move: bool):
    """搬一个文件：move=True 移动，否则复制（默认，源数据保留）。"""
    if move:
        shutil.move(str(path), str(target))
    else:
        shutil.copy2(str(path), str(target))


def collect_groups(src):
    """按文件名主干分组：一组 = 同名的一批文件（图片 + 同名 json）。"""
    groups, junk = {}, []
    for path in sorted(p for p in src.iterdir() if p.is_file()):
        if path.name.lower() in JUNK_FILES:
            junk.append(path)
            continue
        groups.setdefault(path.stem, []).append(path)
    return groups, junk


def inspect_group(src, stem, files):
    """判定一组属于 GT 还是 scr。返回 dict（kind 为 None 表示反常，issue 说明原因）。

    判据只认 json 里的 **root 多边形**（`shape_type=polygon`）与 **masks/<名>.png**
    这两样，与训练侧 `root_polygons` 标志、`load_root_mask_png` 的读取条件完全对齐：
    多边形决定「有没有轮廓真值」，掩码决定「root 通道最后有没有被逐像素真值替换」。
    """
    img = next((f for f in files if f.suffix.lower() in config.IMAGE_EXTS), None)
    json_path = next((f for f in files if f.suffix.lower() == ".json"), None)
    mask = src / MASK_SUBDIR / f"{stem}.png"

    n_poly = n_line = 0
    err = None
    if json_path is not None:
        try:
            lab = parse_other(json_path, verbose=False)
            n_poly = sum(1 for flag in lab.root_polygons if flag)
            n_line = len(lab.root_polygons) - n_poly
        except Exception as exc:               # 坏 json：必须报出来，不能当普通 scr
            err = f"json 解析失败：{type(exc).__name__}: {exc}"

    row = dict(stem=stem, img=img, json=json_path, mask=mask, has_mask=mask.exists(),
               n_poly=n_poly, n_line=n_line, err=err, kind=None, issue=None)
    if err:
        row["issue"] = err
    elif img is None:
        row["issue"] = "只有标注没有图片"
    elif json_path is None:
        row["issue"] = "只有图片没有 json（没有根系折线，做不了 scr）"
    elif n_poly and row["has_mask"]:
        row["kind"] = GT
    elif not n_poly and not row["has_mask"]:
        row["kind"] = SCR
    elif n_poly and not row["has_mask"]:
        row["issue"] = (f"有 {n_poly} 个根轮廓多边形但没有 masks/{stem}.png"
                        f"（GT 要求两样都有；当 scr 又会让这张图走填充口径，与别的 scr 两把尺子）")
    else:
        row["issue"] = (f"有 masks/{stem}.png 但 json 里没有根多边形（只剩 {n_line} 条折线）"
                        f"（多边形的导出与掩码对不上——掩码可能是上一次标注的残留）")
    return row


def split_gt(gt_names, ratio, seed):
    """GT 按**单图**随机抽 test：先排序再按种子打乱，结果只取决于数据与种子。"""
    names = sorted(gt_names)
    random.Random(seed).shuffle(names)
    n_test = min(int(len(names) * ratio + 0.5), len(names))
    return sorted(names[:n_test]), sorted(names[n_test:]), n_test


def planted_side_map(assign):
    """植株 -> {(份, 类): [组名]}，只保留**跨 train/test** 的植株（同株泄漏）。

    判据是「train 与 test **同时**出现」，不是「出现在多份里」的任意组合：
    同一株既在 train/GT 又在 train/scr 是正常的（两边的时点都在训练侧），
    只有一份在 test、另一份在 train 才叫泄漏。
    """
    where = {}
    for subset in SUBSETS:
        for kind in KINDS:
            for name in assign[subset][kind]:
                where.setdefault(plant_key(name), {}).setdefault((subset, kind), []).append(name)
    return {p: d for p, d in where.items()
            if {subset for subset, _kind in d} == set(SUBSETS)}


def write_record(out_dir, src, rows, assign, ratio, seed, problems):
    """写 split.txt：分类口径、划分比例、种子、每组归属、跨侧植株逐株点名。"""
    n_gt = sum(1 for r in rows if r["kind"] == GT)
    n_scr = sum(1 for r in rows if r["kind"] == SCR)
    leaks = planted_side_map(assign)

    lines = [
        f"# 数据集装配记录（{naming.timestamp()} 重装：用 tool/assemble_dataset 装配）",
        f"源文件夹：{src}",
        f"输出目录：{out_dir}",
        f"分类：有多边形 + 掩码 -> GT（{n_gt} 组）；只有折线 -> scr（{n_scr} 组）",
        f"划分：scr 全部进 train（**测试集只能有 GT**）；GT 按**单图**随机抽 {ratio:.4g} 进 test",
        f"      随机种子：{seed}    时间：{naming.timestamp()}",
        "# ⚠️ 按单图抽 = **同株泄漏**：下面点名的植株，其多个时点分居 train/test 两侧。",
        "#    项目平时的规矩是「整株进出」（common.naming.plant_key），这次是**有意的例外**",
        "#    （与 2026-10-08 那次同口径）。代价：test/GT 上的分数偏乐观，只能当",
        "#    「在准确尺子上看趋势」，**不能当泛化性能**。",
        "",
    ]
    if leaks:
        lines.append(f"## 跨侧植株（同株泄漏，{len(leaks)} 株）")
        for plant in sorted(leaks):
            detail = "、".join(f"{subset}/{kind} {len(names)} 组"
                              for (subset, kind), names in sorted(leaks[plant].items()))
            lines.append(f"{plant}：{detail}")
        lines.append("")
    else:
        lines += ["## 跨侧植株（同株泄漏）", "（无）", ""]

    if problems:
        lines.append(f"## 反常组（{len(problems)} 组，--force 放行）")
        for r in problems:
            lines.append(f"{r['stem']}：{r['issue']}")
        lines.append("")

    notes = {GT: "多边形 + 掩码（真值）", SCR: "折线膨胀（伪标签）"}
    for subset in SUBSETS:
        for kind in KINDS:
            names = assign[subset][kind]
            if not names:
                continue
            n_plants = len({plant_key(n) for n in names})
            lines.append(f"## {subset}/{kind}（{len(names)} 组 / {n_plants} 株，{notes[kind]}）")
            lines.append("、".join(names))
            lines.append("")
    (out_dir / "split.txt").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    src = Path(args.dir).expanduser()
    if not src.is_dir():
        print(f"[错误] 文件夹不存在：{src}")
        return 1
    if not 0 < args.test_ratio < 1:
        print(f"[错误] --test-ratio 要在 0~1 之间，当前 {args.test_ratio:g}")
        return 1
    out_dir = Path(args.out).expanduser() if args.out else Path(config.ROOT_DATA_DIR)

    groups, junk = collect_groups(src)
    if not groups:
        print(f"[错误] 文件夹里没有文件：{src}")
        return 1
    if junk:
        print(f"[提示] 忽略 {len(junk)} 个系统文件：{'、'.join(f.name for f in junk)}")

    rows = [inspect_group(src, stem, groups[stem]) for stem in sorted(groups)]

    # ---------- 反常组：默认拦下来 ----------
    problems = [r for r in rows if r["kind"] is None]
    if problems:
        title = "[错误] 有反常组，没有装配任何文件：" if not args.force else "[警告] 反常组（--force 放行）："
        print(title)
        for r in problems:
            print(f"  {r['stem']}：{r['issue']}")
        if not args.force:
            print("  处置：去源里改到两边一致（补导掩码 / 删掉多余掩码 / 补 json），"
                  "或加 --force 按规则降级"
                  "（有多边形无掩码 -> 当 GT 但不拷掩码；有掩码无多边形 -> 当 scr 且**不拷掩码**）。")
            return 1
    good = [r for r in rows if r["kind"] is not None]

    # 源里的孤儿掩码（没有同名 json/图片）：不搬，但要报出来
    stems = set(groups)
    orphans = sorted(p.name for p in (src / MASK_SUBDIR).glob("*.png")
                     if p.stem not in stems) if (src / MASK_SUBDIR).is_dir() else []
    if orphans:
        print(f"[提示] {len(orphans)} 个掩码在源里没有同名图/json，未装配：{'、'.join(orphans)}")

    gt_names = [r["stem"] for r in good if r["kind"] == GT]
    scr_names = [r["stem"] for r in good if r["kind"] == SCR]
    test_gt, train_gt, n_test = split_gt(gt_names, args.test_ratio, args.seed)

    assign = {"train": {GT: train_gt, SCR: sorted(scr_names)},
              "test": {GT: test_gt, SCR: []}}
    leaks = planted_side_map(assign)

    # ---------- 预览 ----------
    print(f"源文件夹：{src}")
    print(f"输出目录：{out_dir}")
    print(f"共 {len(rows)} 组：GT {len(gt_names)} 组（多边形 + 掩码）"
          f"，scr {len(scr_names)} 组（只有折线）"
          + (f"，反常 {len(problems)} 组" if problems else ""))
    print(f"划分：scr 全部进 train（测试集只能有 GT）；"
          f"GT 按单图抽 {args.test_ratio:.4g} -> test {n_test} 组    随机种子：{args.seed}")
    for subset in SUBSETS:
        for kind in KINDS:
            names = assign[subset][kind]
            if not names:
                continue
            n_plant = len({plant_key(n) for n in names})
            head = "、".join(names[:6]) + ("…" if len(names) > 6 else "")
            print(f"  {subset}/{kind}：{len(names)} 组 / {n_plant} 株   {head}")
    if leaks:
        # 提示语里的符号只用能进 GBK 的（Windows 控制台默认 GBK，emoji 会 UnicodeEncodeError）
        print(f"[警告] 跨侧植株（同株泄漏）{len(leaks)} 株：{'、'.join(sorted(leaks))}")
        print("       按单图抽的必然后果（用户 2026-10-15 明确要这个口径）；已写进 split.txt。")

    if args.dry_run:
        print("\n（预览模式，没有写任何文件）")
        return 0

    # ---------- 落盘 ----------
    if out_dir.exists():
        if any(out_dir.iterdir()):
            print(f"[错误] 输出目录已存在且不为空：{out_dir}\n"
                  f"       重装的标准动作：先把 datasets\\root 改名备份（如 "
                  f"root_旧装配备份_{naming.timestamp('%Y-%m-%d')}），再跑本工具。")
            return 1
        print(f"[提示] 输出目录已存在但是空的，直接用它：{out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    n_files = n_masks = 0
    by_stem = {r["stem"]: r for r in good}
    for subset in SUBSETS:
        for kind in KINDS:
            names = assign[subset][kind]
            if not names:
                continue
            dst = out_dir / subset / kind
            dst.mkdir(parents=True, exist_ok=True)
            for name in names:
                row = by_stem[name]
                for path in groups[name]:
                    transfer(path, dst / path.name, args.move)
                    n_files += 1
                # 掩码**只在真 GT 上搬**：反常组（有掩码无多边形）按 scr 处理时不搬 ——
                # 搬了它就会走掩码口径，一份 scr 里一半图一个尺子（见 config 的记）。
                if kind == GT and row["has_mask"]:
                    mt = dst / MASK_SUBDIR / f"{name}.png"
                    mt.parent.mkdir(parents=True, exist_ok=True)
                    transfer(row["mask"], mt, args.move)
                    n_masks += 1

    write_record(out_dir, src, rows, assign, args.test_ratio, args.seed, problems)

    action = "移动" if args.move else "复制"
    print(f"\n完成：{action} {n_files} 个文件 + {n_masks} 个掩码到 {out_dir}")
    print(f"装配记录：{out_dir / 'split.txt'}")
    print("提示：datasets/ 不入库，服务器上那份要另外同步（见 config.py 的划分记录）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
