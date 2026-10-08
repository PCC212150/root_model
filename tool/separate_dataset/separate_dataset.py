"""把数据集文件夹按比例划分为 train / test / val 三份（验证集默认不分配）。

划分单位是「组」而不是单个文件：源文件夹里同名的一批文件算一组
（如 `plant_ S062-1_20251116ST.jpg` / `.json` / `.rsml`），整组进同一份，
不会把图片和标注拆散。

**输出默认是「平铺」的**（2026-09-30 起，与 common/dataset.py 的布局一致）：

    输出/train/   图片 + 标注（*.jpg / *.json / *.rsml）全在同一层
    （test / val 同构）

想回到 2026-09-17–09-29 之间的嵌套布局，加 `--nested`：

    输出/train/images/           图片
    输出/train/labels/roots/     *.rsml  根系标注
    输出/train/labels/other/     *.json  labelme 标注（茎/检查范围/根系折线）
    （test / val 同构）

两种布局 `common/dataset.py` **都能读**（自动识别），所以旧数据集不用急着搬。

用法：
    python separate_dataset.py --dir "C:\\Users\\21215\\Desktop\\数据集总表\\root" --dry-run
    python separate_dataset.py --dir "C:\\Users\\21215\\Desktop\\数据集总表\\root" \\
        --out "D:\\python projects\\Deep_learning_model_for_sugarcane\\datasets\\root"
    python separate_dataset.py --dir "D:\\数据\\20251116ST" --train 0.7 --test 0.2 --val 0.1

输出：默认写到 <源文件夹同级>/<源文件夹名>_split/（重名自动加 -1），
      并生成 split.txt 记录本次划分的比例、种子和每组归属，便于复现。
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
from common.dataset import plant_key  # noqa: E402

SUBSETS = ("train", "test", "val")

# 默认比例：训练 0.8 / 测试 0.2 / 验证 0（验证集默认不分配）
DEFAULT_TRAIN = 0.8
DEFAULT_TEST = 0.2

# 顺手的垃圾文件，不参与划分
JUNK_FILES = {".ds_store", "thumbs.db", "desktop.ini", ".gitkeep"}

# 像素级掩码的子目录（标注工具 create_datasets/root 写的，见 annotate_io.MASK_SUBDIR）。
# 它在子目录里，所以 collect_groups 看不见它 —— 得单独搬，否则划分完训练就退回
# 多边形口径，和没划分的数据成了两把尺子。
MASK_SUBDIR = "masks"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="把数据集文件夹按比例划分为 train / test / val（同名文件整组划分，配对不会拆散）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            '  python separate_dataset.py --dir "C:\\Users\\21215\\Desktop\\RootTracer_RSML\\20251116ST"\n'
            "      -> 0.8 训练 / 0.2 测试，不分配验证集\n"
            '  python separate_dataset.py --dir "D:\\数据\\20251116ST" --train 0.7 --test 0.2 --val 0.1\n'
            "\n"
            "只写 --test（或只写 --train）时，另一项按剩余自动推算；"
            "两者都写则必须和为 1。\n"
            "同一种子 + 同一份数据 = 同样的划分结果。"
        ),
    )
    parser.add_argument("--dir", required=True, help="源数据集文件夹")
    parser.add_argument("--out", default=None,
                        help="输出目录，默认 <源文件夹同级>/<源文件夹名>_split")
    parser.add_argument("--train", type=float, default=None, help=f"训练集比例，默认 {DEFAULT_TRAIN}")
    parser.add_argument("--test", type=float, default=None, help=f"测试集比例，默认 {DEFAULT_TEST}")
    parser.add_argument("--val", type=float, default=None, help="验证集比例，默认 0（不分配）")
    parser.add_argument("--seed", type=int, default=config.SEED, help=f"随机种子，默认 {config.SEED}")
    parser.add_argument("--move", action="store_true", help="移动文件（默认复制，源数据保留）")
    parser.add_argument("--flat", action="store_true",
                        help="平铺布局。**2026-09-30 起这就是默认**，此参数保留只为兼容旧命令")
    parser.add_argument("--nested", action="store_true",
                        help="改用旧的嵌套布局 images/ + labels/roots + labels/other")
    parser.add_argument("--by-file", action="store_true",
                        help="按单个文件组划分（旧行为）。**默认按植株划分** —— 同一植株的"
                             "多个时点整株进同一侧，避免同株泄漏（见 common/dataset.py 的 plant_key）")
    parser.add_argument("--dry-run", action="store_true", help="只预览划分结果，不写任何文件")
    return parser.parse_args(argv)


def resolve_ratios(args):
    """把 --train/--test/--val 解析成和为 1 的三份比例，返回 (ratios, 错误信息)。

    --train 与 --test 可以只写其中一个，另一个按剩余自动推算，例如：
        --test 0.2            -> train 0.8 / test 0.2 / val 0
        --train 0.7 --val 0.1 -> train 0.7 / test 0.2 / val 0.1
    两个都写时必须是显式自洽的（之和为 1），写错就直接报错而不是悄悄改数。
    """
    for name, value in (("train", args.train), ("test", args.test), ("val", args.val)):
        if value is not None and value < 0:
            return None, f"--{name} 不能为负数（当前 {value:g}）"

    val = 0.0 if args.val is None else args.val
    if args.train is not None and args.test is not None:
        train, test = args.train, args.test
        total = train + test + val
        if abs(total - 1.0) > 1e-6:
            return None, (f"比例之和必须为 1，当前为 {total:g}"
                          f"（train {train:g} + test {test:g} + val {val:g}）；"
                          f"只想指定其中一两项时，留空的那项会自动推算")
    elif args.train is not None:
        train, test = args.train, 1.0 - args.train - val
    elif args.test is not None:
        test, train = args.test, 1.0 - args.test - val
    else:
        test, train = DEFAULT_TEST, 1.0 - DEFAULT_TEST - val

    if train < -1e-9 or test < -1e-9:
        return None, (f"比例超出 1：train {train:g} / test {test:g} / val {val:g}，请调小 --val 或其他比例")
    return {"train": max(train, 0.0), "test": max(test, 0.0), "val": val}, None


def collect_groups(src):
    """按文件名主干分组：一组 = 同名的一批文件（图片 + 同名 rsml）。"""
    groups, junk = {}, []
    for path in sorted(p for p in src.iterdir() if p.is_file()):
        if path.name.lower() in JUNK_FILES:
            junk.append(path)
            continue
        groups.setdefault(path.stem, []).append(path)
    return groups, junk


def group_units(group_names, by_plant: bool):
    """把「文件组」归并成「划分单位」。

    by_plant=True（默认）：同一**植株**的所有时点归成一个单位，整株进同一侧 ——
        这是项目一直在守的规矩。`plant_ S062-1_20251116ST` 和 `..._20251126ST`
        是同一株的两个时点，分到训练/测试两边就是同株泄漏。
    by_plant=False：一个文件组就是一个单位（旧行为）。单时点的数据集两者等价。

    返回 {单位名: [文件组名]}。
    """
    units = {}
    for n in group_names:
        key = plant_key(n) if by_plant else n
        units.setdefault(key, []).append(n)
    return units


def dest_subdir(path: Path) -> str:
    """某个文件在 split 目录里该放哪个子目录（项目布局，见 config.py）。

    图片 -> images/ ；.rsml -> labels/roots/ ；.json -> labels/other/ ；
    认不出来的（readme、txt 之类）放 split 根目录，不硬塞进标注文件夹。
    """
    ext = path.suffix.lower()
    if ext in config.IMAGE_EXTS:
        return config.IMAGES_SUBDIR
    if ext == ".rsml":
        return config.ROOTS_LABEL_SUBDIR
    if ext == ".json":
        return config.OTHER_LABEL_SUBDIR
    return ""


def group_kind(files):
    """判断一组的配对情况。

    'pair'      图片 + json（**新格式**：根系折线也在这个 json 里，不需要 .rsml）
    'pair_rsml' 图片 + 只有 .rsml（**旧格式**，还没跑过 tool/merge_annot）
    'no_annot'  只有图片，没有任何标注
    'no_image'  只有标注，没有图片
    'other'     两者都没有

    2026-09-30 改：原来只认 `.rsml`，于是「图片 + json 无 rsml」被判成**「缺标注」**——
    新格式的数据走这个工具会被当成残缺数据。现在 json 与 rsml **有一个就算配对**。
    """
    has_img = any(f.suffix.lower() in config.IMAGE_EXTS for f in files)
    has_json = any(f.suffix.lower() == ".json" for f in files)
    has_rsml = any(f.suffix.lower() == ".rsml" for f in files)
    if not has_img:
        return "no_image" if (has_json or has_rsml) else "other"
    if has_json:
        return "pair"
    return "pair_rsml" if has_rsml else "no_annot"


def split_counts(n, ratios):
    """把 n 组按比例切成三份：四舍五入后余数留给 train，保证三份之和恰为 n。"""
    n_val = min(int(n * ratios["val"] + 0.5), n)
    n_test = min(int(n * ratios["test"] + 0.5), n - n_val)
    return {"train": n - n_val - n_test, "test": n_test, "val": n_val}


def write_record(out_dir, src, ratios, seed, assign):
    """在输出目录写 split.txt，记录比例/种子/每组归属，便于复现和核对。"""
    lines = [
        "# 数据集划分记录",
        f"源文件夹：{src}",
        f"输出目录：{out_dir}",
        f"比例：train {ratios['train']:g} / test {ratios['test']:g} / val {ratios['val']:g}"
        f"    随机种子：{seed}    时间：{naming.timestamp()}",
        "",
    ]
    for name in SUBSETS:
        names = assign[name]
        lines.append(f"## {name}（{len(names)} 组）")
        lines.append("、".join(names) if names else "（空）")
        lines.append("")
    (out_dir / "split.txt").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    src = Path(args.dir).expanduser()
    if not src.is_dir():
        print(f"[错误] 文件夹不存在：{src}")
        return 1

    ratios, err = resolve_ratios(args)
    if err:
        print(f"[错误] {err}")
        return 1

    groups, junk = collect_groups(src)
    n_json = sum(1 for fs in groups.values()
                 for f in fs if f.suffix.lower() == ".json")
    if not groups:
        print(f"[错误] 文件夹里没有文件：{src}")
        return 1
    if junk:
        print(f"[提示] 忽略 {len(junk)} 个系统文件：{'、'.join(f.name for f in junk)}")

    # 侧栏提示：子文件夹不参与划分（与 common/dataset.py 一致，只认当前层）
    subdirs = [d.name for d in src.iterdir() if d.is_dir()]
    if subdirs:
        print(f"[提示] 文件夹内有 {len(subdirs)} 个子文件夹，本工具只划分当前层的文件，子文件夹不动")

    kinds = {name: 0 for name in ("pair", "pair_rsml", "no_annot", "no_image", "other")}
    for files in groups.values():
        kinds[group_kind(files)] += 1
    n_files = sum(len(f) for f in groups.values())

    # 划分单位：默认按植株（同一植株的所有时点整株进同一侧），--by-file 则一组一个
    units = group_units(sorted(groups), by_plant=not args.by_file)
    counts = split_counts(len(units), ratios)
    unit_names = sorted(units)
    random.Random(args.seed).shuffle(unit_names)  # 先排序再打乱：结果只取决于数据与种子
    assign_units = {
        "train": unit_names[:counts["train"]],
        "test": unit_names[counts["train"]:counts["train"] + counts["test"]],
        "val": unit_names[counts["train"] + counts["test"]:],
    }
    # 展开回「文件组名」，后面的落盘逻辑不用改
    assign = {s: [g for u in assign_units[s] for g in units[u]] for s in SUBSETS}

    # ---------- 划分信息 ----------
    print(f"源文件夹：{src}")
    print(f"共 {len(groups)} 组 / {n_files} 个文件"
          f"（新格式 图片+json {kinds['pair']} 组，旧格式 图片+rsml {kinds['pair_rsml']} 组，"
          f"缺标注 {kinds['no_annot']} 组，缺图片 {kinds['no_image']} 组，"
          f"其他 {kinds['other']} 组）")
    if args.by_file:
        print("划分单位：单个文件组（--by-file）")
    else:
        print(f"划分单位：植株 —— {len(units)} 株"
              + (f"（{len(groups)} 组归并而来，同株的多个时点整株进同一侧）"
                 if len(units) != len(groups) else ""))
    if kinds["no_annot"]:
        print(f"[提示] {kinds['no_annot']} 组图片没有任何标注（既无 .json 也无 .rsml），"
              f"检查源文件夹是否漏拷 —— 训练要求图片与同名标注同目录")
    if kinds["pair_rsml"]:
        print(f"[提示] {kinds['pair_rsml']} 组还是**旧格式**（根系在 .rsml 里）。可以照常划分，"
              f"但建议先跑 tool\\merge_annot 把根系并进 json，让全库统一格式")
    # json 是茎/检查范围两个通道的标注，**可以缺**：缺的图只训练根系通道。
    # 但缺太多会让茎/检查范围两个通道学不好 —— 它们直接决定根系统计的准确性
    # （统计范围靠检查范围框限定），所以这里明确报出覆盖率。
    if n_json < len(groups):
        print(f"[提示] labelme json {n_json}/{len(groups)} 组"
              f"（缺 {len(groups) - n_json} 组）—— 缺 json 的图只训练根系通道，"
              f"茎/检查范围两通道不参与；这两路太少会影响根系统计的准确性")
    print(f"比例：train {ratios['train']:g} / test {ratios['test']:g} / val {ratios['val']:g}"
          f"    随机种子：{args.seed}")
    for name in SUBSETS:
        ratio = ratios[name]
        n_g = len(assign[name])
        n_f = sum(len(groups[g]) for g in assign[name])
        if ratio > 0 and counts[name] == 0:
            print(f"[提示] {name} 比例 {ratio:g} 但数据太少（共 {len(units)} 个划分单位），"
                  f"实际分到 0")
        if ratio == 0:
            print(f"  {name}：不分配（比例 0）")
        else:
            unit_word = "组" if args.by_file else "株"
            print(f"  {name}：{counts[name]} {unit_word} / {n_g} 组 / {n_f} 个文件"
                  f"  {'、'.join(assign[name]) if assign[name] else '空'}")

    if args.dry_run:
        print("\n（预览模式，没有写任何文件）")
        return 0

    # ---------- 落盘 ----------
    if args.out:
        out_dir = Path(args.out).expanduser()
        if out_dir.exists():
            # 空目录直接复用：重导数据集时常见做法就是「先把 datasets/root 清空再重跑」。
            # 只有非空才拦 —— 那才是真会跟旧结果混在一起的情况。
            if any(out_dir.iterdir()):
                print(f"[错误] 输出目录已存在且不为空：{out_dir}"
                      f"（换个 --out，或先清空它，避免与旧结果混在一起）")
                return 1
            print(f"[提示] 输出目录已存在但是空的，直接用它：{out_dir}")
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = naming.create_unique_dir(src.parent, f"{src.name}_split")

    for name in SUBSETS:
        if not assign[name]:
            continue  # 不分配的子集不建空文件夹
        dst = out_dir / name
        dst.mkdir(parents=True, exist_ok=True)
        n_mask = 0
        for gname in assign[name]:
            for path in groups[gname]:
                sub = dest_subdir(path) if args.nested else ""
                target = (dst / sub / path.name) if sub else (dst / path.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                if args.move:
                    shutil.move(str(path), str(target))
                else:
                    shutil.copy2(str(path), str(target))
            # 像素级掩码（标注工具写的 `masks/<名>.png`）**每个组搬一次**：
            # 它在子目录里，按"组"是分不到它的（collect_groups 只看顶层文件），
            # 不搬的话划分完训练就退回多边形口径了 —— 两批数据两个尺子。
            # **放在内层循环外面**：一个组有图 + json 两个文件，放里面会搬两遍
            # （结果一样，但计数翻倍）。
            mp = src / MASK_SUBDIR / f"{Path(gname).stem}.png"
            if mp.exists():
                mt = dst / MASK_SUBDIR / mp.name
                mt.parent.mkdir(parents=True, exist_ok=True)
                if args.move:
                    shutil.move(str(mp), str(mt))
                else:
                    shutil.copy2(str(mp), str(mt))
                n_mask += 1
        if n_mask:
            print(f"  {name}：另搬了 {n_mask} 个像素级掩码（{MASK_SUBDIR}/）")
    write_record(out_dir, src, ratios, args.seed, assign)

    action = "移动" if args.move else "复制"
    print(f"\n完成：{action} {sum(len(groups[n]) for name in SUBSETS for n in assign[name])} 个文件到 {out_dir}")
    print(f"划分记录：{out_dir / 'split.txt'}")
    layout = "嵌套（--nested）images/ + labels/roots + labels/other" if args.nested \
        else "平铺（图片与标注同层）"
    print(f"布局：{layout}")
    if out_dir == config.ROOT_DATA_DIR:
        # 直接落在 datasets/root 里，已经是训练能直接读的布局，不用再拷
        print("提示：输出就是 datasets/root 本身，训练/测试脚本可以直接跑，"
              "不需要再拷文件")
    elif not args.move:
        print(f"提示：源文件夹未改动；要把数据接入训练，把 {out_dir}\\train、{out_dir}\\test "
              f"放到 datasets\\root\\ 下（见 config.py 的 TRAIN_DATA_DIR / TEST_DATA_DIR）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
