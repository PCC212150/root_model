"""为实现 **C/P 对照**，把无法成对的图片从两边的文件夹里剔除。

C（`root_C###-#_<日期>CK.jpg`）与 P（`root_P###-#_<日期>PEG.jpg`）是两个处理，
做对照实验要求**同一个「编号-重复-日期」两边各有一张**。实际数据里有三种坑：

0. **前缀不对**：文件名主干的前缀换过（`root_` / `plant_`），正则写死一个的话
   另一套会被整批判成「异常命名」，**把全部文件搬进隔离目录**。所以前缀默认
   从数据里**自动探测**（取出现最多的那个），也可以用 `--prefix` 显式指定。
   另外还有一道防呆：单次要剔除的文件超过总数一半时直接罢工，除非 `--force`。
1. **异常命名**：不符合标准命名的文件（如 `root_P137-1 (2)_20250108PEG.jpg`）——
   归属本身就不确定，留着会污染对照；
2. **单边缺某天**：同一天只有 C 有、P 没有（或反过来），凑不成一对；
3. 上面两种都会让两边的数目对不上。

规则（**纯按文件名**，不改名、不按图像内容猜归属）：

    异常命名的           -> 剔除
    两边都有的日期        -> 保留
    只有一边有的日期      -> **两边都剔除**（保证严格一一对应）
    非图片文件           -> **不碰**（.rsml/.json 等标注原地保留，不在裁剪范围内）

剔除的文件**移到隔离目录**（默认 `<dir>\\_剔除_CP对照\\<C|P>\\`）而不是直接删，
配合日志可以随时搬回来。确认无误后自行删掉隔离目录即可。

用法：
    python pair_cp.py --dir "E:\\baiduwangpan\\DownLoad" --dry-run   # 先预览
    python pair_cp.py --dir "E:\\baiduwangpan\\DownLoad"             # 正式执行
    python pair_cp.py --dir "..." --quarantine "D:\\回收"             # 指定隔离目录
    python pair_cp.py --dir "E:\\baiduwangpan\\DownLoad" \\
        --sides "C_plant,P_plant" --prefix plant                      # plant_ 命名的一批

执行后在 `--dir` 下生成 `CP对照_剔除日志.csv`（UTF-8 BOM，Excel 直接打开），
逐条记录「哪一侧 / 哪个文件 / 编号-重复 / 日期 / 剔除原因 / 搬到哪了」。
"""
import argparse
import csv
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

# 标准命名：<前缀>_<C|P><编号>-<重复>_<8位日期><批次后缀>.jpg（前缀如 root / plant）
STRICT = re.compile(r"^root_([CP])(\d+)-(\d+)_(\d{8})([A-Za-z]*)\.jpg$")

# 探测前缀用：不限定前缀，只看整体形状对不对得上
PREFIX_PROBE = re.compile(r"^([A-Za-z]+)_([CP])(\d+)-(\d+)_(\d{8})([A-Za-z]*)\.jpg$")

# 防呆阈值：一次要剔除的比例超过它就先罢工（多半是前缀/文件夹选错了）
SUSPICIOUS_RATIO = 0.5

# 只有这些后缀才算「图片」，才进裁剪流程；其余（.rsml / .json 标注等）原地不动。
# 与 config.py 的 IMAGE_EXTS 一致，只是本工具独立于项目，不 import config。
# 之前不区分：旧 C/ 里 4 个 root_C001-1_*.rsml 会被当成「异常命名」扫进隔离目录。
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DEFAULT_SIDES = "C,P"
QUARANTINE_NAME = "_剔除_CP对照"
LOG_NAME = "CP对照_剔除日志.csv"

REASON_BAD = "异常命名"
REASON_ONLY = "{side} 独有（对侧缺该日）"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="剔除 C/P 无法成对的图片（异常命名的 + 单边独有的）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("示例：\n"
                '  python pair_cp.py --dir "E:\\baiduwangpan\\DownLoad" --dry-run\n'
                '  python pair_cp.py --dir "E:\\baiduwangpan\\DownLoad"\n'),
    )
    p.add_argument("--dir", required=True, help="含 C 与 P 两个子文件夹的父目录")
    p.add_argument("--sides", default=DEFAULT_SIDES,
                   help=f"两侧的文件夹名，逗号分隔（默认 {DEFAULT_SIDES}）")
    p.add_argument("--quarantine", default=None,
                   help=f"剔除文件的去处（默认 <dir>\\{QUARANTINE_NAME}）")
    p.add_argument("--prefix", default=None,
                   help="文件名前缀（如 root / plant）。默认自动探测；"
                        "探测不出来或想强制指定时才写")
    p.add_argument("--log", default=None,
                   help=f"日志路径（默认 <dir>\\{LOG_NAME}）。"
                        f"重跑时指定别的名字，免得盖掉上一次的记录")
    p.add_argument("--force", action="store_true",
                   help=f"剔除比例超过 {SUSPICIOUS_RATIO:.0%} 时也照做"
                        f"（默认罢工，多半是前缀或文件夹选错了）")
    p.add_argument("--dry-run", action="store_true",
                   help="只报告要剔除哪些，不移动任何文件、不写日志")
    return p.parse_args(argv)


def detect_prefix(dirs):
    """从数据里探测文件名前缀：取出现最多的那个（如 root / plant）。

    两套命名并存时以多数为准；一个都没匹配上就返回 None，交给调用方报错。
    """
    counts = Counter()
    for d in dirs:
        for f in d.iterdir():
            if f.is_file():
                m = PREFIX_PROBE.match(f.name)
                if m:
                    counts[m.group(1)] += 1
    return counts.most_common(1)[0][0] if counts else None


def build_pattern(prefix: str):
    """按前缀造严格正则，分组位置与 STRICT 一致：1=侧 2=编号 3=重复 4=日期 5=后缀。"""
    return re.compile(rf"^{re.escape(prefix)}_([CP])(\d+)-(\d+)_(\d{{8}})([A-Za-z]*)\.jpg$")


def side_letter(folder_name: str):
    """从文件夹名里认出这一侧的处理字母（C / P），认不出返回 None。

    以前文件夹就叫 `C` / `P`，名字即字母；后来改成 `C_plant` / `P_plant`，
    再拿文件夹名直接比就永远匹配不上，整批文件会被误判成「异常命名」。
    这里取文件夹名里出现的**第一个大写 C 或 P**（`C_plant` -> C，`plant_P` -> P），
    小写的 p（plant 的 p）不会被误认。
    """
    m = re.search(r"[CP]", folder_name)
    return m.group(0) if m else None


def parse_side(root: Path, kind: str, pattern):
    """返回 ({(编号,重复,日期): 路径}, [异常命名的图片], [跳过的非图片文件])。

    非图片文件（.rsml / .json 标注等）直接跳过：它们不参与 C/P 配对，
    也不该因为「不符合 jpg 命名」被当成异常图片搬走。
    """
    ok, bad, skipped = {}, [], []
    for f in sorted(root.iterdir()):
        if not f.is_file():
            continue
        if f.suffix.lower() not in IMAGE_EXTS:
            skipped.append(f)
            continue
        m = pattern.match(f.name)
        if m and m.group(1) == kind:
            key = (m.group(2), m.group(3), m.group(4))
            if key in ok:
                bad.append(f)               # 同一天重复出现，同样交给人看
            else:
                ok[key] = f
        else:
            bad.append(f)
    return ok, bad, skipped


def main(argv=None):
    args = parse_args(argv)
    base = Path(args.dir)
    if not base.is_dir():
        sys.exit(f"[错误] 文件夹不存在: {base}")
    sides = [s.strip() for s in args.sides.split(",") if s.strip()]
    if len(sides) != 2:
        sys.exit(f"[错误] --sides 需要正好两个文件夹名，当前: {args.sides}")

    dirs = {s: base / s for s in sides}
    for s, d in dirs.items():
        if not d.is_dir():
            sys.exit(f"[错误] 子文件夹不存在: {d}")

    prefix = args.prefix
    if prefix is None:
        prefix = detect_prefix(dirs.values())
        if prefix is None:
            sys.exit("[错误] 两个文件夹里都没有 `<前缀>_C###-#_<8位日期>xxx.jpg` 这种文件，"
                     "没法探测前缀；确认 --dir/--sides 对不对，或用 --prefix 显式指定")
        print(f"前缀：自动探测到 `{prefix}_`（可用 --prefix 覆盖）")
    else:
        print(f"前缀：`{prefix}_`（--prefix 指定）")
    pattern = build_pattern(prefix)

    data, letters = {}, {}
    for s in sides:
        letter = side_letter(s)
        if letter is None:
            sys.exit(f"[错误] 从文件夹名 `{s}` 里认不出 C/P 字母，"
                     f"请把文件夹名改成含大写 C 或 P 的（如 `C_plant`）")
        letters[s] = letter
        ok, bad, skipped = parse_side(dirs[s], letter, pattern)
        data[s] = (ok, bad, skipped)
        line = (f"{s}（{letter} 侧）: 标准命名 {len(ok)} 个 / 异常命名 {len(bad)} 个"
                f" / 共 {len(ok)+len(bad)} 个")
        if skipped:
            line += f"；另跳过 {len(skipped)} 个非图片文件（不动）"
        print(line)

    a, b = sides
    ka, kb = set(data[a][0]), set(data[b][0])
    keep = ka & kb
    only_a, only_b = ka - kb, kb - ka
    print(f"\n两边都有（可对照）: {len(keep)} 个「编号-重复-日期」")
    print(f"{a} 独有（{b} 缺该日）: {len(only_a)} 个")
    print(f"{b} 独有（{a} 缺该日）: {len(only_b)} 个")

    moves = []                                   # (路径, 侧, 原因, 编号-重复, 日期)
    for s in sides:
        for f in data[s][1]:
            moves.append((f, s, REASON_BAD, "", ""))
    for key in sorted(only_a):
        m = pattern.match(data[a][0][key].name)
        moves.append((data[a][0][key], a, REASON_ONLY.format(side=a),
                      f"{m.group(2)}-{m.group(3)}", m.group(4)))
    for key in sorted(only_b):
        m = pattern.match(data[b][0][key].name)
        moves.append((data[b][0][key], b, REASON_ONLY.format(side=b),
                      f"{m.group(2)}-{m.group(3)}", m.group(4)))

    # 防呆：正常数据只剔零星几个。要剔掉一大半，几乎一定是前缀探测错了 /
    # --sides 指到了含混数据的文件夹 —— 这种时候搬走就等于清空数据集，先停下。
    n_total = sum(len(data[s][0]) + len(data[s][1]) for s in sides)
    if n_total and len(moves) > n_total * SUSPICIOUS_RATIO and not args.force:
        print(f"\n[罢工] 要剔除 {len(moves)} / {n_total} 个（超过 {SUSPICIOUS_RATIO:.0%}），"
              f"不合常理，没有动任何文件。")
        if args.prefix:
            print(f"       前缀是你用 --prefix 指定的 `{prefix}_`，八成写错了；"
                  f"确认这批文件确实长这样再加 --force。")
        else:
            print(f"       自动探测到的前缀是 `{prefix}_`，若这批文件前缀确实不同，"
                  f"用 --prefix 指定；确认无误再加 --force。")
        return 1

    by_reason = defaultdict(list)
    for mv in moves:
        by_reason[mv[2]].append(mv)
    print("\n=== 将剔除 ===")
    for reason, items in by_reason.items():
        print(f"  [{reason}] {len(items)} 个")
        for f, s, _, _, _ in items[:5]:
            print(f"      {s}/{f.name}")
        if len(items) > 5:
            print(f"      … 其余 {len(items)-5} 个（详见日志）")
    for s in sides:
        # 从「总文件数」起算：异常命名的也要被搬走，跳过的非图片文件则原地留下
        n_after = (len(data[s][0]) + len(data[s][1]) + len(data[s][2])
                   - sum(1 for mv in moves if mv[1] == s))
        tail = f"（含跳过的 {len(data[s][2])} 个非图片文件）" if data[s][2] else ""
        print(f"  {s} 剩余 {n_after} 个{tail}")

    if args.dry_run:
        print("\n（预览模式，未移动任何文件，也没写日志）")
        return 0

    q = Path(args.quarantine) if args.quarantine else base / QUARANTINE_NAME
    for f, s, _, _, _ in moves:
        dst = q / s / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(dst))

    log = Path(args.log) if args.log else base / LOG_NAME
    with open(log, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write(f"# C/P 对照裁剪日志    父目录: {base}\n")
        fh.write(f"# 保留 {len(keep)} 组「编号-重复-日期」（两边各一张，严格一一对应）；"
                 f"剔除 {len(moves)} 个文件\n")
        fh.write(f"# 规则：异常命名先剔除；再保留两边都有的日期；只有一边有的，两边都剔除\n")
        fh.write(f"# 剔除的文件在隔离目录里（原样未改），位置见下表最后一列\n")
        fh.write(f"# 隔离目录: {q}\n")
        w = csv.writer(fh)
        w.writerow(["侧", "文件名", "编号-重复", "日期", "剔除原因", "隔离位置"])
        for f, s, reason, key, date in sorted(moves, key=lambda mv: (mv[2], mv[1], mv[0].name)):
            w.writerow([s, f.name, key, date, reason, str(q / s / f.name)])
    print(f"\n已剔除 {len(moves)} 个文件 -> {q}")
    print(f"日志: {log}")

    # 复核：裁完两边必须严格一一对应
    for s in sides:
        data[s] = parse_side(dirs[s], letters[s], pattern)
        if data[s][1]:
            print(f"  [警告] {s} 里仍有 {len(data[s][1])} 个异常文件: "
                  f"{[f.name for f in data[s][1][:3]]}")
    fa, fb = set(data[a][0]), set(data[b][0])
    print(f"\n复核: {a} {len(data[a][0])} 个 / {b} {len(data[b][0])} 个；"
          f"差集 {len(fa - fb)} / {len(fb - fa)}  "
          + ("**严格一一对应**" if fa == fb else "**仍未对齐！**"))
    skipped = [f for s in sides for f in data[s][2]]
    if skipped:
        print(f"跳过的 {len(skipped)} 个非图片文件原地未动（示例: "
              f"{'、'.join(f.name for f in skipped[:3])}…）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
