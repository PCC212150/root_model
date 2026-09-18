"""把推理结果 CSV 画成折线图：**一个编号两张图**（总根长 / 总根系面积）。

文件名解析（2026-09-17 起拆成三段）：

    root_C001-1_20241229CK.jpg
         └方式┘ └编号┘ └重复┘ └─日期─┘└后缀┘
          C     001     1     20241229  CK

- **处理方式**（kind）：C / P / S …（一个字母或几个字母，如 PEG 处理记作 P）
- **编号**（num）：001、002 …
- **重复次序**（rep）：1~4
- **日期**：8 位，横坐标按真实时间间隔排布

两张图：
- `{方式}{编号}_根长.png`   —— 纵坐标 = 总根长 (px)
- `{方式}{编号}_根面积.png` —— 纵坐标 = 总根系面积 (px²)

CSV 里**同时有 C 和 P 时，每种处理方式各画一套**（5 个编号 × 2 种 = 20 张）。
以前只画排序第一种（`kinds[0]`），P 的数据会被静默丢掉。

**对比模式 `--compare C,P`**：把**同一个编号**下两种处理方式画进同一张图，
每张图 8 条线（`C001-1..4` + `P001-1..4`），同样出根长/根面积两份。

用法（在项目根目录下运行）：
    python tool\\analysis_datas\\analysis_datas.py --csv "D:\\merged.csv"
    python tool\\analysis_datas\\analysis_datas.py                          # 用 analysis_datas/ 下所有 csv
    python tool\\analysis_datas\\analysis_datas.py --csv x.csv --compare C,P
    python tool\\analysis_datas\\analysis_datas.py --csv x.csv --compare C,S --ids 001,002,003

输出：analysis_datas/{csv文件名}/，内含两张一套的折线图 + `_总览.png` + `_汇总.csv`
      （+ 原 CSV 里有带空格的图片名时，另存一份 `{原名}_无空格.csv`）。
"""
import argparse
import csv
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates          # noqa: E402
import matplotlib.pyplot as plt            # noqa: E402
from matplotlib.lines import Line2D        # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from common import naming  # noqa: E402

# 图片名：先剥掉 plant_ / root_ 前缀（项目里两种写法并存，还带不带空格两种），
# 再按 方式+编号-重复_日期 解析。**先剥前缀**是关键：否则 `plant_S062-1` 会被
# 解析成 kind=`plant_S`、num=`062`，处理方式就错了。
PREFIXES = ("plant_", "root_")
NAME_PAT = re.compile(
    r"^(?P<kind>[A-Za-z]+?)(?P<num>\d+)-(?P<rep>\d+)"
    r"_(?P<date>\d{8})(?P<suffix>[A-Za-z]*)\.[A-Za-z]+$")

COL_NAME = "图片名"

# inference.py 写出来的列顺序。**没有表头的 CSV 按这个补名** ——
# 合并多个 CSV、或用 Excel 另存时，表头很容易丢掉，而列顺序一直是这个。
DEFAULT_COLUMNS = ["图片名", "根数量", "起点锚定(条)", "总根长(px)", "总根系面积(px²)",
                   "平均根长(px)", "最长根(px)", "各根长度(px)", "茎面积(px²)",
                   "检查区面积(px²)", "check_ok", "root_ok",
                   "总根长(mm)", "总根系面积(mm²)", "平均根长(mm)", "最长根(mm)"]
# 两个纵坐标指标：(内部键, 图里显示名, 纵轴标题, CSV 列名, 数值格式)
METRICS = (
    ("len", "根长", "总根长 (px)", "总根长(px)", "{:.1f}"),
    ("area", "根面积", "总根系面积 (px²)", "总根系面积(px²)", "{:.0f}"),
)

# ---- 配色（dataviz 规范前 4 个分类色槽，已跑 validate_palette.js 校验通过）----
# 明度带 / 彩度下限 / 色盲相邻对 ΔE 9.1 / 常视 ΔE 22.9 全 PASS；
# aqua(#1baf7a) 与 yellow(#eda100) 对底色对比度 <3:1 → 规范要求「可见标签兜底」，
# 本工具用「图例 + 汇总表」兜底（4 条线的图还额外做线端直接标注）。
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
# 对比模式 8 条线**不能上 8 种颜色**（规范禁止生成新色相，且第 5 色起无法保证色盲可辨）。
# 改用**复合编码**：颜色(4) × 线型(2) —— 处理方式看线型，重复次序看颜色。
KIND_LS = ["-", "--", ":", "-."]           # 第 1/2/3… 种处理方式的线型
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="把推理结果 CSV 按编号画成折线图（总根长 / 总根系面积各一份）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=('示例：\n'
                '  python tool\\analysis_datas\\analysis_datas.py --csv "D:\\merged.csv"\n'
                '  python tool\\analysis_datas\\analysis_datas.py --csv merged.csv --compare C,P\n'),
    )
    p.add_argument("--csv", default=None,
                   help="输入 CSV，逗号分隔可给多个；省略则用 analysis_datas/ 下所有 *.csv")
    p.add_argument("--out", type=Path, default=None,
                   help="输出根目录；默认 analysis_datas/（每个 csv 在其中各建一个同名文件夹）")
    p.add_argument("--compare", default=None,
                   help="对比模式：给两种处理方式，逗号分隔（如 C,P 或 C,S）。"
                        "把同一编号下的两种处理画进同一张图（8 条线）")
    p.add_argument("--ids", default=None,
                   help="只画这些编号，逗号分隔（如 001,002）；默认全画")
    p.add_argument("--x-cat", action="store_true",
                   help="横坐标按日期**等距**排，而不是按真实时间间隔。"
                        "对比不同批次时必须要（如 C 是 2024-12、S 是 2025-11，隔了近一年，"
                        "按真实时间轴画 C 会挤成一条线）")
    p.add_argument("--dpi", type=int, default=150, help="单图分辨率，默认 150")
    p.add_argument("--no-overview", action="store_true", help="不生成 _总览.png")
    p.add_argument("--keep-bad", action="store_true",
                   help="保留 check_ok/root_ok=否 的点（默认跳过，避免不可信数据进趋势）")
    return p.parse_args(argv)


def parse_name(raw: str):
    """图片名 -> (方式, 编号, 重复, 日期)；解析不了返回 None。"""
    s = raw.replace(" ", "")
    for pre in PREFIXES:
        if s.startswith(pre):
            s = s[len(pre):]
            break
    m = NAME_PAT.match(s)
    return (m["kind"], m["num"], m["rep"], m["date"]) if m else None


def read_csv_text(path: Path):
    """按几种编码依次尝试读 CSV，返回 (文本, 实际用的编码)。

    为什么要回退：inference.py 写出来的是**带 BOM 的 UTF-8**（utf-8-sig），
    但只要用 Excel 打开再另存一次，Excel 就会**丢掉 BOM、并换成本地代码页**
    （中文 Windows 上是 GBK）—— 再读就报 `UnicodeDecodeError: byte 0xca ...`。
    gb18030 是 GBK 的超集，能覆盖全部中文字符。
    """
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "gb18030"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    # 两种都不行（编码很怪）：latin-1 对任何字节都不抛异常，至少能读进来，
    # 但中文会变乱码 —— 交由调用方在输出里提示。
    return raw.decode("latin-1"), "latin-1（兜底，中文可能乱码）"


def load_csv(path: Path, keep_bad: bool):
    """读 CSV -> {(方式,编号,重复): {日期: {指标: 值}}}，外加统计信息。"""
    text, enc = read_csv_text(path)
    raw_rows = [r for r in csv.reader(text.splitlines())
                if r and any(c.strip() for c in r)]
    if not raw_rows:
        raise SystemExit(f"[错误] {path.name} 是空的（或只有空行）")
    if COL_NAME in raw_rows[0]:
        fieldnames, body = raw_rows[0], raw_rows[1:]
        no_header = False
    else:
        # **没有表头**：合并多个 CSV、或 Excel 另存，都很容易把表头丢掉。
        # 按 inference.py 的列顺序补一个；列数对不上时 zip 自动截断，取值不受影响。
        fieldnames, body, no_header = DEFAULT_COLUMNS, raw_rows, True
    rows = [dict(zip(fieldnames, r)) for r in body]
    data = defaultdict(lambda: defaultdict(dict))
    st = {"总行数": 0, "跳过不可信": 0, "文件名无法解析": 0, "数值无法解析": 0,
          "重复点": [], "含空格": 0, "编码": enc, "无表头": no_header}
    cleaned = []
    for r in rows:
        st["总行数"] += 1
        raw = (r.get(COL_NAME) or "").strip()
        if raw.startswith("#"):
            continue
        name = raw.replace(" ", "")            # 项目约定：按名字解析前先去空格
        if name != raw:
            st["含空格"] += 1
        row = dict(r)
        row[COL_NAME] = name
        cleaned.append(row)
        key = parse_name(name)
        if key is None:
            st["文件名无法解析"] += 1
            continue
        if not keep_bad and ("否" in (r.get("check_ok", "是"), r.get("root_ok", "是"))):
            st["跳过不可信"] += 1
            continue
        vals = {}
        for mk, _, _, col, _fmt in METRICS:
            try:
                vals[mk] = float(str(r.get(col, "")).strip())
            except ValueError:
                pass
        if not vals:
            st["数值无法解析"] += 1
            continue
        kind, num, rep, date = key
        if date in data[(kind, num, rep)]:
            st["重复点"].append(f"{name}（{date}）")
        data[(kind, num, rep)][date] = vals
    st["_cleaned"] = (fieldnames, cleaned)
    return data, st


def write_cleaned(fieldnames, cleaned, out_path: Path):
    """把去掉空格后的 CSV 另存一份副本（**不动原始文件**）。"""
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(cleaned)


def declutter(points, min_gap):
    """线端标签去重叠：按 y 排序后强制相邻间距 >= min_gap，返回 [(文本, y)]。"""
    pts = sorted(points, key=lambda t: t[0])
    ys = [y for y, _ in pts]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    return [(txt, y) for (_, txt), y in zip(pts, ys)]


def _decorate(ax, title, ylabel):
    ax.set_title(title, fontsize=14, color=INK, loc="left", pad=12)
    ax.set_xlabel("拍摄日期", fontsize=10, color=INK2, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=10, color=INK2, labelpad=8)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def plot_chart(title, series, ylabel, value_fmt, out_path: Path, dpi: int,
               annotate_ends: bool, x_cat: bool = False):
    """画一张折线图并保存。

    series: [(标签, 线型, 颜色, {日期: 值})]，已按想要的顺序排好。
    annotate_ends：4 条线时做线端直接标注；8 条线时不标（会糊成一团），
                   身份靠「图例 + 汇总表」承载。
    x_cat：横坐标按**日期等距**排（每个日期一格），而不是按真实时间间隔。
           对比不同批次的处理方式时必须用它 —— 例如 C 是 2024-12、S 是 2025-11，
           隔了近一年，按真实时间轴画的话 C 的一堆日期会全挤成左边一条线。
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    all_dates = sorted({d for _, _, _, pts in series for d in pts})
    slot = {d: i for i, d in enumerate(all_dates)}      # x_cat 用：日期 -> 槽位

    ends, lo, hi = [], float("inf"), float("-inf")
    for label, ls, color, pts in series:
        dates = sorted(pts)
        if x_cat:
            xs = [slot[d] for d in dates]
        else:
            xs = [datetime.strptime(d, "%Y%m%d") for d in dates]
        ys = [pts[d] for d in dates]
        lo, hi = min(lo, min(ys)), max(hi, max(ys))
        ax.plot(xs, ys, color=color, linestyle=ls, lw=1.8, marker="o", ms=5.0,
                markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=3, label=label)
        ends.append((ys[-1], label))

    span = max(hi - lo, 1.0)
    ax.set_ylim(0, hi + span * 0.28)          # 纵轴从 0 起：这是「量」，截断会放大波动

    if x_cat:
        ax.set_xticks(range(len(all_dates)))
        ax.set_xticklabels(all_dates, rotation=45, ha="right")
    else:
        days = (datetime.strptime(all_dates[-1], "%Y%m%d")
                - datetime.strptime(all_dates[0], "%Y%m%d")).days
        if len(all_dates) <= 12 and days <= 120:
            # 同一次拍摄批次内的日期：全都标出来（抽稀反而看不出采样节奏）
            ax.set_xticks([datetime.strptime(d, "%Y%m%d") for d in all_dates])
            ax.set_xticklabels(all_dates, rotation=45, ha="right")
        else:
            # 跨度大（比如跨批次）时不要硬塞：交给 AutoDateLocator 抽稀
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y%m%d"))

    if annotate_ends:
        ylim = ax.get_ylim()
        last_x = ax.get_xlim()[1]
        for txt, y1 in declutter(ends, (ylim[1] - ylim[0]) * 0.062):
            ax.annotate(txt, xy=(last_x, y1), xytext=(5, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=9, color=INK2,
                        annotation_clip=False)
        fig.subplots_adjust(right=0.86, left=0.10, top=0.88, bottom=0.20)
    else:
        fig.subplots_adjust(right=0.98, left=0.10, top=0.86, bottom=0.20)

    _decorate(ax, title, ylabel)
    _legend(ax, series, annotate_ends)
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def _legend(ax, series, annotate_ends):
    """图例。

    4 条线：一条普通图例（颜色即身份）。
    8 条线（对比模式）：拆成两组 —— 颜色说明重复次序、线型说明处理方式，
    6 条图例解释 8 条线，比堆 8 条好读。
    """
    if annotate_ends:
        leg = ax.legend(frameon=False, fontsize=9, ncol=len(series),
                        loc="upper left", handlelength=1.6, columnspacing=1.4)
        for t in leg.get_texts():
            t.set_color(INK2)
        return
    kinds, reps = [], []
    for label, ls, color, _ in series:
        kind, _, rep = label.rpartition("-")
        if ls not in [k[0] for k in kinds]:
            kinds.append((ls, kind))
        if color not in [r[0] for r in reps]:
            reps.append((color, rep))
    h_col = [Line2D([], [], color=c, lw=2.2, label=f"重复 {rep}") for c, rep in reps]
    h_ls = [Line2D([], [], color=INK2, lw=2.2, ls=ls, label=f"{kind} 处理")
            for ls, kind in kinds]
    leg1 = ax.legend(handles=h_col, frameon=False, fontsize=9, ncol=len(h_col),
                     loc="upper left", handlelength=1.8, columnspacing=1.6,
                     title="重复次序", title_fontsize=9)
    leg1.get_title().set_color(INK2)
    for t in leg1.get_texts():
        t.set_color(INK2)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=h_ls, frameon=False, fontsize=9, ncol=len(h_ls),
                     loc="upper right", handlelength=2.4, columnspacing=1.6,
                     title="处理方式", title_fontsize=9)
    leg2.get_title().set_color(INK2)
    for t in leg2.get_texts():
        t.set_color(INK2)


def plot_overview(titles, series_of, out_path: Path, dpi: int = 80):
    """所有编号的缩略图拼成一张总览，快速扫哪个编号有异常。"""
    n = len(titles)
    cols = min(20, max(1, int(n ** 0.5 * 1.6)))
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.15), dpi=dpi)
    fig.patch.set_facecolor(SURFACE)
    axes = axes.ravel() if n > 1 else [axes]
    for ax, title in zip(axes, titles):
        ax.set_facecolor(SURFACE)
        for _, ls, color, pts in series_of(title):
            ds = sorted(pts)
            # 带 marker：只有一个日期的编号只画得出一根单点折线，没 marker 整个格子是空的
            ax.plot([datetime.strptime(d, "%Y%m%d") for d in ds],
                    [pts[d] for d in ds], color=color, linestyle=ls, lw=1.2,
                    marker="o", ms=1.8)
        ax.set_title(title, fontsize=6, color=INK2, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_color(GRID)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("总览：每个编号一张缩略图（同色系含义同单图）", fontsize=10,
                 color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def _rng(vals, mk, fmt, fn=min):
    """从 [{指标: 值}] 里取某指标的最小/最大值，格式化；一个都没有就返回 '-'。"""
    xs = [v[mk] for v in vals if mk in v]
    return fmt.format(fn(xs)) if xs else "-"


def series_for(data, kind, num, reps, metric_mk):
    """给某个 (方式,编号) 组装各重复次序的折线数据，缺该指标的点跳过。"""
    out = []
    for i, rep in enumerate(reps):
        pts = {d: v[metric_mk] for d, v in data[(kind, num, rep)].items()
               if metric_mk in v}
        if pts:
            out.append((f"{kind}{num}-{rep}", "-", SERIES[i % len(SERIES)], pts))
    return out


def main():
    args = parse_args()
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    out_root = args.out or (PROJECT_ROOT / "analysis_datas")
    if args.csv:
        csvs = [Path(s.strip()) for s in str(args.csv).split(",") if s.strip()]
    else:
        csvs = sorted(out_root.glob("*.csv"))
        if not csvs:
            sys.exit(f"[错误] 没给 --csv，{out_root} 下也没有 csv 文件")
    for c in csvs:
        if not c.is_file():
            sys.exit(f"[错误] 找不到 CSV: {c}")

    want = {s.strip() for s in str(args.ids).split(",")} if args.ids else None
    compare = None
    if args.compare:
        compare = [s.strip() for s in str(args.compare).split(",") if s.strip()]
        if len(compare) != 2:
            sys.exit(f"[错误] --compare 要给两种处理方式（如 C,P），当前是 {args.compare!r}")

    for csv_path in csvs:
        data, st = load_csv(csv_path, args.keep_bad)
        all_reps = sorted({r for _, _, r in data}, key=lambda x: (len(x), x))
        kinds = sorted({k for k, _, _ in data})
        nums = sorted({n for _, n, _ in data})
        if want:
            nums = [n for n in nums if n in want]
        if not nums:
            print(f"[跳过] {csv_path.name}: 没有可画的编号\n")
            continue

        out_dir = naming.create_unique_dir(out_root, csv_path.stem)
        print(f"=== {csv_path.name} → {out_dir.name}/ ===")
        print(f"  共 {st['总行数']} 行 | 处理方式 {kinds} | 编号 {len(nums)} 个 | 重复 {all_reps}"
              + (f" | 跳过不可信 {st['跳过不可信']} 行" if st["跳过不可信"] else "")
              + (f" | 文件名无法解析 {st['文件名无法解析']} 行" if st["文件名无法解析"] else ""))
        # 非 UTF-8 时明确报出来：多半是 Excel 另存过一次（BOM 丢了、换成 GBK）
        if st["编码"] != "utf-8-sig":
            print(f"  [提示] 这个 CSV 是 **{st['编码']}** 编码（不是 inference.py 写的 UTF-8+BOM），"
                  f"已按 {st['编码']} 读入。多半是用 Excel 打开后另存过 —— 不影响本次读取，"
                  f"但文件名里的中文若有异常，请核对一下。")
        if st["无表头"]:
            print("  [提示] 这个 CSV **没有表头行**，已按 inference.py 的列顺序"
                  "（图片名/根数量/…/check_ok/root_ok）认列。合并 CSV 时容易丢掉表头 —— "
                  "不影响本次读取。")
        if len(kinds) > 1 and not compare:
            print(f"  {len(kinds)} 种处理方式各画一套；想叠进同一张图对比，用 "
                  f"--compare {','.join(kinds[:2])}")
        if st["含空格"]:
            fn, cl = st["_cleaned"]
            dst = out_dir / f"{csv_path.stem}_无空格.csv"
            write_cleaned(fn, cl, dst)
            print(f"  [清理] {st['含空格']} 个图片名含空格，已按项目约定去掉；"
                  f"副本 {dst.name}（**原始文件未改动**）")
        if st["重复点"]:
            print(f"  [注意] {len(st['重复点'])} 个重复点，已取最后一次：{st['重复点'][:3]}")

        summary, titles = [], []
        overview = {}          # 总览用：标题 -> 该格子的折线
        sum_header = ["处理方式", "编号", "日期数", "各重复次序的点数",
                      "总根长最小", "总根长最大", "总根系面积最小", "总根系面积最大"]
        n_drawn = 0
        if compare:
            k1, k2 = compare
            # 有数据就算「有」：只探 all_reps[0] 的话，某个编号恰好缺第 1 次重复
            # 就会被误判成「只有一种处理」而整块跳过
            has = lambda kd, n: any((kd, n, r) in data for r in all_reps)  # noqa: E731
            have = [n for n in nums if has(k1, n) and has(k2, n)]
            missing = [n for n in nums if n not in have]
            print(f"  对比模式 {k1} vs {k2}：{len(have)} 个编号两种处理都有"
                  + (f"，{len(missing)} 个只有其中一种，跳过" if missing else ""))
            for num in have:
                for mk, mname, ylabel, _col, vfmt in METRICS:
                    ser = []
                    for ki, kind in enumerate(compare):
                        for i, rep in enumerate(all_reps):
                            pts = {d: v[mk] for d, v in data.get((kind, num, rep), {}).items()
                                   if mk in v}
                            if pts:
                                ser.append((f"{kind}{num}-{rep}", KIND_LS[ki % len(KIND_LS)],
                                            SERIES[i % len(SERIES)], pts))
                    if not ser:
                        continue
                    plot_chart(f"{num} 号  {' vs '.join(compare)} 对比 — {mname}",
                               ser, ylabel, vfmt,
                               out_dir / f"{num}_对比_{mname}.png", args.dpi,
                               annotate_ends=False, x_cat=args.x_cat)
                    n_drawn += 1
                allv = [v for kd in compare for r in all_reps
                        for d, v in data.get((kd, num, r), {}).items()]
                dates = {d for kd in compare for r in all_reps
                         for d in data.get((kd, num, r), {})}
                summary.append([
                    ",".join(compare), num, len(dates),
                    " ".join(f"{kd}:{sum(1 for r in all_reps if (kd, num, r) in data)}"
                             for kd in compare),
                    _rng(allv, "len", "{:.1f}"), _rng(allv, "len", "{:.1f}", max),
                    _rng(allv, "area", "{:.0f}"), _rng(allv, "area", "{:.0f}", max)])
            titles = have
            overview = {t: sum((series_for(data, kd, t, all_reps, "len")
                                for kd in compare), []) for t in have}
            # 按**实际画出的**张数报：CSV 里没有 总根系面积 列时面积图会被跳过，
            # 报「编号数 × 指标数」会虚高
            print(f"  已画 {len(have)} 个编号，共 {n_drawn} 张图"
                  + ("（缺 总根系面积 列，面积图跳过）" if n_drawn < len(have) * len(METRICS)
                     else ""))
        else:
            # 每种处理方式各画一套（不是只画 kinds[0]）：CSV 里 C、P 都有时，
            # 两种都要出图，否则 P 的数据会被静默丢掉
            tasks = [(kind, num) for kind in kinds for num in nums
                     if any(data.get((kind, num, r)) for r in all_reps)]
            unit = "套（处理方式×编号）" if len(kinds) > 1 else "个编号"
            for i, (kind, num) in enumerate(tasks, 1):
                tag = f"{kind}{num}"
                for mk, mname, ylabel, _col, vfmt in METRICS:
                    ser = []
                    for j, rep in enumerate(all_reps):
                        pts = {d: v[mk] for d, v in data.get((kind, num, rep), {}).items()
                               if mk in v}
                        if pts:
                            ser.append((f"{tag}-{rep}", "-",
                                        SERIES[j % len(SERIES)], pts))
                    if not ser:
                        continue
                    plot_chart(f"{tag}  {mname}随时间变化", ser, ylabel, vfmt,
                               out_dir / f"{tag}_{mname}.png", args.dpi,
                               annotate_ends=True, x_cat=args.x_cat)
                    n_drawn += 1
                titles.append(tag)
                overview[tag] = series_for(data, kind, num, all_reps, "len")
                allv = [v for r in all_reps
                        for d, v in data.get((kind, num, r), {}).items()]
                dates = {d for r in all_reps for d in data.get((kind, num, r), {})}
                summary.append([
                    kind, num, len(dates),
                    " ".join(f"{rep}:{len(data.get((kind, num, rep), {}))}"
                             for rep in all_reps),
                    _rng(allv, "len", "{:.1f}"), _rng(allv, "len", "{:.1f}", max),
                    _rng(allv, "area", "{:.0f}"), _rng(allv, "area", "{:.0f}", max)])
                if i % 50 == 0 or i == len(tasks):
                    print(f"  已画 {i}/{len(tasks)} {unit}，共 {n_drawn} 张图"
                          + ("（缺 总根系面积 列，面积图跳过）"
                             if n_drawn < i * len(METRICS) else ""))

        with open(out_dir / "_汇总.csv", "w", encoding="utf-8-sig", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(sum_header)
            wr.writerows(summary)
        if not args.no_overview and titles:
            plot_overview(titles, overview.get, out_dir / "_总览.png")
        print(f"  输出: {out_dir}\n")


if __name__ == "__main__":
    main()
