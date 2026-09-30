"""把 `<名>.rsml` 里的根系折线并进 `<名>.json`（labelme）的 shapes 数组。

背景：本项目原来一条样本有两个标注文件 ——

    <名>.rsml   根系折线（RootNav spline，XML）
    <名>.json   茎横截面 + 检查范围（labelme）

2026-09-30 起改成**一个 labelme json 装三个通道**，根系作为折线写进同一个 shapes 数组：

    {"label": "root", "shape_type": "linestrip", "points": [[x, y], ...]}

本工具做的就是这件事：读 rsml 的折线，作为 root 形状**追加**到同名 json 的 shapes 末尾。

    改前  C001-1.json   [check_background, stem]                    +  C001-1.rsml
    改后  C001-1.json   [check_background, stem, root, root, ...]

为什么是文本插入，而不是 json.load → json.dump 写回：
    `load → dump` 会把**整个文件重写一遍** —— 缩进、空格、浮点写法、转义都可能变，
    造出一个巨大的假 diff，也没法证明「只动了这一处」。标注数据不该冒这个险
    （同 [tool/repair_json](../repair_json/readme.md)、[tool/repair_rsml](../repair_rsml/readme.md)）。
    本工具只在文本层面往 `"shapes"` 数组里插入新对象，**其余字节原样保留**，
    并在写盘前自检「抠掉插入的那一段之后与原文逐字节相同」。

    定位 `"shapes"` 用的是**字符串感知的扫描器**而不是正则：正则应排不掉
    「字符串里的」和「嵌套对象里的」同名键（base64 的 imageData 就是现成的反例）。

用法：
    python merge_annot.py --dir "D:\\数据集总表\\root" -r --dry-run    # 先预览
    python merge_annot.py --dir "D:\\数据集总表\\root" -r --backup "D:\\_backup"
    python merge_annot.py --dir "D:\\数据集\\train\\labels\\other"     # 项目布局，零参数可用

**改动前请先 --dry-run 预览一遍。** 这是就地改文件的有损操作。

顺带一提：本工具**不删 `.rsml`** —— 它是这批数据的回退点，也是加载器的兜底。
"""
import argparse
import csv
import fnmatch
import hashlib
import json
import os
import shutil
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from common.rsml_parse import parse_rsml  # noqa: E402

LOG_NAME = "merge_annot_log.txt"

ROOT_LABEL = "root"
ROOT_SHAPE_TYPE = "linestrip"
# labelme 单条形状的确切键序（照抄 6.x 的输出，多一个少一个都会和既有文件不一致）
SHAPE_KEYS = ("label", "points", "group_id", "description",
              "shape_type", "flags", "mask")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="把 rsml 的根系折线并进 labelme json 的 shapes 数组",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=('示例：\n'
                '  python merge_annot.py --dir "D:\\数据集总表\\root" -r --dry-run\n'
                '  python merge_annot.py --dir "D:\\数据集总表\\root" -r\n'
                '  python merge_annot.py --dir "D:\\数据集\\train\\labels\\other"\n'),
    )
    p.add_argument("--dir", required=True, help="目标文件夹（找里面的 .json）")
    p.add_argument("-r", "--recursive", action="store_true", help="递归处理子文件夹")
    p.add_argument("--rsml-dir", default=None,
                   help="rsml 所在文件夹；默认与 json 同目录，"
                        "并自动识别 labels/roots 这种分目录布局")
    p.add_argument("--dry-run", action="store_true",
                   help="只打印将要插入的内容，不实际写文件")
    p.add_argument("--backup", default=None,
                   help="改之前把原 json 按相对路径拷到这个目录（回滚用）。"
                        "**必须在数据目录之外**，否则会被 separate_dataset 当数据卷进划分")
    p.add_argument("--only", default=None,
                   help="只处理主干名匹配这个通配符的文件（先用它转一个试试，再全量跑）")
    return p.parse_args(argv)


def select(files, only):
    """按 --only 过滤。支持通配符（* ? []），也支持直接给主干名。"""
    if not only:
        return files
    hit = [f for f in files if fnmatch.fnmatch(f.stem, only)]
    if not hit:
        sys.exit(f"[错误] --only {only!r} 没匹配到任何文件")
    return hit


# ---------------------------------------------------------------- 读写

def read_text(path: Path) -> str:
    """按 utf-8 读，**不做换行转换**（newline=""）—— 否则 \\n 会在写回时变成 \\r\\n，
    等于把整个文件的换行都改了。"""
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


def write_text(path: Path, text: str):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- JSON 扫描

def _skip_string(text: str, i: int) -> int:
    """text[i] 是开引号；返回闭引号之后的下标（含转义处理）。"""
    n = len(text)
    i += 1
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == '"':
            return i + 1
        i += 1
    return n


def _value_end(text: str, start: int) -> int:
    """返回从 start 开始的那个 JSON 值的结束下标（不含）。"""
    n = len(text)
    if start >= n:
        return start
    ch = text[start]
    if ch == '"':
        return _skip_string(text, start)
    if ch in "[{":
        depth = 0
        i = start
        while i < n:
            c = text[i]
            if c == '"':
                i = _skip_string(text, i)
                continue
            if c in "[{":
                depth += 1
            elif c in "]}":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return n
    i = start
    while i < n and text[i] not in ",}] \t\r\n":
        i += 1
    return i


def find_top_level_key(text: str, key: str):
    """在 JSON **根对象**里找 key，返回 (值起始, 值结束, 键所在列)；找不到返回 None。

    三个约束缺一不可：
      1. 前缀必须是**深度 1** 的字符串 —— 嵌套对象里的同名键不算；
      2. 后面必须紧跟 `:` —— 排除同名的字符串**值**；
      3. 字符串字面量整体跳过（含 \\ 转义）—— 所以 imageData 里的 base64
         不可能被误当成结构。
    """
    n = len(text)
    depth = 0
    i = 0
    while i < n:
        ch = text[i]
        if ch == '"':
            start = i
            i = _skip_string(text, i)
            if depth == 1:
                j = i
                while j < n and text[j] in " \t\r\n":
                    j += 1
                if j < n and text[j] == ":":
                    try:
                        name = json.loads(text[start:i])
                    except json.JSONDecodeError:
                        name = None
                    if name == key:
                        j += 1
                        while j < n and text[j] in " \t\r\n":
                            j += 1
                        col = start - (text.rfind("\n", 0, start) + 1)
                        return j, _value_end(text, j), col
            continue
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        i += 1
    return None


# ---------------------------------------------------------------- 渲染与插入

def render_shape(points, indent: str, eol: str) -> str:
    """把一条折线渲染成 labelme 的单条形状（json.dumps(indent=2) 的排版**就是**
    labelme 的排版，整体平移缩进即可）。"""
    shape = {
        "label": ROOT_LABEL,
        "points": [[float(x), float(y)] for x, y in points],
        "group_id": None,
        "description": "",
        "shape_type": ROOT_SHAPE_TYPE,
        "flags": {},
        "mask": None,
    }
    s = json.dumps(shape, indent=2, ensure_ascii=False)
    if eol != "\n":                    # json.dumps 只用 \n，CRLF 文件要换回去
        s = s.replace("\n", eol)
    return textwrap.indent(s, indent)


def build_insert(text: str, span, polylines):
    """返回 (新文本, 插入段的 [起, 止])。插入段抠掉后必须与原文逐字节相同。"""
    val_start, val_end, col = span
    eol = "\r\n" if "\r\n" in text else "\n"
    indent = " " * (col + 2)
    elems = [render_shape(p, indent, eol) for p in polylines]
    joined = ("," + eol).join(elems)

    if text[val_start] != "[":
        raise ValueError("shapes 不是数组")

    if not text[val_start + 1:val_end - 1].strip():
        # 空数组 "shapes": [] —— 刚存过只含 stem 的 json 长这样
        block = eol + joined + eol + " " * col
        at = val_start + 1
    else:
        # 非空：回退到 ] 前最后一个有效字节，断言它是最后一条形状的 }
        k = val_end - 2
        while k > val_start and text[k] in " \t\r\n":
            k -= 1
        if text[k] != "}":
            raise ValueError(f"shapes 数组的收尾不是 }}（找到 {text[k]!r}）")
        at = k + 1
        block = "," + eol + joined

    return text[:at] + block + text[at:], (at, at + len(block))


# ---------------------------------------------------------------- 写前自检

def verify(old_text: str, new_text: str, span, polylines):
    """自检不通过就抛异常，绝不写盘。

    最强的一条是**文本级**的：抠掉插入段后与原文逐字节相同 —— 它一次性覆盖了
    「其余顶层键、CRLF、base64 全都没变」，比任何字段级比对都硬。
    """
    a, b = span
    if new_text[:a] + new_text[b:] != old_text:
        raise ValueError("抠掉插入段后与原文不一致（改动溢出了插入点）")

    old = json.loads(old_text)
    new = json.loads(new_text)

    if old.get("shapes") is None:
        raise ValueError("原文没有 shapes")
    n = len(old["shapes"])
    if new["shapes"][:n] != old["shapes"]:
        raise ValueError("原有 shapes 被改动了")
    if len(new["shapes"]) != n + len(polylines):
        raise ValueError(f"插入条数不对：期望 {len(polylines)}，"
                         f"实际 {len(new['shapes']) - n}")

    if list(new.keys()) != list(old.keys()):
        raise ValueError("顶层键或键序被改动了")
    for k in old:
        if k != "shapes" and new[k] != old[k]:
            raise ValueError(f"顶层键 {k!r} 的值被改动了")

    inserted = new["shapes"][n:]
    for s in inserted:
        if tuple(s.keys()) != SHAPE_KEYS:
            raise ValueError(f"插入项的键不对：{list(s.keys())}")
        if s["label"] != ROOT_LABEL or s["shape_type"] != ROOT_SHAPE_TYPE:
            raise ValueError("插入项的 label/shape_type 不对")
        if len(s["points"]) < 2:
            raise ValueError("插入项点数少于 2")
        # 浮点必须逐位相同：float → json.dumps → 文本 → json.loads 是精确往返
        if [[float(x), float(y)] for x, y in s["points"]] != s["points"]:
            raise ValueError("坐标不是精确的浮点数")

    got = [[tuple(p) for p in s["points"]] for s in inserted]
    want = [[tuple(p) for p in r.points] for r in polylines]
    if got != want:
        raise ValueError("回读的坐标与 rsml 解析结果不逐位相等")
    return len(inserted)


# ---------------------------------------------------------------- 单个文件

def sibling_rsml(json_path: Path, rsml_dir):
    """找与 json 同名的 rsml。

    先看显式指定的目录；否则**优先按项目布局找** labels/other/x.json → labels/roots/x.rsml，
    找不到再退回「与 json 同目录」（扁平布局）。
    """
    if rsml_dir is not None:
        return rsml_dir / f"{json_path.stem}.rsml"
    p = json_path.parent
    if p.name == "other":
        for name in ("roots", "root"):
            c = p.parent / name / f"{json_path.stem}.rsml"
            if c.exists():
                return c
    return json_path.with_suffix(".rsml")


def format_ok(text: str) -> str:
    """返回 ''（合格）或拒绝原因。不合格就拒改 —— 宁可不动，也不猜。"""
    if text.startswith("\ufeff"):
        return "有 BOM"
    crlf = text.count("\r\n")
    lf = text.count("\n") - crlf
    if crlf and lf:
        return f"换行混用（CRLF {crlf} / LF {lf}）"
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        return f"json 解析失败（{e}）"
    return ""


def merge_one(json_path: Path, rsml_dir, dry_run: bool, backup_dir):
    """返回 (状态, 详情行列表, 附加信息)。状态见 main() 的统计口径。"""
    try:
        text = read_text(json_path)
    except (UnicodeDecodeError, OSError) as e:
        return "读文件失败（跳过）", [str(e)], {}

    bad = format_ok(text)
    if bad:
        return "格式不合格（跳过）", [bad], {}

    span = find_top_level_key(text, "shapes")
    if span is None:
        return "找不到顶层 shapes（跳过）", [], {}

    data = json.loads(text)
    if span[2] != 2:
        return "缩进不是 2 空格（跳过）", [f"键在第 {span[2]} 列"], {}
    n_root = sum(1 for s in data.get("shapes") or [] if s.get("label") == ROOT_LABEL)
    if n_root:
        # 幂等：已经并过了。抽出来的图片/C 里那 2 个就走这条。
        return "已有 root 标注（跳过）", [f"已有 {n_root} 条"], {}

    rsml = sibling_rsml(json_path, rsml_dir)
    if not rsml.exists():
        return "找不到同名 rsml（跳过）", [], {}

    try:
        roots = parse_rsml(rsml)
    except Exception as e:                       # noqa: BLE001 —— XML 各种坏法都归这里
        return "rsml 解析失败（跳过）", [str(e)], {}
    polys = [r.points for r in roots if len(r.points) >= 2]
    if not polys:
        # 合法的「这张图没有根」负样本（plant_S003-3 就是），不是错误
        return "rsml 无根（跳过）", [], {}

    try:
        new_text, new_span = build_insert(text, span, polys)
        verify(text, new_text, new_span, roots)
    except (ValueError, AssertionError) as e:
        return "自检未通过（未改）", [str(e)], {}

    n_pts = sum(len(p) for p in polys)
    x0, y0 = polys[0][0]
    meta = {"old_sha256": sha256(text), "new_sha256": sha256(new_text),
            "n_poly": len(polys), "n_pts": n_pts,
            "grew": len(new_text) - len(text)}
    detail = [f"插入 {len(polys)} 条 / {n_pts} 点 / +{meta['grew']} 字节",
              f"首条 {len(polys[0])} 点，首点 ({x0:.1f}, {y0:.1f})"]

    if not dry_run:
        if backup_dir is not None:
            dst = backup_dir / json_path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(json_path, dst)
        tmp = json_path.with_suffix(json_path.suffix + ".tmp")
        write_text(tmp, new_text)
        os.replace(tmp, json_path)               # 原子替换，绝不留下半个文件
    return "已合并", detail, meta


# ---------------------------------------------------------------- 主流程

def main():
    args = parse_args()
    root = Path(args.dir).resolve()
    if not root.is_dir():
        sys.exit(f"[错误] 目标文件夹不存在: {root}")

    backup_dir = None
    if args.backup:
        backup_dir = Path(args.backup).resolve()
        if backup_dir == root or root in backup_dir.parents:
            sys.exit(f"[错误] --backup 不能放在数据目录里面: {backup_dir}\n"
                     f"        它会连同原 json 一起被 separate_dataset 当成数据划进 train/test。")
    rsml_dir = Path(args.rsml_dir).resolve() if args.rsml_dir else None

    files = sorted(root.rglob("*.json") if args.recursive else root.glob("*.json"))
    # 上一轮跑出来的日志别当数据
    files = [f for f in files if f.name != LOG_NAME]
    files = select(files, args.only)
    if not files:
        sys.exit(f"[错误] {root} 里没有 .json 文件"
                 + ("" if args.recursive else "（子文件夹里的没算，要递归请加 -r）"))

    print(f"目标: {root}（{'递归' if args.recursive else '只看本层'}"
          + (f"；--only {args.only}" if args.only else "") + "）")
    print(f"找到 {len(files)} 个 .json"
          + ("   [dry-run：不改任何文件]\n" if args.dry_run else "\n"))

    stat, changed = {}, []
    for f in files:
        status, detail, meta = merge_one(f, rsml_dir, args.dry_run, backup_dir)
        stat[status] = stat.get(status, 0) + 1
        if status == "已合并":
            changed.append((f, meta))
        # 「找不到同名 rsml」会命中大量未标注的图，只统计不刷屏
        if status != "找不到同名 rsml（跳过）":
            print(f"  {'·' if args.dry_run else '✓'} [{status}] {f.relative_to(root)}")
            for line in detail:
                print(f"        {line}")

    print("\n统计: " + " | ".join(f"{k} {v}" for k, v in sorted(stat.items())))

    if args.dry_run and changed:
        print(f"\n[dry-run] 上面这 {len(changed)} 个文件会被修改。去掉 --dry-run 即正式执行。")
    if changed and not args.dry_run:
        log = root / LOG_NAME
        with open(log, "w", encoding="utf-8-sig", newline="") as fh:
            fh.write("# rsml → labelme json 合并记录\n"
                     "# 回滚：用 --backup 目录里的原件覆盖回去（找不到 backup 就用 .rsml 重跑本工具）\n")
            wr = csv.writer(fh)
            wr.writerow(["相对路径", "状态", "插入折线", "插入点数", "增长字节",
                         "原 sha256", "新 sha256"])
            for f, m in changed:
                wr.writerow([str(f.relative_to(root)), "已合并", m["n_poly"], m["n_pts"],
                             m["grew"], m["old_sha256"], m["new_sha256"]])
        print(f"已修改 {len(changed)} 个文件，改动记录: {log}")
        print("  ⚠️ 日志别留在数据文件夹里 —— separate_dataset 是按「文件夹里所有文件」"
              "划分的，会被当成一组文件划进 train/test。核对完请挪走。")

    if backup_dir is not None and changed and not args.dry_run:
        print(f"原文件备份在: {backup_dir}")


if __name__ == "__main__":
    main()
