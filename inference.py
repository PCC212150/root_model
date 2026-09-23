"""对任意图片文件夹做根系分割推理，统计 根数量 / 各根长度(px) / 总长度(px)，
并同时识别 茎横截面 与 检查范围(check_background)。

用法（readme 风格，两种写法均可）：
    python inference.py --model model_202609091135 --dir D:\\...\\某图片文件夹
    python inference.py --model_202609091135 "D:\\...\\某图片文件夹"   # 兼容
    python inference.py --dir D:\\...\\某图片文件夹                    # 模型省略=最新
    python inference.py --dir D:\\...\\某图片文件夹 --mm-per-px 0.1234  # CSV 追加 mm 列
    python inference.py --dir D:\\...\\某图片文件夹 --size 1536         # 覆盖输入长边
    python inference.py --dir D:\\...\\某图片文件夹 --save-mask         # 额外存 _mask.png
    python inference.py --dir D:\\...\\某图片文件夹 --overlay-jpg       # overlay 存 JPEG（快 47 倍）
    python inference.py --dir <文件夹> --jobs 4                       # 并发 4 张（吞吐约 2~3 倍）
    python inference.py --model model_a,model_b --dir ...              # 多模型集成

输入长边默认取**模型训练时的设置**（从权重里读），只有显式给 --size 才覆盖 ——
尺度必须与训练一致，否则精度会明显下降。

统计口径：
  - 根系只在**模型识别出的检查范围**内统计（范围外不计入），不扣茎；
  - 每条预测折线的**起点会锚定到茎边界**——茎外那圈黑色泡沫环不是根（模型判背景没错），
    但标注是从茎边开始画的，那一段被挡住、实际存在，所以补回来并计入根长。
    这一步只在「标注从茎边画」的口径下成立，当前数据集正是这种；
    若换成「只画看得见的根」的数据集，锚定会让总长系统性偏高，需要重新评估。

结果：result/{目标文件夹名}/
    - {目标文件夹名}.csv    每行一张图（UTF-8 BOM，Excel 直接双击可开）：
                            图片名 根数量 起点锚定(条) 总根长(px) 总根系面积(px²)
                            平均根长(px) 最长根(px) 各根长度(px) 茎面积(px²)
                            检查区面积(px²) check_ok root_ok
                            「各根长度」用分号分隔；--mm-per-px>0 时追加 mm 列；
                            文件末尾是若干以 # 开头的汇总行（Excel 可见，脚本可跳过）
    - {图片名}_overlay.png  原图 + 根系(红) + 茎(橙) + 检查范围(绿框)
                            （--overlay-jpg 时为 .jpg，写一张快 47 倍）
    - {图片名}.rsml         预测根系折线，每条折线一个 plant（不再分主根/侧根）
    - {图片名}_mask.png     统计口径的根系掩码（**默认不存**，加 --save-mask 才出）
目录/文件重名时自动追加 -1、-2 …（项目规范）。

**总根系面积** = 统计口径的根系掩码像素数（已限定在检查范围内），单位 px²。
它比根长脆弱得多：根长是骨架长度、几乎不受线宽/阈值影响，而面积随线宽与二值化阈值
**线性**变化（换个阈值能差一倍）。**适合同一条流水线内做相对比较，不要跨版本比绝对值。**

**逐张输出什么**（2026-09-17 起精简）：默认只有 `_overlay` + `.rsml` 两个文件。
`_stem.png` / `_check.png` 不再输出（overlay 里已用颜色标出），`_mask.png` 需要时加
`--save-mask`。这样每张图少写 3 个 5472x3648 的大 PNG，磁盘和耗时都省一截。
"""
import csv
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common import ckpt, image_io, naming, predict  # noqa: E402
from common.dataset import CH_CHECK, CH_ROOT, CH_STEM  # noqa: E402
from common.rsml_export import write_rsml  # noqa: E402
from common.skeleton_stats import analyze_mask_anchored  # noqa: E402


def parse_argv():
    """解析参数，兼容 readme 的 --model_xxx 与裸参数写法。"""
    model, folder, mm_per_px, size = None, None, None, None
    save_mask = False
    overlay_fmt = "png"
    jobs = 1
    tokens = sys.argv[1:]
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "--save-mask":
            save_mask = True
            i += 1
        elif t == "--jobs":
            # 并发处理张数：GPU 前向串行、CPU 部分并行。默认 1（与旧行为一致）
            jobs = int(tokens[i + 1]) if i + 1 < len(tokens) else 1
            i += 2
        elif t in ("--overlay-jpg", "--overlay-jpeg"):
            # overlay 存 JPEG（q90/4:4:4）：写一张快 47 倍、体积小 8 倍，画质对看结果够用
            overlay_fmt = "jpg"
            i += 1
        elif t == "--model":
            model = tokens[i + 1] if i + 1 < len(tokens) else None
            i += 2
        elif t == "--mm-per-px":
            mm_per_px = float(tokens[i + 1]) if i + 1 < len(tokens) else None
            i += 2
        elif t == "--size":
            size = int(tokens[i + 1]) if i + 1 < len(tokens) else None
            i += 2
        elif t in ("--dir", "--image_dir", "--images", "--folder", "--path"):
            folder = tokens[i + 1] if i + 1 < len(tokens) else None
            i += 2
        elif t.startswith("--model_"):
            model = t[2:]
            i += 1
        elif t.startswith("--"):
            key = t[2:].lstrip("-")
            if "model" in key:
                model = tokens[i + 1] if i + 1 < len(tokens) else None
            else:
                folder = tokens[i + 1] if i + 1 < len(tokens) else None
            i += 2
        else:
            if model is None:
                model = t
            elif folder is None:
                folder = t
            else:
                print(f"[错误] 无法识别的参数: {t}")
                sys.exit(1)
            i += 1
    return model, folder, mm_per_px, size, save_mask, overlay_fmt, jobs


_BLEND_LUT = {}


def _blend_lut(color, alpha):
    """混合用查找表：uint8 像素值 -> 混合后的 uint8 值，形状 (256, 3)。

    与 `arr.astype(float32)*(1-alpha) + color*alpha` 后 clip+取整**逐位相同**
    （同样的 float32 运算，只是提前对 256 个可能取值算好）。
    """
    key = (tuple(color), alpha)
    if key not in _BLEND_LUT:
        v = (np.arange(256, dtype=np.float32)[:, None] * (1.0 - alpha)
             + np.asarray(color, np.float32)[None, :] * alpha)
        _BLEND_LUT[key] = np.clip(v, 0, 255).astype(np.uint8)
    return _BLEND_LUT[key]


def make_overlay(img: np.ndarray, masks, check_box, alpha: float = 0.45) -> np.ndarray:
    """把识别结果叠到原图上：根(红) + 茎(橙)，检查范围画绿框。

    在 uint8 上直接查表混合，不再升到 float32：省掉一张 5472x3648 的 float32 拷贝
    （240MB，多进程并发时这个内存峰值是按份数翻的），也快一截。
    根/茎掩码若重叠，先混的那一层会先量化到 uint8 —— 实测差异在 ±1 灰阶、且只影响
    这张**给人看**的图，CSV / RSML / 掩码一概不受影响（统计在它之前就算完了）。
    """
    out = img.copy()
    cols = np.arange(out.shape[2])          # 逐通道查表，避免 lut[px] 广播成 (N,3,3)
    for ch, color in ((CH_ROOT, (255, 0, 0)), (CH_STEM, (255, 165, 0))):
        m = masks[ch]
        if m is not None and m.any():
            out[m] = _blend_lut(color, alpha)[out[m], cols]
    if check_box is not None:
        h, w = out.shape[:2]
        im = Image.fromarray(out)
        ImageDraw.Draw(im).rectangle(
            [check_box[0], check_box[1], check_box[2] - 1, check_box[3] - 1],
            outline=(0, 255, 0), width=max(2, int(min(w, h) * 0.004)))
        return np.asarray(im)
    return out


def process_one(p, model, size, device, out_dir, mm, overlay_fmt, save_mask,
                gpu_lock, tile=0):
    """处理一张图：预测 → 统计 → 写 overlay/RSML；返回 (CSV 行, 控制台文本)。

    **并发安全**：只有 GPU 那一小段用 gpu_lock 串行 —— 单张图的 GPU 活本来就少
    （模型前向实测 0.33s），串行不拖慢整体，却避免多线程同时抢显存；
    其余全是 numpy / PIL / skimage，各线程各管各的。

    tile>0 时走**原始分辨率滑窗**（切片模型专用，见 ckpt.infer_tile）；滑窗的前向
    同样在锁里，整张图的所有块串行跑完再放锁。
    """
    img = image_io.load_rgb(p)
    with gpu_lock:
        res = predict.predict(model, img, max_side=size, stride=config.STRIDE,
                              device=device, tile=tile)
    masks = res["masks"]
    # 起点锚定到茎：补回被泡沫环挡住的那一段（计入根长，与标注同口径）
    st = analyze_mask_anchored(
        res["mask_counted"], masks[CH_STEM] if len(masks) > CH_STEM else None,
        spur=config.PRED_SPUR_LENGTH, min_len=config.MIN_ROOT_LENGTH,
        factor=config.STEM_ANCHOR_FACTOR, min_px=config.STEM_ANCHOR_MIN_PX,
        max_px=config.STEM_ANCHOR_MAX_PX)
    count, lens, total = st["count"], st["lengths"], st["total"]
    len_str = ";".join(f"{v:.1f}" for v in lens) if lens else "-"
    # 总根系面积 = 统计口径的根系掩码像素数（已限定在检查范围内），单位 px²。
    # 注意它比根长脆弱得多：根长是骨架长度，几乎不受线宽/阈值影响；
    # 面积随线宽、二值化阈值**线性**变化（换个阈值能差一倍）。
    # 适合同一条流水线内做相对比较，不要跨版本比绝对值。
    root_area = int(res["mask_counted"].sum())
    stem_area = int(masks[CH_STEM].sum()) if len(masks) > CH_STEM else 0
    check_area = (int((res["check_box"][2] - res["check_box"][0])
                      * (res["check_box"][3] - res["check_box"][1]))
                  if res["check_ok"] else img.shape[0] * img.shape[1])
    row = [p.name, count, st["anchored_count"], f"{total:.1f}",
           root_area,
           f"{total / count:.1f}" if count else "0.0",
           f"{max(lens):.1f}" if lens else "0.0", len_str,
           stem_area, check_area, "是" if res["check_ok"] else "否",
           "是" if res.get("root_ok", True) else "否"]
    if mm:
        row += [f"{total * mm:.1f}", f"{root_area * mm * mm:.1f}",
                f"{total / count * mm:.1f}" if count else "0.0",
                f"{max(lens) * mm:.1f}" if lens else "0.0"]

    # ---- 保存识别结果图片 ----
    # 默认只出 _overlay（肉眼看结果）+ .rsml（数据）。
    # _stem / _check 两张掩码图 2026-09-17 起不再输出：overlay 里已经用颜色标了，
    # _mask 默认也不存（要看统计口径的掩码时加 --save-mask）。
    # 这三张都是 5472x3648 的大图，一张 20MB 上下，省下来是实打实的磁盘和时间。
    #
    # **写图是全流程最贵的一步**（实测 5472x3648：PNG 默认压缩 1891ms vs
    # 模型前向 333ms，GPU 因此长期闲着）。所以：
    #   PNG 用 compress_level=1 —— 535ms（快 3.5 倍，代价是体积 19→32MB）；
    #   --overlay-jpg 改 JPEG q90/4:4:4 —— 40ms（快 47 倍、体积 2.3MB）。
    #     overlay 是给人看的，JPEG 画质足够；要无损再留 PNG。
    ov = make_overlay(img, masks, res["check_box"])
    ov_path = out_dir / f"{p.stem}_overlay.{overlay_fmt}"
    if overlay_fmt == "jpg":
        Image.fromarray(ov).save(ov_path, quality=90, subsampling=0)
    else:
        Image.fromarray(ov).save(ov_path, compress_level=1)
    if save_mask:
        Image.fromarray((res["mask_counted"].astype(np.uint8) * 255)).save(
            out_dir / f"{p.stem}_mask.png", compress_level=1)

    # ---- 导出 RSML（每条折线一个 plant，不再分主根/侧根） ----
    rsml_path = write_rsml(out_dir / f"{p.stem}.rsml", file_key=p.stem,
                           polylines=st["paths"])
    return row, (f"{p.name}: 根数 {count} | 总长 {total:.1f} px | 根面积 {root_area} px² | "
                 f"各根长 {len_str[:60]}{'…' if len(len_str) > 60 else ''}\n"
                 f"    检查范围 {'已识别' if res['check_ok'] else '未识别(全图统计)'} | "
                 f"起点已锚定到茎 {st['anchored_count']}/{count} 条 | 已保存: "
                 f"{ov_path.name} + .rsml"
                 + (f" + {p.stem}_mask.png" if save_mask else ""))


def main():
    model_arg, folder_arg, mm_arg, size_arg, save_mask, overlay_fmt, jobs = parse_argv()
    if not folder_arg:
        print(__doc__)
        sys.exit(1)
    mm_per_px = config.MM_PER_PX if mm_arg is None else mm_arg

    img_dir = Path(folder_arg)
    if not img_dir.is_dir():
        print(f"[错误] 目标图片文件夹不存在: {img_dir}")
        sys.exit(1)

    pths, names = ckpt.resolve_pths(model_arg)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metas = ckpt.load_models(pths, device)     # 集成时 model 是模型列表
    meta = metas[0]
    # 切片训练的模型走**原始分辨率滑窗**（见 ckpt.infer_tile 的实测对比）；0 = 老路径
    tile = ckpt.infer_tile(metas, size_arg)
    tag = f"集成 {len(names)} 个" if len(names) > 1 else "模型"
    print(f"{tag}: {' + '.join(names)} | 设备: {device} | 输出 {meta['out_ch']} 通道"
          + (f" (epoch {meta['epoch']})" if meta.get("epoch") else ""))
    if tile:
        size = tile
        print(f"[切片模型] **原始分辨率滑窗**推理，块边长 {tile}（整图不缩放）")
    else:
        # 输入尺寸必须与训练时一致（实测：1024 训的模型用 2048 推理，总长误差从 4278px 涨到 9675px）
        size = size_arg or meta.get("size") or config.MAX_SIDE
        if size_arg is None and meta.get("size"):
            print(f"输入长边 {size}（用模型训练时的设置）")
        elif meta.get("size") and size_arg != meta.get("size"):
            print(f"[警告] 输入长边 {size_arg} 与模型训练时（{meta['size']}）不一致："
                  f"尺度不匹配会明显掉精度，建议按训练尺度跑")

    imgs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in config.IMAGE_EXTS)
    if not imgs:
        print(f"[错误] 目标文件夹里没有图片(支持 {sorted(config.IMAGE_EXTS)})")
        sys.exit(1)
    stems = [p.stem for p in imgs]
    dup = sorted({s for s in stems if stems.count(s) > 1})
    if dup:
        print(f"[错误] 文件夹里有同名不同扩展的图片 {dup}，输出文件会互相覆盖，"
              f"请先改名或分开处理")
        sys.exit(1)
    print(f"图片 {len(imgs)} 张: {img_dir}")

    # ---- 结果目录 result/{目标文件夹名}，重名追加 -1… ----
    # create_unique_dir 而非 unique_path：两个推理进程（一张卡一个）同时启动时，
    # 「先查存在、再 mkdir」会双双拿到同一个名字，其中一个 mkdir(exist_ok=False) 直接崩。
    # 同 train.py 用它的理由。目录名唯一，里面的 csv 就不必再去重了。
    out_dir = naming.create_unique_dir(config.RESULT_DIR, img_dir.name)
    csv_path = out_dir / f"{img_dir.name}.csv"
    mm = mm_per_px if mm_per_px and mm_per_px > 0 else 0.0

    header = ["图片名", "根数量", "起点锚定(条)", "总根长(px)", "总根系面积(px²)",
              "平均根长(px)", "最长根(px)", "各根长度(px)", "茎面积(px²)",
              "检查区面积(px²)", "check_ok", "root_ok"]
    if mm:
        header += ["总根长(mm)", "总根系面积(mm²)", "平均根长(mm)", "最长根(mm)"]

    t_start = time.time()
    # 每张图输出：_overlay + .rsml（--save-mask 时再加 _mask.png）
    n_out = 3 if save_mask else 2

    # ---- 逐张推理（--jobs >1 时并发；CSV 与打印仍按图片顺序）----
    jobs = max(1, int(jobs))
    gpu_lock = threading.Lock()
    rows_out = [None] * len(imgs)

    def work(i):
        return process_one(imgs[i], model, size, device, out_dir, mm,
                           overlay_fmt, save_mask, gpu_lock, tile=tile)

    if jobs == 1:
        for k in range(len(imgs)):
            row, log = work(k)
            rows_out[k] = row
            print(f"[{k + 1}/{len(imgs)}] {log}", flush=True)
    else:
        print(f"并发 {jobs} 张：GPU 前向串行、其余步骤并行（CSV 仍按图片顺序写）")
        done = 0
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(work, k): k for k in range(len(imgs))}
            for fut in as_completed(futs):
                k = futs[fut]
                row, log = fut.result()
                rows_out[k] = row
                done += 1
                print(f"[{done}/{len(imgs)}] {log}", flush=True)

    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows(rows_out)
        f.write(f"# 根系统计范围：模型识别出的检查范围（check_background），范围外不计入\n")
        f.write(f"# 起点锚定：每条预测折线的起点已补到茎边界，补回的那一段计入根长"
                f"（与标注口径一致）；「起点锚定(条)」是成功锚定的条数\n")
        f.write(f"# 单位：px（像素）；" + (f"mm 列按 1 px = {mm} mm 换算\n"
                                        if mm else "未做 mm 换算（--mm-per-px 关闭）\n"))
        f.write("# 总根系面积 = 统计口径的根系掩码像素数（已限定在检查范围内）。"
                "注意它比根长脆弱：根长几乎不受线宽/阈值影响，面积随阈值**线性**变化"
                "（换个阈值能差一倍）—— 适合同一条流水线内相对比较，别跨版本比绝对值\n")
        f.write(f"# 参数：模型 {' + '.join(names)}，输入长边 {size}，"
                f"低阈值 {config.PRED_LOW_THRESHOLD}，剪枝 {config.PRED_SPUR_LENGTH}px，"
                f"最短根 {config.MIN_ROOT_LENGTH}px，"
                f"锚定阈值 {config.STEM_ANCHOR_FACTOR}×茎半径"
                f"（{config.STEM_ANCHOR_MIN_PX:.0f}~{config.STEM_ANCHOR_MAX_PX:.0f}px）\n")
        f.write(f"# 共 {len(imgs)} 张图，推理耗时 {time.time() - t_start:.1f}s\n")

    el = time.time() - t_start
    print(f"\n推理完成，总耗时 {el:.1f}s | 平均 {el / len(imgs):.2f}s/张")
    print(f"结果目录: {out_dir}")
    print(f"结果文件: {csv_path}")
    print(f"已保存文件: 每张图 {n_out} 个（图片名_overlay.{overlay_fmt} + .rsml"
          + (" + _mask.png" if save_mask else "") + f"），共 {len(imgs) * n_out} 个")


if __name__ == "__main__":
    main()
