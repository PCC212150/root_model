"""把本项目数据集导出成 nnU-Net v2 的 raw 格式（root / stem / check 各一个二分类数据集）。

**为什么拆成三个数据集**：nnU-Net 默认是 softmax 多类分割，要求类别互斥；本项目的
root / stem / check 是**互相重叠**的三个通道（茎也在检查范围内），不能塞进一个
label map。所以每个通道单独训练一个二分类 nnU-Net，最后按通道拼回三通道预测。

用法（在项目根目录、装了 torch 的环境里跑即可，不需要 nnunetv2）：

    python tool/nnunet/export_dataset.py --out nnUNet_raw \
        --src datasets/root/train --test-src datasets/root/test/GT

输出：

    nnUNet_raw/Dataset501_Root/{imagesTr,labelsTr,imagesTs,dataset.json,case_map.json}
    nnUNet_raw/Dataset502_Stem/...
    nnUNet_raw/Dataset503_Check/...

- 训练集默认用 `datasets/root/train` 下**全部**样本（GT + scr）；
  `--exclude-val` 可排除 config.VAL_PLANTS 对应的验证植株（与当前训练脚本同口径）；
  `--gt-only` 只用 GT 来源，不导 scr 伪标签。
- 每个通道只在 `valid[c] > 0` 的样本上导出（缺 stem/check 标注的图会从该数据集剔除）。
- root 用 `build_target_masks` 的同一口径：GT 有 masks/*.png 时用像素掩码，
  scr 用折线栅格化（config.MASK_LINE_WIDTH）。
- 测试集只导出图像（imagesTs），标签在项目里，用 eval_predictions.py 评测。
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from common import image_io  # noqa: E402
from common.dataset import (CH_CHECK, CH_ROOT, CH_STEM,  # noqa: E402
                            build_target_masks, discover_pairs, load_annot,
                            source_of)
from common.naming import plant_key  # noqa: E402

CHANNELS = (
    ("root", CH_ROOT, "Root"),
    ("stem", CH_STEM, "Stem"),
    ("check", CH_CHECK, "Check"),
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, type=Path,
                   help="nnUNet_raw 根目录（不存在会创建）")
    p.add_argument("--src", type=Path, default=config.TRAIN_DATA_DIR,
                   help="训练集目录（默认 datasets/root/train）")
    p.add_argument("--test-src", type=Path, default=config.TEST_DATA_DIR / "GT",
                   help="测试集目录（默认 datasets/root/test/GT）")
    p.add_argument("--base-id", type=int, default=501,
                   help="数据集编号起点：root=501, stem=502, check=503（默认 501）")
    p.add_argument("--gt-only", action="store_true",
                   help="只导出 GT 来源，不导 scr 伪标签")
    p.add_argument("--exclude-val", action="store_true",
                   help="排除 config.VAL_PLANTS 的验证植株（与训练脚本同口径）")
    p.add_argument("--no-test", action="store_true", help="不导出 imagesTs")
    p.add_argument("--limit", type=int, default=0,
                   help="只导出前 N 组训练样本（调试用；0=全部）")
    return p.parse_args()


def _split_names(names, val_size, seed):
    """与 train/train.py 的 split_by_plant 同逻辑；返回 train_names。"""
    groups = {}
    for n in names:
        groups.setdefault(plant_key(n), []).append(n)
    keys = sorted(groups)
    pinned = [k for k in keys if k in set(config.VAL_PLANTS)] if config.VAL_PLANTS else []
    if pinned:
        val_plants = pinned
    else:
        rng = np.random.RandomState(seed)
        rng.shuffle(keys)
        n = min(max(val_size, 0), max(len(keys) - 1, 0))
        val_plants = sorted(keys[:n])
    val_set = set(val_plants)
    return sorted(n for k, v in groups.items() if k not in val_set for n in v)


def _case_names(names):
    """原始文件名 -> nnU-Net 安全 case 名；保留可读性并保证唯一。"""
    used = {}
    mapping = {}
    for name in names:
        base = re.sub(r"[^A-Za-z0-9_-]", "_", name)
        base = re.sub(r"_+", "_", base).strip("_") or "case"
        case = base
        if case in used:
            used[base] += 1
            case = f"{base}_{used[base]}"
        else:
            used[base] = 1
        mapping[case] = name
    return mapping


def _save_image(img, path):
    Image.fromarray(img).save(path, format="PNG", optimize=False)


def _save_label(mask, path):
    Image.fromarray((mask.astype(np.uint8) * 1)).save(path, format="PNG")


def export_channel(args, ds_id, ch_name, ch_idx, train_pairs, test_pairs,
                   train_map, test_map):
    ds_dir = args.out / f"Dataset{ds_id:03d}_{CHANNELS[ch_idx][2]}"
    images_tr = ds_dir / "imagesTr"
    labels_tr = ds_dir / "labelsTr"
    images_ts = ds_dir / "imagesTs"
    for d in (images_tr, labels_tr, images_ts):
        d.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    for case, name, img_path in train_pairs:
        img = image_io.load_rgb(img_path)
        h0, w0 = img.shape[:2]
        annot = load_annot(args.src, name, (w0, h0), verbose=False)
        masks, valid = build_target_masks(annot, (w0, h0), config.MASK_LINE_WIDTH)
        if valid[ch_idx] < 0.5:
            n_skipped += 1
            continue
        _save_image(img, images_tr / f"{case}_0000.png")
        _save_label(masks[:, :, ch_idx], labels_tr / f"{case}.png")
        n_written += 1

    n_test = 0
    if not args.no_test:
        for case, name, img_path in test_pairs:
            img = image_io.load_rgb(img_path)
            _save_image(img, images_ts / f"{case}_0000.png")
            n_test += 1

    if n_written == 0:
        sys.exit(f"[错误] Dataset{ds_id:03d}_{CHANNELS[ch_idx][2]} 没有可导出的样本"
                 f"（全部缺 {ch_name} 标注）")

    dataset_json = {
        "channel_names": {"0": "R", "1": "G", "2": "B"},
        "labels": {"background": 0, ch_name: 1},
        "numTraining": n_written,
        "file_ending": ".png",
    }
    (ds_dir / "dataset.json").write_text(
        json.dumps(dataset_json, indent=2, ensure_ascii=False), encoding="utf-8")
    (ds_dir / "case_map.json").write_text(
        json.dumps({"train": train_map, "test": test_map}, indent=2,
                   ensure_ascii=False), encoding="utf-8")
    print(f"[{ds_dir.name}] 训练 {n_written} 张（跳过缺 {ch_name} 标注 {n_skipped} 张）"
          f" | 测试图 {n_test} 张")


def main():
    args = parse_args()
    pairs = discover_pairs(args.src)
    if not pairs:
        sys.exit(f"[错误] {args.src} 里没有样本")
    if args.gt_only:
        # 扁平布局没有来源子目录时 source_of 返回 ""，也按 GT 处理
        pairs = [p for p in pairs if source_of(args.src, p[1]) != "scr"]
        print(f"[gt-only] 去掉 scr 后：{len(pairs)} 组")
    if args.exclude_val:
        train_names = set(_split_names([p[0] for p in pairs],
                                       config.VAL_SIZE, config.SEED))
        pairs = [p for p in pairs if p[0] in train_names]
        print(f"[exclude-val] 排除验证植株后：{len(pairs)} 组")

    pairs = sorted(pairs, key=lambda p: (plant_key(p[0]), p[0]))
    if args.limit:
        pairs = pairs[:args.limit]
    train_map = _case_names([p[0] for p in pairs])
    path_by_name = {name: img_path for name, img_path, _ in pairs}
    train_pairs = [(case, name, path_by_name[name])
                   for case, name in train_map.items()]

    test_pairs = []
    test_map = {}
    if not args.no_test:
        tp = sorted(discover_pairs(args.test_src))
        if args.limit:
            tp = tp[:args.limit]
        test_map = _case_names([n for n, _, _ in tp])
        test_path = {name: img_path for name, img_path, _ in tp}
        test_pairs = [(case, name, test_path[name])
                      for case, name in test_map.items()]

    args.out.mkdir(parents=True, exist_ok=True)
    for i, (ch_name, _ch_idx, _label) in enumerate(CHANNELS):
        export_channel(args, args.base_id + i, ch_name, i,
                       train_pairs, test_pairs, train_map, test_map)

    print(f"\n完成：{args.out}")
    print("下一步见 tool/nnunet/README.md")


if __name__ == "__main__":
    main()
