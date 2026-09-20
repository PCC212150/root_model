# 数据集划分工具（train / test / val）

目的：把一个**扁平**的数据集文件夹按比例划分为 train / test / val，并**输出成项目要的嵌套布局**
（2026-09-17 起）。

   源（扁平）                          输出（嵌套）
   root/                               root/train/images/
   ├── plant_ S062-1_xxx.jpg      →    root/train/labels/roots/
   ├── plant_ S062-1_xxx.rsml          root/train/labels/other/
   ├── plant_ S062-1_xxx.json          root/test/…（同构）
   └── …

也就是说「扁平 → 嵌套」这一步转换直接在划分里做掉了，不需要另外的转换脚本。
想要过去那种所有文件平铺在一个 split 目录里的输出，加 `--flat`。

**划分单位默认是「植株」而不是单个文件**：同一植株的多个时点整株进同一侧
（`plant_ S062-1_20251116ST` 和 `..._20251126ST` 是同一株的两个时点，分到两边就是同株泄漏）。
植株识别用 [common/dataset.py](../../common/dataset.py) 的 `plant_key`。想按单个文件划分加 `--by-file`。

无论按哪种单位，**同名的图片/rsml/json 永远在一组、不会被拆散** —— 拆散会导致训练时
整组数据被丢掉（`discover_pairs` 要求图片与同名 `.rsml` 在同一侧）。

## 运行环境

依赖项目内的 `common/naming.py`（重名加 `-1` 的项目规范）和 `config.py`（读取默认随机种子、图片扩展名），
所以要在本项目里运行，用 Anaconda 环境 `pcc` 即可（`conda activate pcc`）。划分本身只用标准库，不额外装包。

## 使用方法

1）**先预览**（推荐，确认哪些组去了哪一份）：

```
cd tool\separate_dataset
python separate_dataset.py --dir "C:\Users\21215\Desktop\RootTracer_RSML\20251116ST" --dry-run
```

2）**确认后正式执行**（去掉 `--dry-run`）：

```
python separate_dataset.py --dir "C:\Users\21215\Desktop\RootTracer_RSML\20251116ST"
python separate_dataset.py --dir "C:\Users\21215\Desktop\RootTracer_RSML\20251116ST" --test 0.3
python separate_dataset.py --dir "C:\Users\21215\Desktop\RootTracer_RSML\20251116ST" --train 0.7 --test 0.2 --val 0.1
```

也可以不切目录，在项目根目录下运行：

```
python tool\separate_dataset\separate_dataset.py --dir "C:\Users\21215\Desktop\RootTracer_RSML\20251116ST"
```

## 参数

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `--dir` | 是 | 源数据集文件夹。路径含空格/中文时用引号包住 |
| `--train` | 否 | 训练集比例，默认 0.8 |
| `--test` | 否 | 测试集比例，默认 0.2 |
| `--val` | 否 | 验证集比例，**默认 0（不分配，也不会建空的 val 文件夹）** |
| `--out` | 否 | 输出目录，默认 `<源文件夹同级>/<源文件夹名>_split`（重名自动加 `-1`） |
| `--seed` | 否 | 随机种子，默认 42（与 [config.py](../../config.py) 的 `SEED` 一致） |
| `--move` | 否 | 移动文件而不是复制（默认复制，源数据保留） |
| `--dry-run` | 否 | 只预览划分结果，不写任何文件 |

## 划分规则

- **比例可以只写一部分**，没写的那项按剩余自动推算：

  | 命令 | 实际划分 |
  | --- | --- |
  | 不带比例参数 | train 0.8 / test 0.2 / val 0 |
  | `--test 0.3` | train 0.7 / test 0.3 / val 0 |
  | `--train 0.6` | train 0.6 / test 0.4 / val 0 |
  | `--train 0.7 --val 0.1` | train 0.7 / test 0.2 / val 0.1 |

  `--train` 和 `--test` 都写时必须和为 1，写错直接报错，不会悄悄改数。
- **组数取整**：按比例四舍五入算出 test/val 的组数，剩下的全部给 train，保证三份之和等于总组数、不丢不重。
- **划分可复现**：先按名称排序再按种子打乱，所以「同一份数据 + 同一种子」每次划分结果完全一样。
  想换一种划分就改 `--seed`。
- **只处理当前层**：源文件夹里的子文件夹不动（与训练代码只读当前层一致），会在运行时提示。
- **系统文件**忽略不参与划分：`.DS_Store`、`Thumbs.db`、`desktop.ini`。
- **配对检查**：运行时会统计「图片+rsml 齐全 / 缺标注 / 缺图片」各多少组，不成对时给出提示
  （缺标注的图片在训练时会被跳过）。

## 输出

```
<源文件夹同级>\<源文件夹名>_split\
├── train\                # png 与同名 rsml 成对放一起
├── test\
├── val\                  # 仅当 --val > 0 时创建
└── split.txt             # 本次划分记录：比例、随机种子、时间、每份的组名清单
```

`split.txt` 留存了这次的划分依据，之后想核对「某张图当时分到哪一份」直接看它；
重新划分时它会跟着新的输出目录一起生成，旧的一份不会被覆盖（输出目录重名自动加 `-1`）。

## 输出示例

```
源文件夹：C:\Users\21215\Desktop\RootTracer_RSML\20251116ST
共 11 组 / 21 个文件（图片+rsml 齐全 10 组，缺标注 0 组，缺图片 1 组，其他 0 组）
比例：train 0.8 / test 0.2 / val 0    随机种子：42
  train：9 组（18 个文件）  plant_ S07-1、plant_ S03-1、…
  test：2 组（3 个文件）  plant_ S010-1、plant_ S099-1
  val：不分配（比例 0）

完成：复制 21 个文件到 C:\Users\21215\Desktop\RootTracer_RSML\20251116ST_split
划分记录：C:\Users\21215\Desktop\RootTracer_RSML\20251116ST_split\split.txt
```

## 接进训练流程

划分完源数据没有被改动（默认复制）。要让训练用上，把两份数据放到项目约定的位置：

```
copy <输出目录>\train\*   datasets\root\train\
copy <输出目录>\test\*    datasets\root\test\
```

即 [config.py](../../config.py) 里的 `TRAIN_DATA_DIR` / `TEST_DATA_DIR`。

> **注意：本项目的数据集目录已经改成「图片与标注分开放」**（本工具的输出仍是混放，
> 与本项目当前布局不同）：
> ```
> datasets\root\train\
> ├── images\            # 图片
> └── labels\
>     ├── roots\         # 根系 rsml
>     └── other\         # 茎/检查范围的 labelme json
> ```
> 拷进来后需要按扩展名分别归位（图片放 `images\`，`.rsml` 放 `labels\roots\`，
> `.json` 放 `labels\other\`）。训练代码对旧布局（混放）仍能读，只是标注
> `labels/other` 那两类会缺失。

## 常见问题

- **`[错误] 文件夹不存在`**：`--dir` 路径写错或不是文件夹；路径含空格/中文时记得加引号。
- **`[错误] 比例之和必须为 1`**：`--train` 和 `--test` 都写了但加起来不是 1。只写一项即可自动推算。
- **`[错误] 比例超出 1`**：比例加起来大于 1（例如 `--val 1.2`），调小即可。
- **`[错误] 输出目录已存在且不为空`**：用了 `--out` 指定一个已存在、**里面还有文件**的目录。
  换个名字，或先清空它，避免新旧划分结果混在一起。
  **空目录不拦**（重导数据集时的常见做法就是先把 `datasets\root` 清空再重跑），会提示一句直接复用；
  不指定 `--out` 时输出目录重名会自动加 `-1`，也不会有这个问题。
- **`[提示] ... 但数据太少 ... 实际分到 0 组`**：组数太少，按比例算下来这份分不到数据。
  增加数据量，或调大该份的比例。
- **想重新划分**：删掉输出目录再跑一次即可，源数据不受影响（除非用了 `--move`）。
