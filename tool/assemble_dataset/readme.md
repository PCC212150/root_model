# 数据集装配工具（源文件夹 → `datasets/root/{train,test}/{GT,scr}`）

目的：把标注工具的产出**按真值口径分类**再装配成训练能直接读的目录结构。

它是 [separate_dataset](../separate_dataset/readme.md) 的姊妹工具，但管的是**另一件事**：
separate_dataset 只按比例随机切分，不认识「多边形/掩码/折线」；本工具要先把每组判成
GT 还是 scr，再决定它去哪、掩码搬不搬。

## 分类口径

| 源里的样子 | 类别 | 训练时走哪条路 |
| --- | --- | --- |
| json 里有 **root 多边形**（`shape_type=polygon`）**且**有 `masks/<名>.png` | `GT` | `masks/<名>.png` **替换** root 通道（逐像素真值） |
| 只有 root **折线**（`linestrip`/`line`）、没有掩码 | `scr` | 折线栅格化成 10px 带（`config.MASK_LINE_WIDTH`，伪标签） |

> 这不是「标得好/标得差」的分档，是**真值口径的分档**：GT 与 scr 是两把尺子
> （同一份预测换尺子，Dice 自己就会跳 0.12），所以两批必须各归各的目录，
> 不能混在一份里。判据直接对齐训练侧的两处实现 ——
> `common/dataset.py` 的 `root_polygons` 标志与 `load_root_mask_png`。

**两边对不上的组一律拦下来**（不装配任何文件），在源里改到一致再跑：

- 有多边形、没掩码：去标注工具**重新导出掩码**（或删掉多边形让它回到 scr）；
- 有掩码、没有多边形：掩码多半是上一次标注的**残留**，确认后删掉掩码。

`--force` 可以放行，放行时按规则降级（有掩码无多边形 → 当 scr 且**不搬掩码**；
有多边形无掩码 → 当 GT 但不搬掩码），并把这些组写进 `split.txt` 的反常小节。

## 划分口径

- **scr 全部进 train** —— 测试集只能有 GT（用户 2026-10-15 定的硬约束）；
- **GT 按单图随机抽 `--test-ratio` 进 test**（默认 1/3），其余进 train/GT；
- ⚠️ 按单图抽 = **同株泄漏**：同一植株的时点会分居 train/test 两侧。项目平时的规矩是
  「整株进出」（`common.naming.plant_key`），这里是**有意的例外**（与 2026-10-08 那次
  同口径，见 config.py 的划分记录）。工具会把跨侧植株逐株点名并写进 `split.txt` ——
  test/GT 上的分数因此**偏乐观，只能当「在准确尺子上看趋势」，不能当泛化性能**。

种子与比例都写进 `split.txt`，「同一份数据 + 同一种子」重跑结果完全一致。

## 运行环境

只依赖标准库 + `common/labelme.py`（不 import torch），**本地没装 torch 的机器也能跑**。
项目里的 `common/naming.py` 同理（`plant_key` 就是为了这个从 `common/dataset.py` 搬过去的）。

## 使用方法

1）**先预览**（确认 GT/scr 分类与 test 抽中的是哪几组）：

```
python tool\assemble_dataset\assemble_dataset.py --dir "C:\Users\21215\Desktop\数据集总表\root\新" --dry-run
```

2）**正式装配**。输出目录非空会被拒绝（避免新旧混在一起），重装的标准动作是**先把旧的
改名备份**再跑：

```
cd "D:\python projects\Deep_learning_model_for_sugarcane\root_model\datasets"
move root root_旧装配备份_2026-10-15

cd ..
python tool\assemble_dataset\assemble_dataset.py --dir "C:\Users\21215\Desktop\数据集总表\root\新"
```

源数据默认**复制**（保留），加 `--move` 才是移动。

## 参数

| 参数 | 必填 | 说明 |
| --- | --- | --- |
| `--dir` | 是 | 源文件夹（平铺：`<名>.jpg/.json` + `masks/`）。含空格/中文时用引号包住 |
| `--out` | 否 | 输出目录，默认 `config.ROOT_DATA_DIR`（即 `datasets/root`） |
| `--test-ratio` | 否 | GT 里抽多大比例进 test，默认 1/3 |
| `--seed` | 否 | 随机种子，默认 42（与 `config.SEED` 一致） |
| `--move` | 否 | 移动而非复制（默认复制） |
| `--force` | 否 | 反常组不再拦，按规则降级处理 |
| `--dry-run` | 否 | 只预览，不写任何文件 |

## 输出

```
datasets/root/
├── train/
│   ├── GT/    <名>.jpg  <名>.json  masks/<名>.png
│   └── scr/   <名>.jpg  <名>.json
├── test/
│   └── GT/    <名>.jpg  <名>.json  masks/<名>.png      # test 下不建 scr
└── split.txt  装配记录：分类口径、比例、种子、每组归属、跨侧植株名单
```

布局与 `common/dataset.py` 的 `source_dirs` 一致：**每个来源子目录都是一份独立完整的
扁平数据集**（图 + json + 自己的 `masks/`），所以 `masks/` 必须跟着组一起搬 ——
漏搬就会让那批图退回多边形口径，和没搬的成了两把尺子。

## 常见问题

- **[错误] 有反常组，没有装配任何文件**：源里有「多边形与掩码对不上」的组，按提示去源里
  改一致；确认无所谓再用 `--force` 放行。
- **[错误] 输出目录已存在且不为空**：先把旧的改名备份，别在里面直接重跑。
- **[警告] 跨侧植株（同株泄漏）**：按单图抽的**预期行为**，不是 bug。想避免就改用
  [separate_dataset](../separate_dataset/readme.md) 的整株进出口径（但那样 test 里就
  只能放整株都是 GT 的植株）。
- **装配完还要同步服务器**：`datasets/` 不入库（见 `.gitignore`），服务器上那份要另外传。
