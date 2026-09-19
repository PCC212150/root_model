# 参数扫描工具（根数 / 根长 统计）

目的：在**带 RSML 真值**的数据集上，把「掩码 → 骨架 → 分链 → 根数/总长统计」这一段的参数
扫一遍，按与真值的误差挑最优组合，避免靠手改 [config.py](../../config.py) 一个个试。

扫的维度：

| 维度 | 对应 config | 说明 |
| --- | --- | --- |
| `--low` | `PRED_LOW_THRESHOLD` | 滞回阈值低阈值（高阈值固定 0.5），把细弱处断开的段接回来 |
| `--spur` | `PRED_SPUR_LENGTH` | 骨架剪枝长度：短于此长度的末梢当噪声去掉 |
| `--min-len` | `MIN_ROOT_LENGTH` | 独立骨架段短于此长度直接忽略 |
| `--erode` | `erode_iters`（代码里写死为 1） | 掩码腐蚀次数，消除线宽厚度伪影 |
| `--normalize-count` | `normalize_count`（函数参数） | 「计数归一」：把过碎的短轨迹拼回现有轨迹。打开时根数更稳，但会把互相贴着的根并成一根，两种口径都值得看 |

统计口径与 [test/test.py](../../test/test.py)、[inference.py](../../inference.py) **完全一致**：
根总数/总长来自 [common/skeleton_stats.py](../../common/skeleton_stats.py)，掩码本身是
「根系概率 ∩ 模型识别出的检查范围」（沿用 `predict.predict` 的同一条流水线），
所以这里扫出来的最优参数就是部署时实际生效的参数。

## 运行环境

用 Anaconda 虚拟环境 **pcc**（`conda activate pcc`），依赖 torch / numpy / pillow / scikit-image
（与主项目相同，无新增依赖）。**必须在项目根目录**下运行（脚本按项目结构找 `common/` 与 `config.py`）。

## 使用方法

1）**先看组合数和预计耗时**（不推理、不写文件）：

```
python tool\tune_stats\tune_stats.py --dir datasets\root\train --dry-run
```

2）**快速试两张**（确认没报错、看看数字量级）：

```
python tool\tune_stats\tune_stats.py --dir datasets\root\train --limit 2 --low 0.05,0.10 --spur 20,60
```

3）**正式扫**（默认网格 low×spur×min_len×normalize = 3×4×1×2 = 24 组）：

```
python tool\tune_stats\tune_stats.py --dir datasets\root\train
python tool\tune_stats\tune_stats.py --dir datasets\root\train --spur 15,20,30,45,60 --min-len 15,20,30
```

4）**上限实验**：拿真值折线画的掩码当输入，测「后处理的天花板」，与模型质量无关：

```
python tool\tune_stats\tune_stats.py --dir datasets\root\test --gt-mask
```

> 扫描请在**训练集**上做，测试集只看不调（否则等于拿测试集调参）。

## 参数

| 参数 | 必填 | 默认 | 说明 |
| --- | --- | --- | --- |
| `--dir` | 是 | — | 数据集目录（含 `images/` 与 `labels/roots/`），路径含空格要加引号 |
| `--model` | 否 | 最新 | 模型文件夹名（可省 `model_` 前缀），规则同 test.py |
| `--low` | 否 | `0.05,0.10,0.20` | 滞回低阈值候选 |
| `--spur` | 否 | `20,30,45,60` | 骨架剪枝长度候选(px) |
| `--min-len` | 否 | `20` | 最短根长候选(px) |
| `--erode` | 否 | `1` | 腐蚀次数候选 |
| `--normalize-count` | 否 | `both` | `on` / `off` / `both`（两种口径都扫） |
| `--anchor-min` | 否 | `config.STEM_ANCHOR_MIN_PX` | 锚定阈值下限候选(px，原图尺度) |
| `--anchor-max` | 否 | `config.STEM_ANCHOR_MAX_PX` | 锚定阈值上限候选(px，原图尺度)。**实测这个才是真正生效的那个** |
| `--limit` | 否 | 0=全部 | 只跑前 N 张（按文件名排序） |
| `--gt-mask` | 否 | 关 | 用真值掩码代替模型输出（上限实验，不需模型） |
| `--size` | 否 | `config.MAX_SIDE` | 模型输入长边像素 |
| `--top` | 否 | 15 | 控制台打印前 N 名（**表格文件里写全**） |
| `--sort` | 否 | `score` | 排序依据：`score` / `total` / `total_len` |
| `--out` | 否 | `result/tune_stats` | 结果目录（重名自动加 `-1`） |
| `--dry-run` | 否 | 关 | 只打印组合数与预计耗时 |
| `--cpu` | 否 | 关 | 强制 CPU |

`-h` 可看帮助。

## 行为说明

- **一张图只推理一次**：概率图整张前向一次，多组 `--low` 复用它（滞回阈值在概率图上做）。
- **骨架只算一次**：`掩码 → 骨架 → 邻接表` 与 `spur/min_len` 无关，所以每个
  `(low, erode)` 只算一次，多组阈值复用（`_strands_from_adj` 内部拷贝邻接表，不会改坏缓存）。
  这也是快的原因：不复用的话每组合都要重新骨架化。
- **等价性自检**：脚本开跑后会拿第一张图的第一个组合，用
  `analyze_mask_ex` 重新算一遍并断言结果一致；不一致直接报错退出（防止「复用」悄悄改变口径）。
- **评分**：`score = 根总数MAE + 总长MAE/100`，即先把「根数」口径对齐，长度只做量级约束。
  表里两个 MAE 原始值都列出来，方便自行取舍。
- **必须在同一份数据上比较**：换了数据集（比如新标注批次）要重新扫。

## 输出

```
result\tune_stats\
├── tune_stats_YYYYMMDDHHMM.txt    # 排名表（所有组合）+ 最优组合 + 与当前 config 的对比
└── tune_stats_YYYYMMDDHHMM.json   # 最优组合（可直接抄进 config.py）
```

txt 示例（列省略）：

```
# 数据集: datasets\root\train   图片数 25
# 模型: model_202609130020 | 设备: cuda | 输入长边 1024
# 口径: 根系只在模型识别出的检查范围内统计（与 test.py / inference.py 一致）
# 排序: score | 综合分 = 根总数MAE + 总长MAE/100
    1  0.10     30       20      1         on |    3.42   812.40 |   11.54
# 最优组合: --low 0.1 --spur 30 --min-len 20 --erode 1 --normalize-count on
# 与当前 config（low=0.10, spur=60, min_len=20, normalize=on）对比: 根总数MAE 5.20 -> 3.42, ...
# 耗时: 总 210.5s | 单图 8.77s | 组合平均 2.338s
```

## 锚定那一维：`STEM_ANCHOR_FACTOR` 实际是失效的（2026-09-19）

锚定阈值 = `clamp(FACTOR × 茎等效半径, anchor_min, anchor_max)`，而
**本数据集茎的等效半径约 215~252px（原图尺度）**（实测 S068-4 145467px 面积 → r_eq 215.2；
S002-3 199909px → 252.3）。`FACTOR = 6` 乘出来是 **1291~1514**，
**恒被 `anchor_max = 600` 截断** —— 所以调 FACTOR 完全无效，能动的只有 `anchor_max`。

实测扫描（`model_202609190013`，固定 low=0.10 / spur=45 / normalize=on）：

| anchor_max | 训练集 29 张 总长MAE | 留出集 5 张 总长MAE |
| --- | --- | --- |
| 200 | 3152 | 1856 |
| 400 | 2240 | 1195 |
| **600（原值）** | **1546** | **682** |
| 900 | 1752 | 1658 |
| 1400 | 2745 | 2339 |

两边都是 **U 形、最低点落在 600** —— **锚定参数无需调整**，原值就是最优。
训练集上单独看会偏好 900（配合 spur=60 能到 1370），但那个组合在留出集上是 1213，
比当前配置的 682 差得多，是典型的训练集过拟合。**这也再次说明留出集复核不能省。**

## 把最优参数落到 config.py

1）打开 `result/tune_stats/tune_stats_*.json`，把 `low_thresh` / `pred_spur_length` /
`min_root_length` 抄到 [config.py](../../config.py) 的 `PRED_LOW_THRESHOLD` /
`PRED_SPUR_LENGTH` / `MIN_ROOT_LENGTH`（顺手更新上面那两行注释，写明本次是在什么数据上扫的）；

2）跑一遍 `python test\test.py` 看测试集的汇总有没有跟着变好（**扫描是在训练集上做的，
测试集只看不调**，否则等于拿测试集调参）。

## 常见问题

- **`[错误] ...下没有「图片 + rsml」配对数据`**：`--dir` 指到了没有标注的文件夹；
  统计类参数必须要有真值才能算误差。
- **`[错误] 找不到模型 / 没有 .pth`**：模型名写错，或 `model/` 下还没有训练结果；
  只想测后处理上限就加 `--gt-mask`。
- **跑太久**：组合数是各维度长度相乘，`--limit` 先试小样本、`--top` 只影响打印；
  先用 `--dry-run` 看组合数与预计耗时。
- **等价性自检报错**：说明 `common/skeleton_stats.py` 的分链逻辑被改过，脚本复用的路径与
  `analyze_mask_ex` 不一致，先确认改动是否预期。
- **`normalize=on` 和 `off` 差别很大**：正常。`on` 的分根数更稳（旧标注验证过 23 根→23），
  代价是把互相贴着或很短的根并进邻近的根；`off` 保留这些根却会把贴着的根拆碎、根数偏多。
  表里两种口径都列出来了，按你的用途取舍。
