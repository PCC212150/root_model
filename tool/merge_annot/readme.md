# 把 rsml 的根系折线并进 labelme json

标注格式统一：**一个 labelme json 装三个通道**。本工具把 `<名>.rsml` 里的根系折线，
作为 `label: "root"` + `shape_type: "linestrip"` 追加到同名 json 的 `shapes` 数组。

```
改前  C001-1.json   [check_background, stem]                     +  C001-1.rsml
改后  C001-1.json   [check_background, stem, root, root, ...]
```

## 为什么并

两条并存会出**静默的错**：训练管线的配对键是老格式的 `.rsml`
（[common/dataset.py](../../common/dataset.py) 的 `discover_pairs`），
新标注的图没有 rsml，会被**直接跳过、不报错** —— 整批新数据导进来只会得到空数据集。

## 为什么是文本插入，不是 json 往返

`json.load` → `json.dump` 会把**整个文件重写一遍**，建出一个巨大的假 diff，
也没法证明「只动了这一处」。标注数据不该冒这个险
（同 [tool/repair_json](../repair_json/readme.md)、[tool/repair_rsml](../repair_rsml/readme.md)）。

本工具只在文本层面往 `"shapes"` 数组里插入新对象，**其余字节原样保留**，
并且写盘前必须通过这条自检：

```
新文本抠掉插入的那一段  ==  原文        ← 逐字节相同
```

它一次性覆盖了「其余顶层键、CRLF 换行、base64 全都没变」，比任何字段级比对都硬。

顺带一提：定位 `"shapes"` 用的是**字符串感知的扫描器**，不是正则 ——
正则排不掉「字符串里的」和「嵌套对象里的」同名键，而这个文件里恰好两者都有
（`imageData` 的 base64、形状对象里可能出现的同名字段）。

## 用法

在项目根目录下运行：

```
python tool\merge_annot\merge_annot.py --dir "D:\目标文件夹" -r --dry-run          # 先预览
python tool\merge_annot\merge_annot.py --dir "D:\目标文件夹" -r --backup "D:\_bak" # 正式
python tool\merge_annot\merge_annot.py --dir "D:\目标文件夹" --only "root_C001-1*" # 只转一个试试
python tool\merge_annot\merge_annot.py --dir "D:\数据集\train\labels\other"        # 项目布局，零参数
```

| 参数 | 说明 |
| --- | --- |
| `--dir` | 目标文件夹（**必填**）。rsml 默认与 json 同目录；**自动识别项目布局** `labels/other/x.json` → `labels/roots/x.rsml` |
| `-r` / `--recursive` | 递归子文件夹 |
| `--rsml-dir` | 显式指定 rsml 目录，覆盖自动推断 |
| `--only` | 只处理主干名匹配通配符的文件；**全量跑之前先用它转一个核对** |
| `--backup DIR` | 改前把原 json 按相对路径拷到 DIR。**必须在数据目录之外**（否则会被 `separate_dataset` 当数据卷进划分） |
| `--dry-run` | 只打印将要插入的内容，不写文件 |

## 判定与处理口径

| 情况 | 处理 |
| --- | --- |
| json 里已有 root 形状 | ✅ 跳过（**幂等**，重跑零改动） |
| rsml 有 0 条根 | ✅ 跳过。这是**合法的「这张图没有根」负样本**，不是错误 |
| 找不到同名 rsml | ✅ 跳过，只统计不刷屏（未标注的图很多） |
| 只有 rsml、没有 json | ⚠️ **只报告，跳过**。不发明标注文件 |
| 有 BOM / 换行混用 / 不是合法 json | ❌ 拒绝改（**不猜**） |
| 顶层 `"shapes"` 找不到、缩进不是 2 空格 | ❌ 拒绝改 |
| 自检未通过 | ❌ 不写盘 |

## 数据不会变

折线**只来自 `parse_rsml`**，不自己走一遍 XML。这样「json 里的 root 集合」**按构造**
就等于旧加载器看到的那个集合，`<2` 点丢弃之类的行为自动一致。

坐标走的是 `float(属性)` → `json.dumps` → 文本 → `json.loads`，**精确往返**：
Python 的 json 编码器用 `float.__repr__`，最短且能还原同一个 IEEE-754 double。
所以画出来的掩码与原来**逐像素相同**。

> ⚠️ **唯一能毁掉这一步的是自己格式化数字**：`round(x, 2)`、`f"{x:.6f}"`，
> 或者「保留原文」把 `x="1749"` 照抄成 `1749`。都别做 —— 解析成 float 再重新 repr，
> `1749 → 1749.0` 只是观感差异，语义完全相同。

## 跑完怎么核对

用 [tool/check_convert](../check_convert/readme.md)：

```
python tool\check_convert\check_convert.py --dir "D:\目标文件夹" -r --capture before.json   # 转换前
python tool\check_convert\check_convert.py --dir "D:\目标文件夹" -r --compare before.json   # 转换后
```

## 输出

改完之后在目标文件夹里生成 `merge_annot_log.txt`，逐条记
「相对路径 / 插入折线数 / 插入点数 / 增长字节 / 原 sha256 / 新 sha256」。

> ⚠️ **日志别留在数据文件夹里**：这个文件夹接下来若要去划数据集
> （[tool/separate_dataset](../separate_dataset/readme.md)），先把它挪走 ——
> 划分工具是按「文件夹里所有文件」分组的，一个 `.txt` 会被当成一组文件划进 train/test。

**`.rsml` 一个都不删。** 它是这批数据的回退点，也是加载器的兜底；
真想清掉就等核对通过之后单独做。

## 实测（2026-09-30，桌面 `数据集总表\root`）

```
找到 75 个 .json
  已合并              72     （C 32 + P 5 + S 35）
  rsml 无根（跳过）     1     （plant_S003-3，合法负样本）
  已有 root 标注（跳过） 2     （抽出来的图片\C 里先前手标的那 2 个）
合计新增 2174 条折线
```

跑完用 `check_convert` 核对：**75 组逐项相同**（折线数 / 点数 / 坐标哈希 /
原图分辨率掩码哈希 / 图片与 json 尺寸 / 顶层键与键序 / stem 与 check 的条数），
74 组如期从 rsml 翻到 json。重跑一次幂等，零字节改动。
