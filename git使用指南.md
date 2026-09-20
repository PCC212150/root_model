# Git 使用指南（本项目的工作流）

场景：**Windows 笔记本写代码 ↔ RTX 3090 服务器跑训练**，单人维护，直接推 `main`。

仓库：`https://github.com/PCC212150/Deep_learning_model_for_sugarcane.git`

0.镜像克隆：git clone https://gh-proxy.com/https://github.com/PCC212150/Deep_learning_model_for_sugarcane.git

1.


---

## 一、先搞清楚：什么进 Git，什么不进

| 内容 | 进 Git？ | 同步方式 |
| --- | --- | --- |
| 代码 `*.py`、`config.py`、`readme.md` | ✅ 进 | `git push` / `git pull` |
| `datasets/` 数据集 | ❌ 不进 | **手动**（U 盘 / rsync / 网盘） |
| `model/` 模型权重 | ❌ 不进 | 手动；每个 `.pth` 119MB，**超 GitHub 单文件 100MB 硬限制** |
| `result/`、`inference_pictures/` | ❌ 不进 | 手动 |
| `experimental_reports/` 实验报告 | ❌ 不进 | 研究产物，不进代码仓库（含插图与归档） |
| `.claude/`、`.vscode/` | ❌ 不进 | 不需要同步 |

> **一句话**：Git 只管代码。数据集和模型永远走别的通道。
> 规则都写在 [.gitignore](.gitignore) 里，别去改它。

---

## 二、日常改代码（笔记本上）

```bash
# 1. 看改了什么
git status
git diff

# 2. 挑要提交的文件加进暂存区（别闭眼 git add -A）
git add config.py train/train.py common/naming.py

# 3. 提交（说明写清楚"为什么改"，不是"改了啥"）
git commit -m "修复茎通道欠分割：stem Dice权重 0.5→1.0，加 pos_weight 10"

# 4. 推之前先看远端有没有新东西
git fetch origin
git log --oneline HEAD..origin/main     # 远端比本地多的提交（有输出=要先处理）
git log --oneline origin/main..HEAD     # 本地比远端多的提交（有输出=你有东西要推）

# 5. 推
git push origin main
```

**常用查看命令**

```bash
git log --oneline -10          # 最近 10 条提交
git show <提交号>              # 看某次提交改了什么
git diff HEAD~1 --stat         # 上一次提交动了哪些文件
git restore <文件>             # 撤销工作区里对某文件的修改（未提交的改动会丢）
```

---

## 三、服务器上更新代码

```bash
cd ~/project-PCC/Deep_learning_model_for_sugarcane
git fetch origin
git log --oneline HEAD..origin/main     # 先看会拉到什么，别闭眼拉
git pull --ff-only origin main          # 只接受快进合并；有分叉会报错而不是乱合并
```

拉完先确认代码是新的，再启动训练：

```bash
git log --oneline -3                    # 确认 HEAD 是你刚推的那条
python -c "import config; print(config.LOSS_POS_WEIGHT)"   # 抽查改动的参数
```

---

## 四、常见问题

### 1. push 被拒：`! [rejected] ... fetch first`

远端有你本地没有的提交。**不要直接 `git pull`** —— 默认会生成一个多余的合并提交，历史变乱。

```bash
git fetch origin
git log --oneline --graph HEAD..origin/main     # 远端多出来的提交
git diff HEAD origin/main --stat                # 它们动了哪些文件
```

确认没冲突（最好是压根没碰同一批文件）再：

```bash
git rebase origin/main        # 把你的提交挪到远端最新之上
git push origin main
```

> rebase 过程中有冲突：改完冲突文件 → `git add <文件>` → `git rebase --continue`；
> 想放弃回到原样：`git rebase --abort`。

### 2. 连不上 GitHub

**Git 不会自动走 Windows 系统代理**，Clash 开着也没用，得显式告诉它。

先确认 Clash 在跑（端口按你实际设置，常见 `7897` / `7890`），然后临时用：

```bash
git -c http.proxy=http://127.0.0.1:7897 -c https.proxy=http://127.0.0.1:7897 push origin main
```

想长期生效：

```bash
git config --global http.proxy  http://127.0.0.1:7897
git config --global https.proxy http://127.0.0.1:7897
```

取消（比如换了网络环境导致反而连不上）：

```bash
git config --global --unset http.proxy
git config --global --unset https.proxy
```

还连不上，先看 Clash 的 **mode 是不是 rule 模式、核心日志有没有报错**，再试 `ping github.com`。

### 3. 远端目录/文件被网页删了

本项目实际发生过（直接在 GitHub 网页上删文件）。

```bash
git fetch origin
git log --oneline origin/main -5     # 看远端最近动了什么
```

- 想**保留自己的改动**：按上面第 1 条 rebase；
- 想**完全以远端为准**（本地未提交的改动会全部丢失，慎用）：
  ```bash
  git reset --hard origin/main
  ```

### 4. 不小心把数据集/模型加进了暂存区

```bash
git reset HEAD datasets/      # 只取消暂存，文件本身不删
```

然后确认忽略规则确实生效：

```bash
git check-ignore -v datasets/root/train/images/xxx.png
# 有输出并指向 .gitignore 的那一行 = 规则生效
```

### 5. 检查本机的 Git 身份

```bash
git config --get user.name
git config --get user.email
```

---

## 五、本项目的三条约定

1. **单人开发，直接推 `main`**，不开分支、不做 PR。
2. **训练在服务器上跑**。代码走 git，数据集走手动同步。
3. **数据集必须两边一致** —— 这一条踩过坑：服务器上曾留着旧的 31 张全集（没做整株划分），
   而本地是 24 训练 / 7 测试。结果服务器训出来的模型**见过本地全部 7 张测试图**，
   两边的指标完全没法比。

   **每次同步完数据集，都跑一遍确认**：

   ```bash
   python test/test.py            # 应当正好评测 7 张测试图
   ls datasets/root/train/labels/other/   # 三通道训练要求这份 json 存在
   ```

   `labels/other/` 缺失时**不会报错** —— 代码会告警一次然后把 stem / check 两路的损失
   置零继续训，最后给你一个茎和检查范围全是废输出的模型。很隐蔽，务必先 `ls` 一眼。
