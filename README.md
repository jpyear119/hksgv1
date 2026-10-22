# BT/rTreeV2 frame selection: frozen local research code

用于 BT/rTreeV2 研究前端的选帧代码公开快照，供阅读和复核。当前实验比较的是 **Fine Action 时间区间覆盖率**。
本仓库没有生成 BT/rTreeV2 成品，也没有测量 BT 正确率、视觉理解正确率或机器人执行成功率。

## 当前版本与源码对应关系

正式源码和实验脚本均从本地现有 Python 文件逐字节复制。此次发布只新增发布说明、配置示例、忽略规则和文件哈希清单，没有另写一套选帧算法。
发布前已授权的容差变更仅位于 `evaluate.py`、`run1019.py`、`tests/compare_fine_budget.py`：默认容差为 0 秒，CLI 可选 0 或 0.5 秒，并传递参数、记录实际指标标签。
命中公式、HGB、特征、baseline 选择器及随机种子没有因此改变。
`SOURCE_SHA256.json` 可用于核对公开源码字节和冻结模型；README 是本次公开快照的使用说明，源码中的历史注释、可选 reference 参数属于既有实现。

| 文件 | 用途 |
|---|---|
| `data.py` | 输入读写、模式检查、官方时钟与标注处理 |
| `prepare.py` | 原始输入准备、候选池和 Lite 特征、原时长预算 |
| `train.py` | TRAIN70 上训练 Lite HGB |
| `predict.py` | Lite 原选帧算法 |
| `vision.py` | RGB/HSV/CLIP、MP4 帧对齐与特征缓存 |
| `baselines.py` | 时间、HSV、CLIP、SOP 等 baseline 的本地实现 |
| `evaluate.py` | 冻结预测后的 Gold 导出与 one-hit 评分 |
| `run1019.py` | 原时长预算全流程入口 |
| `tests/compare_fine_budget.py` | 冻结模型、每视频 Fine 数量预算实验入口（并非 pytest 单元测试） |

## 实验条件

- 固定 DEV23，共 10,344 个官方 Fine Action 区间。每个视频的 K 等于该视频 Fine 区间数量；每种方法总预算为 10,344 帧。
- 这是**已知动作数量的 oracle 预算**。选择阶段获得数量，没有把 Fine 边界作为选帧输入；冻结所有选择以后才导出评价 Gold。
- Lite 使用既有候选池及冻结 HGB。冻结模型沿用原时长预算训练，没有针对新 K 重新训练。
- 视觉方法重新提取 4×K 网格点特征，DEV23 共 41,376 个点。这里的 `4K` 指预算 K 的四倍，与 4K 分辨率无关。
- 主结果只保留一个 Uniform：`UNIFORM_FULL_ALL_FRAMES`，从官方物理帧时钟均匀选择 K 帧。它与 Lite 的基于特征和模型的选择不同。
- 原实验入口仍保留历史 Uniform 变体以保持选择代码原样；主表排除了这两种变体。原运行包含 12 个方法族、30 个方法/种子配置；主表展示 10 个方法族。
- 两类 Random 各使用 10 个固定种子，表中是均值；CSV 保留总体标准差与种子数。方法名中的 adapted/reconstructed 表示本地实现，不代表官方作者发布的实现。
- Lite 使用 Hand RLE、Object/coarse oracle 输入及 TRAIN Fine 监督；各 baseline 使用时间、HSV、CLIP、SOP 等不同信息。这是同 DEV、同数量预算的系统级比较。

## 评价与结果

动作区间为 `[a,b]`，只要至少一个选帧时间 t 满足 `a-tolerance <= t <= b+tolerance`，该区间命中一次。
容差 0 秒要求落在动作区间内；0.5 秒把两端各外扩 0.5 秒。它们不是动作中心点的 ±0/0.5 秒。
每区间最多记一次；同一帧可以命中多个重叠区间。Micro 是总命中数/总区间数；Macro 是每视频命中率的平均值。
0 和 0.5 秒是在查看 DEV 的 ±2 秒饱和情况后选择的探索性评价设置，后续确认性结论仍需独立验证。

以下为既有本地完整实验的结果快照，此次上传没有重新提取特征或重跑全部选帧：

| Method | Micro, 0 s | Macro, 0 s | Micro, 0.5 s | Macro, 0.5 s |
|---|---:|---:|---:|---:|
| Lite | 58.22% | 58.82% | 86.48% | 86.39% |
| Uniform (all physical frames) | 53.85% | 54.33% | 82.09% | 82.40% |
| MaxInfo Chunk CLIP (adapted) | 52.47% | 53.08% | 72.51% | 73.57% |
| VSUMM HSV K-means | 48.67% | 48.84% | 66.12% | 67.65% |
| MaxInfo CLIP MaxVol (adapted) | 46.93% | 48.39% | 65.93% | 68.40% |
| Random 4K (10 seeds) | 46.24% | 46.05% | 65.62% | 66.01% |
| MaxVol HSV D-opt | 44.41% | 46.05% | 62.21% | 65.01% |
| Random all frames (10 seeds) | 43.05% | 43.15% | 61.37% | 61.72% |
| Inverse Exp SOP Stage | 40.91% | 41.09% | 56.61% | 56.81% |
| AKS SOP CLIP (reconstructed/adapted) | 38.30% | 41.33% | 53.61% | 58.13% |

原始精度、命中数与随机种子标准差见 [`reports/dev23_fine_count_0_0p5.csv`](reports/dev23_fine_count_0_0p5.csv)。

## 运行所需的本地资源

仓库包含源码、依赖记录、上述汇总 CSV，以及冻结 Lite 模型和 `TRAIN_RECEIPT.json`。
外部机器仍需自行具备 HD-EPIC 视频/标注、研究 split、portable 输入、官方帧时钟和缓存收据，以及本地 CLIP snapshot。
这些数据、原始图片、完整实验结果、CLIP 权重和 Python 环境没有随仓库分发。仅 clone 仓库不能立即重现全部数据实验。

1. 保持两套 Python 环境：Lite/评价对应 `requirements.txt`；视觉特征/baseline 对应 `requirements-baselines.txt`。两个文件记录本机实际版本，本次发布没有安装包。
2. 将 `paths.example.json` 复制为 `paths.json`。数据路径相对于源码根目录；把 `feature_python` 和 `ffprobe` 改为本机现有工具的完整路径。路径示例中的 `tools/` 是占位目录，不随仓库提供。
3. 配置示例中的 TRAIN70/DEV23、split、annotations、clocks、features、video_sources 和 clip_snapshot；缓存必须匹配对应数据和官方时钟的收据。
4. Fine-budget 入口固定读取 `results/DEV23_FINAL_20261020_1019/train/HGKS_LITE_MODEL.joblib` 及同目录训练回执，两者已按原路径附上。模型 SHA-256：`893e6304635956e08dc579e672f26ed93e745980c3979ad0d49df46210b661b1`。

输入准备接口可分别查看 `python prepare.py raw --help` 和 `python prepare.py inputs --help`。
输入/缓存准备仍需完整外部资源，本快照没有把历史原始数据准备和迁移审计工具整套打包。

## 命令示例

以下用已配置的 Lite 环境 `python`；每个 `--out` 必须是新的输出位置。完整流程运行时间较长。

```powershell
# 只看入口参数，无需数据即可查看帮助
python evaluate.py score --help
python run1019.py --help
python tests/compare_fine_budget.py --help

# 当前实验：冻结 HGB；K=每视频 Fine 数量；容差 0
python tests/compare_fine_budget.py run --out results/FINE_COUNT_T0 --tolerance 0

# 原始入口：时长预算 max(1, ceil(duration_seconds/30))；会重新训练
# 这条命令的预算不是上面结果表的 Fine 数量预算
python run1019.py run --out results/DURATION_BUDGET_T0 --mode cached --tolerance 0
```

无需为了改评价容差重复选帧。对同一份已有冻结预测分别评分即可；旁边必须保留对应 `DEV_SELECTION_RECEIPT.json`，Gold 来自冻结后的导出：

```powershell
python evaluate.py score --predictions results/FINE_COUNT_T0/native/1017_LITE_FINAL/DEV_METHOD_FROZEN.json --targets results/FINE_COUNT_T0/gold/DEV23_FINE.json --out results/LITE_SCORE_T0.json --tolerance 0
python evaluate.py score --predictions results/FINE_COUNT_T0/native/1017_LITE_FINAL/DEV_METHOD_FROZEN.json --targets results/FINE_COUNT_T0/gold/DEV23_FINE.json --out results/LITE_SCORE_T05.json --tolerance 0.5
```

每次调用只输出一种容差。Fine-count 实验评分不要加 `--input`，该可选检查沿用原时长预算。
不要在已经完成的结果目录再次运行实验入口的 evaluate 阶段：Gold 和评分输出采用禁止覆盖写入；单独 score 使用新的文件名即可。
历史 reference 选项所需参考文件未随此公开快照分发，上述示例不依赖它们。

## 验证范围

发布准备会核对所有复制文件 SHA-256、Python 语法，以及用公开包中的 `evaluate.py` 对本机冻结 Lite/Uniform 预测分别执行默认 0 和显式 0.5 秒评分，比较既有汇总表。
验证还检查三个 CLI 拒绝 2 秒容差，且不运行训练、全量 pytest、特征提取或 BT 生成。
这不替代在另一台机器和另一份数据上验证环境可复现性。
