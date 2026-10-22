# hksgv1：用相同帧数预算比较不同选帧策略

一段长视频包含许多操作，但下游系统只能查看 K 张图片。选帧要解决的问题是：**这 K 张图落在哪些时刻，才能覆盖更多实际发生的细粒度动作？**

本仓库比较 Lite 与时间采样、颜色聚类、视觉语义多样性、SOP 文本相关性等策略。下面先解释 baseline 在做什么、参考哪里、代码如何实现，再说明比较条件与结果。
这里测量的是 **Fine Action 时间区间覆盖率**：选到动作发生时的一帧，表示获得了观察该动作的时间机会；是否看清动作、能否正确理解、最终 BT/rTreeV2 是否正确，需要另外评价。本仓库没有生成 BT/rTreeV2 成品。

## 1. Baseline 一览：每个对照回答什么问题

主表有 **Lite + 9 个 baseline 方法族**。Random 的两个版本只是候选范围不同；同一版本运行 10 个种子，仍算一个方法族。
下表中的“适配”表示借鉴论文机制、按本任务重新规定特征或帧数等条件，不能作为作者官方完整实现来引用。

| 名称 / CSV 中的 method_id | 通俗解释 | 选帧实际使用的信息 | 文献关系 | 比较它的意义 |
|---|---|---|---|---|
| **Lite（本项目方法）**<br>`1017_LITE_FINAL` | 用手与物体线索预测候选价值，再兼顾时间分散 | Hand RLE、Object/coarse 标注、TRAIN Fine 监督训练的 HGB | 本项目冻结方法 | 检查这些线索构成的完整系统能否增加动作时间覆盖 |
| **Uniform**<br>`UNIFORM_FULL_ALL_FRAMES` | 沿视频等间隔取 K 帧 | 官方全部物理帧的时间/次序 | 基础对照，无单一对应论文 | 最重要的低成本参照：是否值得设计复杂策略 |
| **Random all frames**<br>`RANDOM_ALL_FRAMES_10_SEEDS` | 在全部物理帧中随机取 K 帧 | 帧的时间/次序、固定种子 | 基础对照 | 检查结果是否超过随机选择，以及随机波动有多大 |
| **Random 4K**<br>`RANDOM_4K_10_SEEDS` | 在视觉方法使用的 4×K 个网格点中随机取 K 个 | 4×K 时间网格、固定种子 | 基础对照 | 给使用相同网格的视觉方法提供随机参照 |
| **Inverse Exp SOP Stage**<br>`INVERSE_EXP_SOP_STAGE` | 每个粗阶段分配若干帧，阶段末尾更容易被选中 | 4×K 网格、官方 coarse 阶段 | 借鉴 Materia 等的逆指数采样 [R4]，加入长视频阶段适配 | 检查“阶段后期更有用”的时间偏好是否适合动作覆盖 |
| **VSUMM-style**<br>`VSUMM_HIST_KMEANS` | 把颜色相似的画面分组，每组找一张代表图 | RGB 的 40 维 HSV 直方图 | VSUMM 颜色聚类思想 [R1] 的本地适配 | 检查颜色代表性是否足以覆盖不同操作 |
| **MaxVol HSV / D-opt**<br>`MAXVOL_HIST_DOPT` | 找一组颜色特征尽量互补的帧 | 同一份 HSV40 | 本地机制对照，没有单一对应的完整论文算法 | 与 VSUMM 比较相同颜色输入下的不同选择规则 |
| **MaxInfo CLIP（全局适配）**<br>`MAXINFO_CLIP_MAXVOL_ADAPT` | 在全视频中找语义特征互补的帧 | 预训练 CLIP 图像向量 | MaxInfo [R2] 启发；本地用 logdet 贪心替代 `rect_maxvol` | 检查更强的语义特征和全局多样性是否有用 |
| **MaxInfo Chunk CLIP（分块适配）**<br>`MAXINFO_CHUNK_CLIP_ADAPT` | 先分时间块，再在每块选语义互补帧 | 相同的 CLIP 图像向量、时间块 | 借鉴 MaxInfo 的 Chunk 思路 [R2]；块数和配额为本地配置 | 检查限制时间集中、兼顾各段覆盖是否有帮助 |
| **AKS SOP CLIP（查询适配）**<br>`AKS_SOP_CLIP_RECONSTRUCTED_QUERY_ADAPT` | 按画面与操作步骤的相关性，自适应分段选帧 | CLIP 图像向量、SOP 步骤文本 | AKS [R3] 启发；问题查询改为 SOP，补足固定 K | 检查任务文本相关性是否比单纯视觉多样性更适合本任务 |

SOP 指操作步骤文本；coarse 指视频的高层活动阶段，Fine Action 指评价用的细粒度动作区间。
HSV 是颜色描述；CLIP 把图像、文本变成可比较的数值向量。向量相似不等于两个动作相同。

## 2. 先看数据怎样进入代码

当前结果表由 [`tests/compare_fine_budget.py`](tests/compare_fine_budget.py) 编排。它调用已有选择器，不重新实现各个算法：

```text
同一批 DEV23 视频 + 每视频同一个 K
  ├─ Lite：既有 Hand/Object 候选池 → 冻结 HGB + 时间分散
  ├─ Uniform / Random all frames：官方全部帧时钟 → 时间选择
  └─ 4×K 网格
       ├─ Random 4K / Inverse Exp：时间或 coarse 阶段选择
       └─ 实际 RGB → HSV40 / CLIP512 → 对应视觉选择器
                    ↓
       统一映射到合法 MP4 物理帧、检查恰好 K 个且不重复
                    ↓
       冻结全部选择及 SHA → 导出 Fine Gold → 同一评价函数
```

**4K 不是 4K 分辨率。** 例如预算 K=100，视觉方法预采样的是 400 个时间点，再从中选 100 个。
代码 [`baselines.grid()`](baselines.py#L28) 使用 `t_i = (i+0.5) × duration / (4×K)`，每个等长时间格取中心点。
[`dense_features()`](tests/compare_fine_budget.py#L136) 实际解码这些时间点对应的 RGB，调用 [`vision.histograms()`](vision.py#L168) 和 [`vision.encode_clip()`](vision.py#L216) 提特征。

HSV40 的产生过程是缩放到 96×54，再分别统计 H/S/V 的 16/12/12 个区间，每个通道归一化。
CLIP 使用本地预训练 `openai/clip-vit-base-patch32`，输出 512 维图像向量，本任务没有微调 CLIP；CLIP 的基础论文是 [R5]。
这些视觉方法共享网格和对应特征。Lite 保留自己的候选池，因此**所有方法的候选范围并不完全相同**。

统一映射使用 [`vision.snap_to_mp4()`](vision.py#L178)：找到请求时间处或之后的首个合法帧，发生重复时按固定规则调整，并记录请求秒数、真实帧号和实际秒数。
最终评价使用实际帧时间。时间方法的选择决策不读取 RGB；这个实验入口会为全部方法一起准备视觉特征，因此当前结果不是计算耗时比较。

## 3. 各方法的来源、实现和适配

### 3.1 Uniform：复杂方法必须面对的直接参照

把完整视频的物理帧序列分成 K 个等长下标区间，各取靠近中间的一帧。
代码是 [`baselines.uniform_indices()`](baselines.py#L49)，全帧调用见 [`baselines_stage()`](tests/compare_fine_budget.py#L218)。它只知道帧数和 K，没有训练，也不看动作标签。

比如某段“拿刀—切菜—转移食材”持续很短，均匀采样可能漏掉整个过程；如果预算足够密、动作区间较长或彼此重叠，Uniform 也可能获得很高覆盖率。
因此，Uniform 表现好并不要求 Fine Action 在时间上均匀分布。**若复杂方法无法稳定超过它，额外复杂度的收益就需要重新审视。**

主表只保留这个全帧 Uniform。源码中的 `UNIFORM_FULL_VIDEO`（4×K 网格）和 `UNIFORM_POOL_CURRENT`（Lite 候选池）用于保留原实验记录，不作为额外主表方法重复展示。

### 3.2 Random：排除“随便选也差不多”的可能

[`baselines.random_indices()`](baselines.py#L58) 用 NumPy `default_rng(...).choice(..., replace=False)` 无放回选 K 个点，然后按时间排序。
全帧版与网格版的区别仅是候选范围；它们是标准随机对照，无需把它们归到某篇特定论文。

两版各使用预先固定的 10 个基础种子 `20261012..20261021`，每个视频加上 `video_id` 字符编码之和作为偏移；统计全部种子的均值和总体标准差，未挑最好一次。
调用见 [`tests/compare_fine_budget.py`](tests/compare_fine_budget.py#L221)。10 个种子的波动只描述抽样随机性，不是跨视频泛化的置信区间。

### 3.3 VSUMM-style：每类颜色选一张代表帧

**文献来源：** de Avila 等，*VSUMM: A mechanism designed to produce static video summaries and a novel evaluation method*，Pattern Recognition Letters，2011 [R1]。
原研究面向静态视频摘要，以颜色特征和 K-means 寻找代表帧，并与人工摘要比较；作者资料提供 Open Video 等视频及 CUS 评价。它原本的摘要评价与这里的动作时间覆盖是两种指标。[作者资料](https://github.com/sandraavila/vsumm)

本地 [`vsumm_indices()`](baselines.py#L144) 接收 HSV40，运行 `KMeans(n_clusters=K, random_state=20261012, n_init=4, max_iter=120)`，然后为每个聚类中心找距离最近且尚未使用的真实候选，得到恰好 K 帧。
HSV40、4×K 预采样、固定 K 和这些聚类设置属于本地配置；没有把作者完整 VSUMM 流程和 CUS 评价搬过来，因此称 **VSUMM-style**。

它检验的是“看起来有代表性”是否足够。例如同一砧板上的拿刀、切菜、收刀可能颜色接近，颜色聚类没有明确要求把这些细动作分别保留。

### 3.4 MaxVol HSV / D-opt：同样的颜色输入，改为集合互补

这是本地机制对照，使用最大体积 / D-optimal 信息多样性的思想，没有独立对应的一套完整论文方法。其名称不表示作者官方 MaxInfo，也不对应其他同名 MaxVol 视频论文的复现。

[`maxvol_indices()`](baselines.py#L217) 先调用 [`svd_reduced()`](baselines.py#L167)：特征减均值、SVD 降维，保留 `r=max(1,min(32,K,N-1,d))` 个方向，用 `U × singular_values^0.3` 后对每行归一化。
随后 [`_d_opt_logdet_indices()`](baselines.py#L188) 逐轮选择最能增加正则 logdet 目标的未选点：

```text
A = 0.6I + Σ(z_i z_iᵀ)，i 为已选帧
每轮选使 z_jᵀ A⁻¹ z_j 最大的未选帧，更新 A 的逆矩阵，直到 K 帧。
```

可以把它理解成：已经选了某一类颜色变化后，继续寻找能够补充其他特征方向的画面。
与 VSUMM 共享 HSV 输入和网格，使这两个本地配置之间的比较更容易定位到选择规则；结果仍是这两个具体实现的比较。

### 3.5 MaxInfo CLIP：用语义特征寻找多样性，全局与分块两版

**文献来源：** Pengyi Li 等，*MaxInfo: A Training-Free Key-Frame Selection Method Using Maximum Volume for Enhanced Video Understanding*，WACV 2026 [R2]。
原研究通过帧嵌入、SVD 和最大体积选择减少冗余，包含 Fast、Slow、Chunk 版本，在 LongVideoBench、EgoSchema 等视频理解任务评价。[论文](https://openaccess.thecvf.com/content/WACV2026/html/Li_MaxInfo_A_Training-Free_Key-Frame_Selection_Method_Using_Maximum_Volume_for_WACV_2026_paper.html)

本仓库的两行结果对应：

| 本地版本 | 如何实现 | 与另一版相比增加了什么 |
|---|---|---|
| 全局 `MAXINFO_CLIP_MAXVOL_ADAPT` | 把全视频的 CLIP512 交给上述 `maxvol_indices()` | 在全部候选上选择语义特征互补集合 |
| 分块 `MAXINFO_CHUNK_CLIP_ADAPT` | [`maxinfo_chunk_indices()`](baselines.py#L224) 按时间连续分成 `min(16,K)` 块，按块内候选数分配预算，各块分别运行同一 SVD/logdet 选择，再合并 | 约束帧分布在各个时间块，减少全局选择集中于局部的可能 |

**与作者实现的关键差异：**作者公开的 [Fast notebook](https://github.com/FusionBrainLab/MaxInfo/blob/main/Code-Submit-LLaVA-Video-Fast-Version.ipynb) 示例使用 CLIP ViT-L/14-336、SVD 的 U 子空间和 `rect_maxvol`；本地使用 ViT-B/32、上述缩放/归一化和固定 K 的正则 logdet 贪心。
本地 Chunk 还固定了最多 16 块及配额规则。保留了“语义互补 / 时间分块”的设计思想，但没有执行完整官方选择器、VLLM 推理和论文评测。
因此结果表标为 **MaxInfo 启发的适配版**，不能把它的高低解释成官方 MaxInfo 的胜负。

它们有助于回答：只是保留语义上不同的画面，能否达到动作覆盖目标？增加时间分散约束是否更合适？全局与分块共享同一 CLIP 和选择内核，便于观察这项设计差异。

### 3.6 AKS SOP CLIP：选与操作步骤有关的画面

**文献来源：** Xi Tang 等，*Adaptive Keyframe Sampling for Long Video Understanding*，CVPR 2025 [R3]。
AKS 同时考虑帧与输入提示的相关性和视频覆盖，原实现支持 BLIP/CLIP/SeViLA，并以 LongVideoBench、Video-MME 等视频问答任务为应用场景。[论文](https://arxiv.org/abs/2502.21271)、[作者实现](https://github.com/ncTimTang/AKS)

本地没有把 Fine 区间或官方 QA Gold 交给选择器，而是做了以下适配：

1. [`vision.encode_clip()`](vision.py#L216) 分别编码每个 SOP 步骤，逐条归一化后取均值，再归一化为一个查询向量；图像也做单位归一化。
2. [`aks_indices()`](baselines.py#L266) 计算候选图像与查询的余弦相似度，在视频内缩放到 0..1。
3. [`aks_segments()`](baselines.py#L243) 借鉴作者 [`frame_select.py`](https://github.com/ncTimTang/AKS/blob/main/frame_select.py) 的 meanstd 条件递归二分时间序列，使用 `t1=0.8, t2=-100, max_depth=5`；各段按深度分配初选配额并取高分帧。
4. 初选去重，过多则按分数保留 K 个；不足则按 `相似度 + 0.16 × 时间分散奖励` 补到 K。时间奖励使用网格下标的最近距离。

主要差别是：作者的逐问题相关性查询，变成了整个操作流程的 SOP 平均查询；固定 K 的去重/补选也是本地规则。
这能检验 SOP 相关性策略，但步骤平均可能稀释短小操作，且不要求每个 Fine Action 都得到一帧。它的结果不能代表原版 AKS 在问答任务中的效果。

### 3.7 Inverse Exp SOP Stage：每段更偏向末尾

**来源：** Daniele Materia、Francesco Ragusa、Giovanni Maria Farinella，*Leveraging Gaze and Set-of-Mark in VLLMs for Human-Object Interaction Anticipation from Egocentric Videos*；作者仓库标注 ICPR 2026 [R4]。
原任务是第一人称人-物交互预判，完整系统还涉及 gaze、Set-of-Mark 和 VLLM。这里借鉴的是其中的时间采样机制。[作者仓库](https://github.com/fpv-iplab/leveraging_gaze_som_vllms_human_obj_anticipation)

作者的 [`sample_from_inv_exp_distrib()`](https://github.com/fpv-iplab/leveraging_gaze_som_vllms_human_obj_anticipation/blob/main/src/inference/obj_ia_inf_llava_ov.py) 按距末帧的下标距离给指数权重，并固定包含末帧。
本地 [`inverse_exp_indices()`](baselines.py#L108) 先按官方 coarse 阶段对网格点分组，按组的候选数量分配 K；有配额的组保留最后一点，其余按 `w ∝ exp(3 × (pos-1))` 无放回抽样，`pos` 是组内归一化位置。
这改变了原采样的作用范围、位置尺度、指数强度和随机数设置；没有运行 gaze/SoM/VLLM，也没有复现原交互预判评价。

它检验“阶段后段可能更接近有用状态”的偏好。若关键细动作发生在前半段，或一个粗阶段包含很多动作，偏向末尾反而可能漏掉它们。

### 3.8 Lite 的区别：学习候选价值，同时兼顾时间分散

Lite 使用手运动、物体事件、邻近时刻变化等组成的 141 维候选特征，训练一个 HistGradientBoostingClassifier（HGB）。
[`train.oracle_labels()`](train.py#L25) 用 **TRAIN70** 的 Fine 区间构造预算约束下的候选正例；预测的是这种候选选择标签，不是动作类别或“画面看清动作”的概率。

推断时 [`predict.select_video()`](predict.py#L38) 将 HGB 输出转为视频内排名，每轮用“候选排名 + 到已选帧的时间新颖度”选一个未用候选，直到 K 帧。
当前实验只在 [`lite()`](tests/compare_fine_budget.py#L106) 调用期间把预算依赖设为给定 K，保留原 `select_video()`、特征、权重和冻结模型；没有针对新 K 重训。
原 TRAIN 候选标签使用的 0.55 秒容差也保持原样，它与本轮 **评价** 的 0/0.5 秒是不同设置。

## 4. 为什么这组比较有意义，结论应怎样理解

这组 baseline 覆盖了不同选择理由：时间均匀、随机、阶段末尾偏好、颜色代表性、特征多样性、语义多样性和 SOP 相关性。
因此能直接观察：在同一批视频、同一帧数限制下，Lite 系统是否比这些现有本地策略覆盖更多 Fine 区间；如果 Uniform 已足够好，也会在同一指标上体现出来。

比较的共同条件是：**相同 DEV23、每视频相同 K、每种方法恰好 K 个不重复合法物理帧、同一份 Gold、同一个评价函数和同一种容差**。
所有选择先冻结并记录 SHA，之后才导出 Gold 评分；Random 使用全部预设种子。预算由统一阶段提供已知 Fine 数量，选择器没有读取 Fine 区间边界来选点。

同时需要把“相同预算”与“相同信息”分清楚：

| 条件 | 当前实验如何处理 | 对结论的影响 |
|---|---|---|
| 最终帧数与评价协议 | 每视频 K 和评分完全一致 | 结果可作为当前配置的同预算覆盖率比较 |
| 上游信息与监督 | Lite 有 Hand/Object/coarse 和 TRAIN Fine 监督；视觉方法主要用 RGB，AKS 加 SOP，Inverse 加 coarse | 收益可能同时来自输入、候选池、监督和选择规则，不能单独归因给学习器 |
| 候选空间 | Lite 自有池；视觉方法用 4×K；Uniform/Random 全帧版使用全部帧时钟 | 同预算不表示搜索范围或预处理成本相同 |
| 与原论文的一致性 | 保留本地适配身份，原任务/特征/选择器的不同已列出 | 不能声称击败 VSUMM/MaxInfo/AKS 的官方完整系统或论文排行榜 |
| 泛化与统计 | 当前是 DEV；两种容差是在观察 ±2 秒饱和后选择；Random 标准差仅反映种子变化 | 是探索性观察，尚未给出独立 TEST 或统计显著性结论 |

如果要单独验证“选择算法本身更好”，还需要共同候选池与可用输入、Lite 消融以及独立留出评价；这些是后续实验，本轮没有通过改动现有算法来补做。

## 5. 当前实验条件、评价与结果

固定 DEV23 共 **10,344 个官方 Fine Action 区间**，每个视频的 K 等于该视频区间数量，每种方法总预算 10,344 帧；视觉网格合计 41,376 个点。
这是已知动作数量的 **oracle 预算**，用于研究给定预算下的选帧表现；实际部署中如何估计动作数量不在本次实验内。
K 等于区间数不表示“一帧对应一个动作”：有些区间会重叠，一帧可能命中多个动作，也可能多个选帧重复命中同一动作而漏掉别的动作。

动作区间是 `[a,b]`，只要存在选帧时间 t 满足 `a-tolerance <= t <= b+tolerance`，这个区间记一次命中。
容差 0 秒要求在区间内；0.5 秒把区间两端各外扩 0.5 秒。它们不是相对于动作中心点的容差。
[`evaluate.hit_indicators()`](evaluate.py#L16) 是唯一命中实现；[`multi_video_report()`](evaluate.py#L36) 汇总 Micro（总命中/总区间）和 Macro（各视频命中率的平均值）。

预算较密、动作持续较长、区间重叠以及较宽容差都可能让 Uniform 接近饱和；这并不能推出动作分布均匀。
当前报告 0 和 0.5 秒，是在查看 DEV 的 ±2 秒饱和情况后选择的探索性设置；两种容差对全部方法一致，没有逐方法挑评价口径。

以下是既有本地完整实验的结果快照。本次只改 README，没有重新训练、提取特征或改变选择结果：

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

以 Micro 为例，Lite 相对全帧 Uniform 的差值为： **0 秒 +4.37 个百分点（多命中 452 个区间）**； **0.5 秒 +4.40 个百分点（多命中 455 个区间）**。这是上述完整系统在当前 DEV 协议中的观察结果，未做统计显著性检验。

## 6. 源码阅读入口与冻结版本

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

## 7. 运行所需的本地资源

仓库包含源码、依赖记录、上述汇总 CSV，以及冻结 Lite 模型和 `TRAIN_RECEIPT.json`。
外部机器仍需自行具备 HD-EPIC 视频/标注、研究 split、portable 输入、官方帧时钟和缓存收据，以及本地 CLIP snapshot。
这些数据、原始图片、完整实验结果、CLIP 权重和 Python 环境没有随仓库分发。仅 clone 仓库不能立即重现全部数据实验。

1. 保持两套 Python 环境：Lite/评价对应 `requirements.txt`；视觉特征/baseline 对应 `requirements-baselines.txt`。两个文件记录本机实际版本，本次发布没有安装包。
2. 将 `paths.example.json` 复制为 `paths.json`。数据路径相对于源码根目录；把 `feature_python` 和 `ffprobe` 改为本机现有工具的完整路径。路径示例中的 `tools/` 是占位目录，不随仓库提供。
3. 配置示例中的 TRAIN70/DEV23、split、annotations、clocks、features、video_sources 和 clip_snapshot；缓存必须匹配对应数据和官方时钟的收据。
4. Fine-budget 入口固定读取 `results/DEV23_FINAL_20261020_1019/train/HGKS_LITE_MODEL.joblib` 及同目录训练回执，两者已按原路径附上。模型 SHA-256：`893e6304635956e08dc579e672f26ed93e745980c3979ad0d49df46210b661b1`。

输入准备接口可分别查看 `python prepare.py raw --help` 和 `python prepare.py inputs --help`。
输入/缓存准备仍需完整外部资源，本快照没有把历史原始数据准备和迁移审计工具整套打包。

## 8. 命令示例

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

## 9. 已做的验证与复现范围

初次发布已检查复制文件 SHA-256、Python 语法，并用公开包中的 `evaluate.py` 对本机冻结 Lite/Uniform 预测分别执行默认 0 和显式 0.5 秒评分，结果与汇总表一致；三个 CLI 均按预期拒绝 2 秒容差。
此次 README 更新核对方法名、函数链接、文献出处、结果表和哈希清单，仅修改文档及其清单项。算法、模型、原始结果 CSV 保持原字节；没有重跑训练、全量 pytest、特征提取或 BT 生成。
本机验证不替代另一台机器上的完整复现，所需外部数据和依赖仍需配置。

## 10. 文献与作者资料

以下来源于 2026-10-22 核对。参考文献说明思想来源；具体被测算法以本仓库 Python 实现及上文列出的适配为准。

| 编号 | 文献 | 可核对的一手来源 |
|---|---|---|
| R1 | de Avila 等，VSUMM，Pattern Recognition Letters 32(1):56–68，2011 | [论文 DOI](https://doi.org/10.1016/j.patrec.2010.08.004)、[作者维护的数据、摘要与 CUS 资料](https://github.com/sandraavila/vsumm)。该资料仓库不表示本地直接使用作者完整选帧程序。 |
| R2 | Pengyi Li 等，MaxInfo，WACV 2026，pp.7198–7207（预印本发表于 2025） | [CVF 论文](https://openaccess.thecvf.com/content/WACV2026/html/Li_MaxInfo_A_Training-Free_Key-Frame_Selection_Method_Using_Maximum_Volume_for_WACV_2026_paper.html)、[作者代码](https://github.com/FusionBrainLab/MaxInfo)。已对照上文链接的 Fast notebook。 |
| R3 | Xi Tang 等，AKS，CVPR 2025 | [论文预印本](https://arxiv.org/abs/2502.21271)、[作者代码](https://github.com/ncTimTang/AKS)。已对照 `frame_select.py` 的分段与配额规则。 |
| R4 | Daniele Materia、Francesco Ragusa、Giovanni Maria Farinella，交互预判中的逆指数采样；作者引用信息标注 ICPR 2026 | [作者仓库与引用信息](https://github.com/fpv-iplab/leveraging_gaze_som_vllms_human_obj_anticipation)、[采样函数所在文件](https://github.com/fpv-iplab/leveraging_gaze_som_vllms_human_obj_anticipation/blob/main/src/inference/obj_ia_inf_llava_ov.py)。本仓库只借鉴时间采样，不复现完整交互预判系统。 |
| R5 | Alec Radford 等，*Learning Transferable Visual Models From Natural Language Supervision*，2021 | [CLIP 论文](https://arxiv.org/abs/2103.00020)。这是共享预训练特征编码器的来源，并不是本表另一个独立选帧 baseline。 |

Uniform、Random 属于常规控制策略，MaxVol HSV 是本地机制对照，不为它们补造“对应的官方论文”。
