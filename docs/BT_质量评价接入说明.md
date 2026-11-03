# BT质量比较如何接入1019

**现有主线**：`prepare.py → train.py → predict.py / baselines.py → run1019.py → evaluate.py`，其中原`evaluate.py`评价的是关键帧对Fine Action时间区间的覆盖。它不是BT正确性。此新增模块**不改动这条主线的任何函数**。

**新增下游**：关键帧图片 → 独立GPT生成的原始`BT_OUTPUT.json` → `bt_quality.compare` → 审核后的质量比较表。BT生成模块未包含在本提交中。

## 1. 代码入口（Windows PowerShell）

从仓库根目录运行轻量测试：

`python -m unittest discover -s tests -p test_bt_quality.py -v`

已有历史方法清单与BT作业时，运行：

`python -m bt_quality.compare --methods "D:\BT\PRIVATE\LATEST_BT_METHOD_TABLE.csv" --bt-root "D:\BT\GPT_PUBLIC\jobs" --out "reports\bt_quality\historical"`

会输出：
- `BT_METHOD_COMPARISON.csv`：方法、视频、K、Action数、结构检查、F1可用状态；
- `BT_COMPARISON_MANIFEST.json`：实验行数与输入哈希；
- `BT_COMPARISON_README_CN.md`：中文说明和评价边界。

如需复算P08七动作的**临时试点**，追加两个参数：

`--pilot-gold "D:\BT\P08_7step_reference_actions.csv" --pilot-judgments "D:\BT\P08_preliminary_adjudications.csv"`

它只能得到“明确/可能”动作覆盖范围，不是正式F1。

## 2. 人工审核后使用完整Gold

单独建立：

`gold_LOCKED/<video_id>/gold.csv`
`gold_LOCKED/<video_id>/manifest.json`
`gold_LOCKED/<video_id>/relations.csv`
`reviewed/<job_id>.csv`

命令追加：

`--gold-root "D:\BT\gold_LOCKED" --judgments-root "D:\BT\reviewed"`

严格遵守 `docs/BT_评价协议_v1.0.md`，人工Gold还没锁定时，不要手工把 `scope_complete` 改成true。

## 3. 哪些东西不进公开GitHub

本提交只含通用Python评价器、单元测试、协议与说明。**不提交**原始GPT输入/输出、未审核的Gold、未授权HD-EPIC数据、帧图片、视频、模型权重、API凭证或有隐私信息的审查表。真实评分数据留在受控本地目录；获准公开的精简结果可另行审查后提交。

历史记录：30条方法行、24份有BT原文，其中P08的8份有七动作局部临时比较；仍未完成正式人工Gold，也未测试机器人执行。不要将这些数据描述成BT质量已经通过验收。

## 4. 不变的边界

此模块不修改1018/1019的HGB、HO、141维特征、K预算函数或任何现有选帧策略，也不会更改GPT历史输出。将来不同方法/不同K的`BT_OUTPUT.json`只要使用相同字段，就可以复用本模块比较。
