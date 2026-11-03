# HGKS / HD-EPIC → BT 质量评价协议 v1.0

版本：2026-11-03。适用于 **从视频或抽帧恢复演示中实际发生的任务相关动作**，不等于机器人执行成功率。

## 评价对象：三个层次不能混用

1. **原始标注**：HD-EPIC Fine Narrations 的视频ID、起止时间、动作文本、原始标注ID。只作为证据，不擅自把标签视为完整BT。
2. **参考Gold**：先独立于预测结果，人工合并/筛选任务相关动作实例，标注标准动词、对象、目标、时间及明确依赖；同一视频同一范围对所有方法共用。
3. **模型预测**：各方法原始 `BT_OUTPUT.json`。必须按真实job与视频ID追踪；不允许用模型答案回填Gold。

### Gold的冻结门槛

`gold/<video_id>/gold.csv`：逐动作实例至少包含 `reference_id`，应附带原Fine ID、时间、动词、对象、目标、任务相关性、合并原因和不确定性。

`gold/<video_id>/manifest.json`：至少记录 `video_id`、`scope_id`、`gold_version`、`start_s`、`end_s`、`source_sha256`、`annotation_status`、`scope_complete`、审阅者和允许的排除规则ID。

`gold/<video_id>/relations.csv`：`from_reference_id,to_reference_id,relation_type`。关系必须区分 `OBSERVED_BEFORE`（视频里观察到的先后）与 `PREREQUISITE`（真实的必要依赖）；时间先后不能自动当成逻辑必需。

每个方法的人工审核文件 `reviewed/<job_id>.csv` 至少有 `event_id,status,reference_id,exclusion_rule_id`。每个预测Action必须由人判断一次。`status`只允许 `TP`、`FP`、`EXCLUDE`；`EXCLUDE`必须引用Gold manifest事先列出的排除规则。`TP`与Gold一对一匹配，禁止一个参考实例多次命中。待确认的 `MAYBE` 不算半个TP，必须裁决后才算正式分数。

**只有 Gold 的 `annotation_status=LOCKED` 且 `scope_complete=true`、所有预测Action均已逐条裁决，评分器才输出正式Action F1。** 这些标记必须真实反映人工审查，不可自动置位。

### 动作粒度

任务相关实例优先按 `verb + object + target` 与具体时间事件对齐。连续搅拌如无状态变化中断可以合并；加料前后搅拌必须分开。非任务相关动作的排除规则在看到预测结果之前固定。未知Target不可由模型猜测冒充官方事实。

### 主指标：Action Precision / Recall / F1

TP：独立确认匹配的预测动作实例；
FP：未匹配、虚构、无关或重复的预测动作；
FN：未被任何预测动作匹配的Gold动作。

`Precision=TP/(TP+FP)`；
`Recall=TP/(TP+FN)`；
`F1=2TP/(2TP+FP+FN)`。

必须同时报告TP/FP/FN，不允许用“Action数量多”替代准确率。没有完成全范围Gold之前，P08七动作试点只能报告**局部覆盖范围**，不能报告全视频Action F1。

### 顺序、结构与可执行性分别报告

- `Order end-to-end Recall`：树确保正确顺序的参考关系边 / Gold全部参考边；缺失端点属于未恢复。
- `Conditional Order Accuracy`：两个端点都匹配的Gold边中，被树正确排序的比例；没有可比边为NA。
- `PREREQUISITE` 必须单独报告，不能把观察顺序当成必要依赖。
- 自动顺序评分只支持纯 `Sequence(Action...)`；`Fallback` / `Condition` / 并行分支需要真实BT语义/仿真检查，不扁平化算顺序。
- `Basic Structural Validity` 只检查当前项目JSON、节点引用及字段，不代表BT.CPP XML或机器人技能调用正确。
- 视觉证据按实际输入图片复核：`DIRECT_ACTION`、`STATE_CHANGE`、`TEXT_PRIOR_ONLY`、`UNSUPPORTED`、`UNREVIEWED`。模型自报VISUAL不是真值。
- `Simulation Success` 仅在动作技能、状态、目标、模拟器和初始条件完整时计算；此前一律NA。

### 可比性

同一个视频、相同范围和Gold版本，固定模型、Prompt、输出schema、图片布局和评分代码，再比较不同K或选帧算法。跨视频不同K不是K消融实验。报告必须包含Gold版本、方法、K、模型版本、TP/FP/FN、F1、顺序、结构、证据及执行状态。

若变更动作合并、Gold范围、关系或判定规则，发布新协议版本，不能在原v1.0之下覆盖历史成绩。

## 参考资料

- HD-EPIC官方标注：https://github.com/hd-epic/hd-epic-annotations
- Zhao等，Video-to-BT（ICRA 2026）：https://arxiv.org/html/2509.16611
- BehaviorTree.CPP节点语义：https://www.behaviortree.dev/docs/learn-the-basics/bt_basics/

注意：这里的Action F1与Order Recall是本基准定义的具体公式，不宣称与某篇论文指标逐字相同。
