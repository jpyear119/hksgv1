"""统一比较入口：读取各方法真实 BT，报告结构、P08临时覆盖、人工Gold正式分数。

从 1019 仓库根目录运行：
    python -m bt_quality.compare --methods 方法表.csv --bt-root GPT_PUBLIC/jobs --out reports/bt_quality

可加 --pilot-gold/--pilot-judgments 查看 P08 非正式局部覆盖。
可加 --gold-root/--judgments-root 对人工审核完成的同范围 BT 计算正式 F1。

不会修改训练/选帧算法、原始 BT 或输入文件，也不会自动调用 GPT。
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from .score_bt import check_bt, full_score, load_csv, pilot_scores


COLUMNS = [
    "video_id", "method", "K", "job_id", "source",
    "bt_available", "n_actions", "basic_structure_valid",
    "pilot_gold_n", "pilot_definite", "pilot_possible",
    "pilot_coverage_min", "pilot_coverage_max",
    "action_TP", "action_FP", "action_FN",
    "action_precision", "action_recall", "action_f1",
    "order_observed_e2e_recall", "order_prerequisite_e2e_recall",
    "visual_grounding", "robot_execution_success", "quality_status", "notes",
]


def sha256(path):
    """计算来源文件哈希，保证未来可以对照相同的数据版本。"""
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def locate_locked_gold(gold_root, video_id):
    """只按严格 video_id 定位，不猜测近似视频。"""
    if gold_root is None:
        return None
    folder = gold_root / video_id
    files = [folder / name for name in ("gold.csv", "manifest.json", "relations.csv")]
    return files if all(file.is_file() for file in files) else None


def evaluate_rows(methods, bt_root, pilot_gold=None, pilot_judgments=None,
                  gold_root=None, judgments_root=None):
    """输入真实 BT 与人工 Gold，输出固定列的逐方法比较结果。"""
    bt_root = Path(bt_root)
    if (pilot_gold is None) != (pilot_judgments is None):
        raise ValueError("pilot_gold and pilot_judgments must be provided together")
    pilot = {}
    if pilot_gold is not None:
        for item in pilot_scores(load_csv(pilot_gold), load_csv(pilot_judgments),
                                 methods, bt_root):
            pilot[item["job_id"]] = item

    results = []
    for method in methods:
        result = {column: "" for column in COLUMNS}
        for column in ("video_id", "method", "K", "job_id", "source"):
            result[column] = method.get(column, "")
        result["visual_grounding"] = "NOT_REVIEWED"
        result["robot_execution_success"] = "NOT_MEASURED"
        job_id = method.get("job_id", "")
        # 不允许方法表把输入目录跳转到外部文件。
        if job_id and (Path(job_id).name != job_id or job_id in {".", ".."}):
            raise ValueError(f"Invalid job_id path: {job_id}")
        path = bt_root / job_id / "BT_OUTPUT.json" if job_id else None
        if path is None or not path.is_file():
            result["bt_available"] = 0
            result["quality_status"] = "MISSING_FULL_BT"
            result["notes"] = "仅有历史汇总，缺少完整 BT_OUTPUT.json；不得虚构分数"
            results.append(result)
            continue

        bt = json.loads(path.read_text(encoding="utf-8"))
        if bt.get("video_id") != method["video_id"]:
            raise ValueError(f"Job {job_id}: video_id mismatch")
        if bt.get("job_id") and bt["job_id"] != job_id:
            raise ValueError(f"Job {job_id}: job_id mismatch")
        events = bt["events"]
        structure = check_bt(bt["bt"], events)
        result["bt_available"] = 1
        result["n_actions"] = len(events)
        result["basic_structure_valid"] = int(structure["structural_valid"])
        result["quality_status"] = "STRUCTURE_ONLY_NOT_SEMANTIC_SCORE"
        if structure["structural_errors"]:
            result["notes"] = ";".join(structure["structural_errors"])

        if job_id in pilot:
            source = pilot[job_id]
            for key, dest in (
                ("n_pilot_reference_actions", "pilot_gold_n"),
                ("n_definite", "pilot_definite"),
                ("n_possible", "pilot_possible"),
                ("min_coverage", "pilot_coverage_min"),
                ("max_coverage", "pilot_coverage_max"),
            ):
                result[dest] = source[key]
            result["quality_status"] = "PROVISIONAL_7_ACTION_PILOT_NO_F1"

        gold_files = locate_locked_gold(
            Path(gold_root) if gold_root is not None else None, method["video_id"])
        review_file = (
            Path(judgments_root) / f"{job_id}.csv" if judgments_root is not None else None
        )
        if gold_files and review_file is not None and review_file.is_file():
            gold_path, manifest_path, relations_path = gold_files
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("video_id") != bt["video_id"]:
                raise ValueError(f"{manifest_path}: Gold video_id mismatch")
            scores = full_score(
                load_csv(gold_path), manifest, load_csv(review_file),
                bt, load_csv(relations_path),
            )
            result.update({
                "action_TP": scores["TP"], "action_FP": scores["FP"],
                "action_FN": scores["FN"],
                "action_precision": scores["precision"],
                "action_recall": scores["recall"],
                "action_f1": scores["F1"],
                "order_observed_e2e_recall":
                    scores["order"]["OBSERVED_BEFORE"]["end_to_end_recall"],
                "order_prerequisite_e2e_recall":
                    scores["order"]["PREREQUISITE"]["end_to_end_recall"],
                "quality_status": "LOCKED_GOLD_AND_ALL_ACTIONS_REVIEWED",
            })
        elif gold_files or (review_file is not None and review_file.is_file()):
            result["quality_status"] = "INCOMPLETE_GOLD_OR_HUMAN_REVIEW"
            result["notes"] = "需要完整 Gold 和完整逐 Action 人工裁决"
        results.append(result)
    return results


def write_results(rows, out, metadata):
    """只写新的结果目录，不覆盖输入；列顺序固定以便后续跨K比较。"""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "BT_METHOD_COMPARISON.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "protocol_version": "BT_评价协议_v1.0",
        "total_method_rows": len(rows),
        "full_bt_available": sum(int(row["bt_available"]) for row in rows),
        "basic_structure_pass": sum(row["basic_structure_valid"] == 1 for row in rows),
        "provisional_pilot_rows": sum(
            row["quality_status"] == "PROVISIONAL_7_ACTION_PILOT_NO_F1" for row in rows
        ),
        "locked_gold_scored_rows": sum(
            row["quality_status"] == "LOCKED_GOLD_AND_ALL_ACTIONS_REVIEWED"
            for row in rows
        ),
        "not_real_execution": True,
        "input_source": metadata,
    }
    (out / "BT_COMPARISON_MANIFEST.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    text = [
        "# BT下游质量比较（真实运行输出）", "",
        f"方法记录：{len(rows)}，完整BT：{summary['full_bt_available']}，"
        f"基础结构通过：{summary['basic_structure_pass']}。",
        f"P08局部试点：{summary['provisional_pilot_rows']}；"
        f"完整人工Gold评分：{summary['locked_gold_scored_rows']}。",
        "", "**结构合法≠动作正确；P08七动作覆盖≠Action F1；未运行机器人执行。**", "",
        "逐方法数据见 BT_METHOD_COMPARISON.csv。同视频、同Gold与同Prompt/VLM版本"
        "的条件下比较不同K，不能用跨视频不同K推断预算曲线。",
    ]
    (out / "BT_COMPARISON_README_CN.md").write_text(
        "\n".join(text) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", type=Path, required=True)
    parser.add_argument("--bt-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pilot-gold", type=Path)
    parser.add_argument("--pilot-judgments", type=Path)
    parser.add_argument("--gold-root", type=Path)
    parser.add_argument("--judgments-root", type=Path)
    args = parser.parse_args()
    methods = load_csv(args.methods)
    required = {"video_id", "method", "K", "job_id", "source"}
    if not methods or not required.issubset(methods[0]):
        parser.error(f"Methods CSV requires {sorted(required)}")
    rows = evaluate_rows(
        methods, args.bt_root, args.pilot_gold, args.pilot_judgments,
        args.gold_root, args.judgments_root,
    )
    provenance = {"methods_sha256": sha256(args.methods)}
    for name, path in (
        ("pilot_gold_sha256", args.pilot_gold),
        ("pilot_judgments_sha256", args.pilot_judgments),
    ):
        if path is not None:
            provenance[name] = sha256(path)
    summary = write_results(rows, args.out, provenance)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
