"""离线 BT 质量评分：基础结构、局部覆盖、人工锁定 Gold 后的动作 F1 与顺序。

注意：P08 七动作试点不是完整 Gold。没有完整且逐项裁决的 Gold 时，
本模块不会输出正式 Action F1。不会调用 GPT，也不会修改选帧模型。
"""
import argparse
import csv
import json
from pathlib import Path


LEGAL_TYPES = {"Sequence", "Fallback", "Condition", "Action"}


def load_csv(path):
    """读取 UTF-8/Excel 导出的 CSV。"""
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def check_bt(tree, events):
    """仅检查本项目 JSON 协议，不表示 BT.CPP 或机器人执行有效。"""
    counts = {kind: 0 for kind in LEGAL_TYPES}
    references = []
    errors = []

    def walk(node):
        if not isinstance(node, dict):
            errors.append("not_object")
            return
        kind = node.get("type")
        if kind not in LEGAL_TYPES:
            errors.append(f"unknown_node:{kind}")
            return
        counts[kind] += 1
        if kind in {"Sequence", "Fallback"}:
            children = node.get("children")
            if not isinstance(children, list) or not children:
                errors.append("empty_children")
                return
            for child in children:
                walk(child)
        elif kind == "Action":
            if not node.get("event_id"):
                errors.append("action_without_event_id")
            references.append(node.get("event_id"))
        elif kind == "Condition":
            if not node.get("predicate") or not node.get("object"):
                errors.append("condition_incomplete")

    walk(tree)
    event_ids = [event.get("event_id") for event in events]
    if len(event_ids) != len(set(event_ids)):
        errors.append("duplicate_events")
    if len(references) != len(set(references)):
        errors.append("duplicated_action_refs")
    if set(references) != set(event_ids):
        errors.append("action_ref_set_mismatch")
    return {
        "structural_valid": not errors,
        "structural_errors": sorted(set(errors)),
        "nodes": counts,
    }


def simple_sequence_order(node):
    """只展开纯 Sequence/Action 树；Condition/Fallback 不能扁平化。"""
    if not isinstance(node, dict):
        return None
    kind = node.get("type")
    if kind == "Action":
        return [node["event_id"]]
    if kind == "Sequence":
        order = []
        for child in node.get("children", []):
            partial = simple_sequence_order(child)
            if partial is None:
                return None
            order.extend(partial)
        return order
    return None


def pilot_scores(gold, adjudications, methods, root):
    """仅计算 P08 七动作临时参考的覆盖范围，不输出 Precision/F1。"""
    reference_ids = {row["reference_id"] for row in gold}
    if not reference_ids:
        raise ValueError("Empty pilot Gold")
    results = []
    for method in methods:
        if not (method["source"] == "NEW_REAL_GPT_6_1_SOL"
                and method["video_id"] == "P08-20240618-171546"):
            continue
        path = Path(root) / method["job_id"] / "BT_OUTPUT.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        events = record["events"]
        event_ids = {event["event_id"] for event in events}
        structure = check_bt(record["bt"], events)
        judgments = [row for row in adjudications if row["method"] == method["method"]]
        seen_references = set()
        seen_events = set()
        definite, possible = [], []
        for row in judgments:
            reference_id, event_id, grade = row["reference_id"], row["event_id"], row["grade"]
            if reference_id not in reference_ids or event_id not in event_ids:
                raise ValueError(f"Unknown reference/event: {row}")
            if reference_id in seen_references or event_id in seen_events:
                raise ValueError(f"Not a one-to-one match: {row}")
            if grade not in {"明确", "可能"}:
                raise ValueError(f"Unknown grade: {grade}")
            seen_references.add(reference_id)
            seen_events.add(event_id)
            (definite if grade == "明确" else possible).append(row)

        order = simple_sequence_order(record["bt"])
        agreements = total_pairs = 0
        if order is not None:
            certain = sorted(definite, key=lambda row: row["reference_id"])
            for i, left in enumerate(certain):
                for right in certain[i + 1:]:
                    agreements += int(order.index(left["event_id"]) < order.index(right["event_id"]))
                    total_pairs += 1
        results.append({
            "method": method["method"],
            "job_id": method["job_id"],
            "K": int(method["K"]),
            "n_predicted_full_video_actions": len(events),
            "n_pilot_reference_actions": len(gold),
            "n_definite": len(definite),
            "n_possible": len(possible),
            "min_coverage": len(definite) / len(gold),
            "max_coverage": (len(definite) + len(possible)) / len(gold),
            "matched_order_correct_pairs": agreements if order is not None else "",
            "matched_order_total_pairs": total_pairs if order is not None else "",
            "basic_structural_valid": int(structure["structural_valid"]),
            "full_action_F1": "NOT_MEASURED",
            "robot_execution_success": "NOT_MEASURED",
            "status": "PROVISIONAL_PILOT_NOT_GOLD",
        })
    return results


def full_score(gold, manifest, judgments, bt, edges):
    """仅在 Gold 完整锁定、每个预测 Action 均经人工裁决后计算正式 F1。"""
    if not (manifest.get("annotation_status") == "LOCKED"
            and manifest.get("scope_complete") is True):
        raise ValueError("REFUSE_FULL_SCORE: Gold must be LOCKED and scope_complete=true")
    if manifest.get("video_id") and manifest["video_id"] != bt.get("video_id"):
        raise ValueError("Gold and predicted video_id differ")
    reference_ids = [row["reference_id"] for row in gold]
    if not reference_ids or len(reference_ids) != len(set(reference_ids)):
        raise ValueError("Missing or duplicate Gold reference IDs")
    structure = check_bt(bt["bt"], bt["events"])
    events = {row["event_id"]: row for row in bt["events"]}
    if len(events) != len(bt["events"]):
        raise ValueError("Duplicate predicted events")
    reviewed_ids = [row["event_id"] for row in judgments]
    if len(reviewed_ids) != len(events) or set(reviewed_ids) != set(events):
        raise ValueError("Every predicted action must be explicitly adjudicated exactly once")

    matched = {}
    fp = excluded = 0
    for judgment in judgments:
        status = judgment["status"]
        event_id = judgment["event_id"]
        reference_id = judgment.get("reference_id", "").strip()
        if status == "TP":
            if reference_id not in reference_ids or reference_id in matched:
                raise ValueError("TP must uniquely match an existing Gold reference")
            matched[reference_id] = event_id
        elif status == "FP":
            if reference_id:
                raise ValueError("FP cannot reference Gold")
            fp += 1
        elif status == "EXCLUDE":
            rule_id = judgment.get("exclusion_rule_id", "").strip()
            if reference_id or rule_id not in manifest.get("allowed_exclusion_rule_ids", []):
                raise ValueError("EXCLUDE requires a predeclared exclusion rule")
            excluded += 1
        else:
            raise ValueError(f"Unreviewed prediction: {event_id}, status={status}")

    tp = len(matched)
    fn = len(reference_ids) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    order = simple_sequence_order(bt["bt"])
    order_scores = {}
    for relation_type in ("OBSERVED_BEFORE", "PREREQUISITE"):
        relations = [row for row in edges if row["relation_type"] == relation_type]
        if not relations:
            order_scores[relation_type] = {
                "n_edges": 0, "end_to_end_recall": None, "conditional_recall": None,
            }
            continue
        if order is None:
            order_scores[relation_type] = {
                "n_edges": len(relations), "end_to_end_recall": None,
                "conditional_recall": None,
                "reason": "Branch/condition nodes require semantic execution evaluation",
            }
            continue
        correct = eligible = reversed_count = 0
        for edge in relations:
            first, second = edge["from_reference_id"], edge["to_reference_id"]
            if first not in reference_ids or second not in reference_ids or first == second:
                raise ValueError("Invalid Gold relation")
            if first in matched and second in matched:
                eligible += 1
                correct_direction = order.index(matched[first]) < order.index(matched[second])
                correct += int(correct_direction)
                reversed_count += int(not correct_direction)
        order_scores[relation_type] = {
            "n_edges": len(relations),
            "n_matched_endpoint_edges": eligible,
            "n_correct": correct,
            "n_reversed": reversed_count,
            "end_to_end_recall": correct / len(relations),
            "conditional_recall": correct / eligible if eligible else None,
        }
    return {
        "scope": manifest.get("scope_id"),
        "gold_version": manifest.get("gold_version"),
        "TP": tp, "FP": fp, "FN": fn, "EXCLUDE": excluded,
        "precision": precision, "recall": recall, "F1": f1,
        "structure": structure,
        "order": order_scores,
        "grounding": None, "execution_success": None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["pilot", "full"])
    parser.add_argument("--gold", required=True)
    parser.add_argument("--adjudications", required=True)
    parser.add_argument("--bt")
    parser.add_argument("--bt-root")
    parser.add_argument("--methods")
    parser.add_argument("--manifest")
    parser.add_argument("--edges")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.mode == "pilot":
        if not args.bt_root or not args.methods:
            parser.error("pilot requires --bt-root and --methods")
        rows = pilot_scores(load_csv(args.gold), load_csv(args.adjudications),
                            load_csv(args.methods), args.bt_root)
        if not rows:
            raise ValueError("No pilot predictions matched the supplied method manifest")
        with Path(args.out).open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"Pilot rows: {len(rows)}; no Action F1 claimed")
    else:
        if not args.bt or not args.manifest or not args.edges:
            parser.error("full requires --bt --manifest --edges")
        result = full_score(
            load_csv(args.gold), json.loads(Path(args.manifest).read_text(encoding="utf-8")),
            load_csv(args.adjudications),
            json.loads(Path(args.bt).read_text(encoding="utf-8")),
            load_csv(args.edges),
        )
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
