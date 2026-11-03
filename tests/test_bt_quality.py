"""BT质量评分单元测试：人工Gold与结构评价不可以互相替代。"""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bt_quality.score_bt import check_bt, full_score, simple_sequence_order
from bt_quality.compare import evaluate_rows, write_results


class BTQualityTests(unittest.TestCase):
    def setUp(self):
        self.bt = {
            "video_id": "video_A",
            "bt": {"type": "Sequence", "children": [
                {"type": "Action", "event_id": "e1"},
                {"type": "Action", "event_id": "e2"},
            ]},
            "events": [{"event_id": "e1"}, {"event_id": "e2"}],
        }
        self.gold = [{"reference_id": "g1"}, {"reference_id": "g2"}]
        self.review = [
            {"event_id": "e1", "status": "TP", "reference_id": "g1"},
            {"event_id": "e2", "status": "TP", "reference_id": "g2"},
        ]
        self.manifest = {
            "annotation_status": "LOCKED",
            "scope_complete": True,
            "video_id": "video_A",
            "gold_version": "test-gold",
        }
        self.edges = [{
            "relation_type": "OBSERVED_BEFORE",
            "from_reference_id": "g1", "to_reference_id": "g2",
        }]

    def test_perfect_actions_and_order(self):
        result = full_score(self.gold, self.manifest, self.review, self.bt, self.edges)
        self.assertEqual((result["TP"], result["FP"], result["FN"]), (2, 0, 0))
        self.assertEqual(result["F1"], 1.0)
        self.assertEqual(result["order"]["OBSERVED_BEFORE"]["end_to_end_recall"], 1.0)
        self.assertIsNone(result["execution_success"])

    def test_reversed_order_keeps_f1_but_loses_order(self):
        self.bt["bt"]["children"].reverse()
        result = full_score(self.gold, self.manifest, self.review, self.bt, self.edges)
        self.assertEqual(result["F1"], 1.0)
        self.assertEqual(result["order"]["OBSERVED_BEFORE"]["end_to_end_recall"], 0.0)

    def test_missing_action_is_fn(self):
        self.review[1] = {"event_id": "e2", "status": "FP", "reference_id": ""}
        result = full_score(self.gold, self.manifest, self.review, self.bt, self.edges)
        self.assertEqual((result["TP"], result["FP"], result["FN"]), (1, 1, 1))
        self.assertAlmostEqual(result["F1"], 0.5)

    def test_unlocked_gold_refused(self):
        self.manifest["scope_complete"] = False
        with self.assertRaisesRegex(ValueError, "REFUSE_FULL_SCORE"):
            full_score(self.gold, self.manifest, self.review, self.bt, self.edges)

    def test_unreviewed_action_refused(self):
        with self.assertRaisesRegex(ValueError, "Every predicted"):
            full_score(self.gold, self.manifest, self.review[:1], self.bt, self.edges)

    def test_duplicate_gold_match_refused(self):
        self.review[1]["reference_id"] = "g1"
        with self.assertRaisesRegex(ValueError, "uniquely"):
            full_score(self.gold, self.manifest, self.review, self.bt, self.edges)

    def test_wrong_video_refused(self):
        self.manifest["video_id"] = "video_B"
        with self.assertRaisesRegex(ValueError, "video_id"):
            full_score(self.gold, self.manifest, self.review, self.bt, self.edges)

    def test_structure_checks_action_ref(self):
        self.bt["bt"]["children"][1]["event_id"] = "missing"
        self.assertFalse(check_bt(self.bt["bt"], self.bt["events"])["structural_valid"])

    def test_branch_is_not_flattened(self):
        self.assertIsNone(simple_sequence_order({
            "type": "Fallback", "children": [{"type": "Action", "event_id": "e1"}]
        }))

    def test_missing_legacy_bt_is_not_a_zero_score(self):
        with TemporaryDirectory() as folder:
            result = evaluate_rows([
                {"video_id": "P08", "method": "LEGACY", "K": "40",
                 "job_id": "", "source": "OLD"}
            ], Path(folder))
            self.assertEqual(result[0]["quality_status"], "MISSING_FULL_BT")
            self.assertEqual(result[0]["action_f1"], "")
            summary = write_results(result, Path(folder) / "output", {})
            self.assertEqual(summary["total_method_rows"], 1)
            self.assertTrue((Path(folder) / "output" / "BT_METHOD_COMPARISON.csv").exists())


if __name__ == "__main__":
    unittest.main()
