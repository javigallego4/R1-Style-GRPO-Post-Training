import unittest
import tempfile
from pathlib import Path

from r1_grpo_kaggle.evaluate import (
    build_evaluation_row,
    compare_evaluation_results,
    summarize_evaluation_rows,
    write_comparison_markdown,
    write_comparison_metrics_csv,
    write_evaluation_result,
)


class EvaluationTests(unittest.TestCase):
    def test_build_evaluation_row_records_correct_parse_and_rewards(self):
        sample = {
            "question": "What is 2 + 3?",
            "answer": "5",
        }
        completion = "<reasoning>2 + 3 = 5</reasoning><answer>5</answer>"

        row = build_evaluation_row(sample, completion)

        self.assertTrue(row["correct"])
        self.assertFalse(row["parse_failed"])
        self.assertEqual(row["predicted_answer"], "5")
        self.assertEqual(row["expected_answer"], "5")
        self.assertEqual(row["rewards"]["correctness"], 1.0)
        self.assertEqual(row["rewards"]["strict_format"], 1.0)

    def test_build_evaluation_row_marks_parse_failure(self):
        sample = {
            "question": "What is 2 + 3?",
            "answer": "5",
        }
        completion = "I do not know."

        row = build_evaluation_row(sample, completion)

        self.assertFalse(row["correct"])
        self.assertTrue(row["parse_failed"])
        self.assertIsNone(row["predicted_answer"])

    def test_summarize_evaluation_rows_computes_rates_and_reward_means(self):
        rows = [
            {
                "correct": True,
                "parse_failed": False,
                "rewards": {
                    "correctness": 1.0,
                    "strict_format": 1.0,
                    "soft_format": 1.0,
                },
            },
            {
                "correct": False,
                "parse_failed": True,
                "rewards": {
                    "correctness": 0.0,
                    "strict_format": 0.0,
                    "soft_format": 0.0,
                },
            },
        ]

        metrics = summarize_evaluation_rows(rows)

        self.assertEqual(metrics["sample_count"], 2)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["parse_failure_rate"], 0.5)
        self.assertEqual(metrics["strict_format_rate"], 0.5)
        self.assertEqual(metrics["soft_format_rate"], 0.5)
        self.assertEqual(metrics["reward_means"]["correctness"], 0.5)

    def test_summarize_evaluation_rows_handles_empty_input(self):
        metrics = summarize_evaluation_rows([])

        self.assertEqual(metrics["sample_count"], 0)
        self.assertEqual(metrics["accuracy"], 0.0)
        self.assertEqual(metrics["parse_failure_rate"], 0.0)
        self.assertEqual(metrics["reward_means"], {})

    def test_compare_evaluation_results_computes_metric_and_reward_deltas(self):
        base_result = {
            "label": "base",
            "adapter_path": None,
            "metrics": {
                "accuracy": 0.25,
                "parse_failure_rate": 0.50,
                "reward_means": {"correctness": 0.25},
            },
        }
        adapter_result = {
            "label": "adapter",
            "adapter_path": "outputs/adapter",
            "metrics": {
                "accuracy": 0.75,
                "parse_failure_rate": 0.25,
                "reward_means": {"correctness": 0.75},
            },
        }

        comparison = compare_evaluation_results(base_result, adapter_result)

        self.assertEqual(comparison["deltas"]["accuracy"], 0.5)
        self.assertEqual(comparison["deltas"]["parse_failure_rate"], -0.25)
        self.assertEqual(comparison["reward_deltas"]["correctness"], 0.5)
        self.assertEqual(comparison["adapter"]["adapter_path"], "outputs/adapter")

    def test_write_evaluation_and_comparison_outputs(self):
        result = {
            "label": "base",
            "adapter_path": None,
            "metrics": {"accuracy": 1.0, "reward_means": {"correctness": 1.0}},
            "examples": [
                {
                    "question": "What is 2 + 3?",
                    "predicted_answer": "5",
                    "expected_answer": "5",
                    "correct": True,
                    "parse_failed": False,
                    "completion": "<answer>5</answer>",
                }
            ],
        }
        comparison = {
            "base": {"label": "base", "adapter_path": None, "metrics": result["metrics"]},
            "adapter": {
                "label": "adapter",
                "adapter_path": "outputs/adapter",
                "metrics": result["metrics"],
            },
            "deltas": {"accuracy": 0.0},
            "reward_deltas": {"correctness": 0.0},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            paths = write_evaluation_result(result, output_dir, "quick")
            metrics_csv = write_comparison_metrics_csv(
                output_dir / "quick_comparison_metrics.csv",
                comparison,
            )
            markdown = write_comparison_markdown(
                output_dir / "quick_comparison.md",
                comparison,
            )

            self.assertTrue(Path(paths["json"]).exists())
            self.assertTrue(Path(paths["examples_csv"]).exists())
            self.assertIn("accuracy", metrics_csv.read_text(encoding="utf-8"))
            self.assertIn("Evaluation Comparison", markdown.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
