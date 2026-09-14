import unittest

from r1_grpo_kaggle.evaluate import (
    build_evaluation_row,
    summarize_evaluation_rows,
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


if __name__ == "__main__":
    unittest.main()
