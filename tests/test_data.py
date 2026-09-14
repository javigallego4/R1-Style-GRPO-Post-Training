import unittest

from r1_grpo_kaggle.data import build_prompt, summarize_prepared_dataset


class DataTests(unittest.TestCase):
    def test_build_prompt_includes_system_and_question(self):
        config = {
            "prompt": {
                "system": "System message.",
                "template": "Problem: {question}",
            }
        }
        prompt = build_prompt("2 + 2?", config)
        self.assertIn("System message.", prompt)
        self.assertIn("Problem: 2 + 2?", prompt)

    def test_summarize_prepared_dataset_counts_unparsed_answers(self):
        dataset = [
            {"answer_parse_failed": False},
            {"answer_parse_failed": True},
            {"answer_parse_failed": False},
        ]

        summary = summarize_prepared_dataset(dataset)

        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["malformed_answer_count"], 1)
        self.assertAlmostEqual(summary["malformed_answer_rate"], 1 / 3)

    def test_summarize_prepared_dataset_handles_empty_dataset(self):
        summary = summarize_prepared_dataset([])

        self.assertEqual(summary["sample_count"], 0)
        self.assertEqual(summary["malformed_answer_count"], 0)
        self.assertEqual(summary["malformed_answer_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
