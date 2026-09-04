import unittest

from r1_grpo_kaggle.data import build_prompt


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


if __name__ == "__main__":
    unittest.main()

