from decimal import Decimal
import unittest

from r1_grpo_kaggle.rewards import (
    correctness_reward_func,
    extract_answer_block,
    extract_gsm8k_answer,
    has_soft_format,
    has_strict_format,
    integer_answer_reward_func,
    normalize_number,
    reasoning_tags_reward_func,
)


class RewardTests(unittest.TestCase):
    def test_extract_gsm8k_answer_uses_final_marker(self):
        answer = "There are 2 + 3 = <<2+3=5>>5 apples. #### 5"
        self.assertEqual(extract_gsm8k_answer(answer), "5")

    def test_extract_gsm8k_answer_falls_back_to_last_number(self):
        self.assertEqual(extract_gsm8k_answer("First 10, then 20."), "20")

    def test_extract_answer_block_prefers_xml_answer(self):
        completion = "<reasoning>2 + 3</reasoning><answer>5</answer>"
        self.assertEqual(extract_answer_block(completion), "5")

    def test_normalize_number_handles_commas(self):
        self.assertEqual(normalize_number("1,234"), Decimal("1234"))

    def test_correctness_reward_matches_numeric_answer(self):
        rewards = correctness_reward_func(completions=["<answer>5</answer>"], answer=["5"])
        self.assertEqual(rewards, [1.0])

    def test_integer_answer_reward_requires_parseable_integer(self):
        rewards = integer_answer_reward_func(
            completions=["<answer>5</answer>", "<answer>5.5</answer>"]
        )
        self.assertEqual(rewards, [1.0, 0.0])

    def test_format_detection(self):
        completion = "<reasoning>2 + 3 = 5</reasoning><answer>5</answer>"
        self.assertTrue(has_soft_format(completion))
        self.assertTrue(has_strict_format(completion))

    def test_reasoning_tags_reward_counts_tags(self):
        rewards = reasoning_tags_reward_func(
            completions=["<reasoning>x</reasoning><answer>1</answer>"]
        )
        self.assertEqual(rewards, [1.0])


if __name__ == "__main__":
    unittest.main()
