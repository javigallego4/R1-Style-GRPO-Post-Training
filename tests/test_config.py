import unittest

from r1_grpo_kaggle.config import load_config


class ConfigTests(unittest.TestCase):
    def test_default_config_loads(self):
        config = load_config("/Users/javigallego/Desktop/r1-grpo-kaggle/configs/default.yaml")
        self.assertEqual(
            config["model"]["name"],
            "unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit",
        )
        self.assertFalse(config["export"]["publish_adapter"])
        self.assertTrue(config["tracking"]["enabled"])
        self.assertEqual(config["tracking"]["mode"], "online")
        self.assertIn("WANDB_API_KEY", config["tracking"]["secret_names"])
        self.assertFalse(config["tracking"]["log_adapter_artifacts"])

    def test_smoke_config_loads(self):
        config = load_config("/Users/javigallego/Desktop/r1-grpo-kaggle/configs/smoke.yaml")
        self.assertEqual(config["training"]["max_steps"], 10)
        self.assertEqual(config["training"]["generation_batch_size"], 2)
        self.assertFalse(config["tracking"]["enabled"])
        self.assertEqual(config["tracking"]["mode"], "disabled")
        self.assertFalse(config["export"]["publish_adapter"])

    def test_kaggle_smoke_config_enables_tracking(self):
        config = load_config("/Users/javigallego/Desktop/r1-grpo-kaggle/configs/kaggle_smoke.yaml")
        self.assertEqual(config["training"]["max_steps"], 10)
        self.assertTrue(config["tracking"]["enabled"])
        self.assertEqual(config["tracking"]["mode"], "online")
        self.assertEqual(config["tracking"]["run_name"], "kaggle-smoke")
        self.assertFalse(config["tracking"]["log_adapter_artifacts"])
        self.assertFalse(config["export"]["publish_adapter"])


if __name__ == "__main__":
    unittest.main()
