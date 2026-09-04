import os
import unittest
from unittest.mock import patch

from r1_grpo_kaggle.tracking import (
    configure_wandb,
    configured_secret_names,
    ensure_wandb_api_key,
    sanitized_tracking_config,
    wandb_init_kwargs,
)


class TrackingTests(unittest.TestCase):
    def test_disabled_tracking_does_not_set_wandb_environment(self):
        config = {
            "project": {"name": "r1-grpo-kaggle"},
            "model": {"name": "unsloth/test-model"},
            "dataset": {"name": "openai/gsm8k"},
            "tracking": {"enabled": False, "project_name": "custom-project"},
        }
        with patch.dict(os.environ, {}, clear=True):
            configure_wandb(config)
            self.assertNotIn("WANDB_PROJECT", os.environ)
            self.assertNotIn("WANDB_RUN_NAME", os.environ)

    def test_configure_wandb_sets_project_and_stable_run_name(self):
        config = {
            "project": {"name": "r1-grpo-kaggle"},
            "model": {"name": "unsloth/test-model"},
            "dataset": {"name": "openai/gsm8k"},
            "tracking": {"enabled": True, "project_name": "custom-project", "run_name": "small-run"},
        }
        with patch.dict(os.environ, {}, clear=True):
            configure_wandb(config)
            self.assertEqual(os.environ["WANDB_PROJECT"], "custom-project")
            self.assertEqual(os.environ["WANDB_RUN_NAME"], "small-run")

    def test_configure_wandb_sets_optional_environment_values(self):
        config = {
            "project": {"name": "r1-grpo-kaggle"},
            "model": {"name": "unsloth/test-model"},
            "dataset": {"name": "openai/gsm8k"},
            "tracking": {
                "enabled": True,
                "project_name": "custom-project",
                "entity": "custom-entity",
                "run_name": "small-run",
                "mode": "offline",
            },
        }
        with patch.dict(os.environ, {}, clear=True):
            configure_wandb(config)
            self.assertEqual(os.environ["WANDB_ENTITY"], "custom-entity")
            self.assertEqual(os.environ["WANDB_MODE"], "offline")

    def test_configured_secret_names_uses_yaml_values(self):
        config = {"tracking": {"secret_names": ["WANDB_API_KEY", "CUSTOM_WANDB"]}}
        self.assertEqual(configured_secret_names(config), ("WANDB_API_KEY", "CUSTOM_WANDB"))

    def test_ensure_wandb_api_key_accepts_standard_environment_variable(self):
        with patch.dict(os.environ, {"WANDB_API_KEY": "secret-value"}, clear=True):
            self.assertTrue(ensure_wandb_api_key())
            self.assertEqual(os.environ["WANDB_API_KEY"], "secret-value")

    def test_ensure_wandb_api_key_accepts_kaggle_style_alias(self):
        with patch.dict(os.environ, {"wandb_api_key": "secret-value"}, clear=True):
            self.assertTrue(ensure_wandb_api_key())
            self.assertEqual(os.environ["WANDB_API_KEY"], "secret-value")

    def test_sanitized_tracking_config_removes_secret_like_tracking_keys(self):
        config = {
            "tracking": {
                "enabled": True,
                "api_key": "secret-value",
                "token": "secret-token",
                "project_name": "r1-grpo-kaggle",
                "secret_names": ["WANDB_API_KEY"],
            }
        }
        safe_config = sanitized_tracking_config(config)
        self.assertNotIn("api_key", safe_config["tracking"])
        self.assertNotIn("token", safe_config["tracking"])
        self.assertNotIn("secret_names", safe_config["tracking"])
        self.assertEqual(safe_config["tracking"]["project_name"], "r1-grpo-kaggle")

    def test_wandb_init_kwargs_uses_config_metadata(self):
        config = {
            "project": {"name": "r1-grpo-kaggle"},
            "model": {"name": "unsloth/test-model"},
            "dataset": {"name": "openai/gsm8k"},
            "tracking": {
                "enabled": True,
                "project_name": "custom-project",
                "entity": "custom-entity",
                "run_name": "small-run",
                "group": "ablation",
                "job_type": "grpo-training",
                "tags": ["grpo", "smoke"],
                "notes": "short run",
                "mode": "offline",
                "secret_names": ["WANDB_API_KEY"],
            },
        }
        kwargs = wandb_init_kwargs(config)
        self.assertEqual(kwargs["project"], "custom-project")
        self.assertEqual(kwargs["entity"], "custom-entity")
        self.assertEqual(kwargs["name"], "small-run")
        self.assertEqual(kwargs["group"], "ablation")
        self.assertEqual(kwargs["job_type"], "grpo-training")
        self.assertEqual(kwargs["tags"], ["grpo", "smoke"])
        self.assertEqual(kwargs["notes"], "short run")
        self.assertEqual(kwargs["mode"], "offline")
        self.assertNotIn("secret_names", kwargs["config"]["tracking"])


if __name__ == "__main__":
    unittest.main()
