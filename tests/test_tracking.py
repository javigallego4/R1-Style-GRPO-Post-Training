import os
import builtins
import tempfile
import unittest
from unittest.mock import patch

from r1_grpo_kaggle.tracking import (
    configure_wandb,
    configured_secret_names,
    ensure_wandb_api_key,
    kaggle_secret_diagnostics,
    require_wandb_api_key,
    run_wandb_probe,
    sanitized_tracking_config,
    wandb_status,
    wandb_init_kwargs,
    write_run_manifest,
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

    @patch("builtins.__import__", side_effect=ImportError)
    def test_kaggle_secret_diagnostics_handles_missing_package(self, _mock_import):
        diagnostics = kaggle_secret_diagnostics(("WANDB_API_KEY",))
        self.assertFalse(diagnostics["kaggle_secrets_available"])
        self.assertEqual(diagnostics["checks"], [])

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

    def test_wandb_status_reports_key_presence_without_secret_value(self):
        config = {
            "project": {"name": "r1-grpo-kaggle"},
            "model": {"name": "unsloth/test-model"},
            "dataset": {"name": "openai/gsm8k"},
            "tracking": {
                "enabled": True,
                "project_name": "custom-project",
                "run_name": "small-run",
                "mode": "online",
                "secret_names": ["WANDB_API_KEY"],
            },
        }
        with patch.dict(os.environ, {"WANDB_API_KEY": "secret-value"}, clear=True):
            status = wandb_status(config)
        self.assertTrue(status["api_key_available"])
        self.assertNotIn("secret-value", str(status))

    def test_require_wandb_api_key_fails_clearly_for_online_without_key(self):
        config = {
            "project": {"name": "r1-grpo-kaggle"},
            "model": {"name": "unsloth/test-model"},
            "dataset": {"name": "openai/gsm8k"},
            "tracking": {
                "enabled": True,
                "project_name": "custom-project",
                "run_name": "small-run",
                "mode": "online",
                "secret_names": ["WANDB_API_KEY"],
            },
        }
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "WANDB_API_KEY"):
                require_wandb_api_key(config)

    def test_write_run_manifest_saves_sanitized_local_status(self):
        config = {
            "project": {"name": "r1-grpo-kaggle"},
            "model": {"name": "unsloth/test-model"},
            "dataset": {"name": "openai/gsm8k"},
            "tracking": {
                "enabled": True,
                "project_name": "custom-project",
                "run_name": "small-run",
                "mode": "offline",
                "api_key": "secret-value",
                "secret_names": ["WANDB_API_KEY"],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_path = write_run_manifest(config, tmpdir, "failed", error="boom")
            manifest = manifest_path.read_text(encoding="utf-8")
        self.assertIn('"status": "failed"', manifest)
        self.assertIn('"error": "boom"', manifest)
        self.assertNotIn("secret-value", manifest)
        self.assertNotIn("secret_names", manifest)

    @patch.dict(os.environ, {"WANDB_API_KEY": "secret-value"}, clear=True)
    @patch("r1_grpo_kaggle.tracking.wandb_init_kwargs")
    @patch("builtins.__import__")
    def test_run_wandb_probe_logs_minimal_metric(self, mock_import, mock_kwargs):
        class FakeRun:
            url = "https://wandb.ai/example/run"

            def __init__(self):
                self.finished = False

            def finish(self):
                self.finished = True

        class FakeWandb:
            def __init__(self):
                self.run = FakeRun()
                self.logged = []

            def init(self, **_kwargs):
                return self.run

            def log(self, payload, step=None):
                self.logged.append((payload, step))

        fake_wandb = FakeWandb()
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "wandb":
                return fake_wandb
            return real_import(name, *args, **kwargs)

        mock_import.side_effect = fake_import
        mock_kwargs.return_value = {"project": "custom-project", "name": "small-run"}
        config = {
            "project": {"name": "r1-grpo-kaggle"},
            "model": {"name": "unsloth/test-model"},
            "dataset": {"name": "openai/gsm8k"},
            "tracking": {
                "enabled": True,
                "project_name": "custom-project",
                "run_name": "small-run",
                "mode": "online",
                "secret_names": ["WANDB_API_KEY"],
            },
        }

        result = run_wandb_probe(config)

        self.assertTrue(result["ok"])
        self.assertEqual(result["run_url"], "https://wandb.ai/example/run")
        self.assertEqual(fake_wandb.logged, [({"probe/ok": 1, "probe/enabled": 1}, 0)])
        self.assertTrue(fake_wandb.run.finished)


if __name__ == "__main__":
    unittest.main()
