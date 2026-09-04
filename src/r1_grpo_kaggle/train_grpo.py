from __future__ import annotations

import argparse
import os
from pathlib import Path

from .config import load_config
from .data import prepare_train_dataset
from .rewards import reward_functions
from .tracking import is_wandb_enabled, wandb_project_name, wandb_run_name


def build_lora_config(config: dict):
    from peft import LoraConfig

    lora_cfg = config["lora"]
    return LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type="CAUSAL_LM",
    )


def load_policy_model(config: dict):
    model_cfg = config["model"]
    if model_cfg.get("use_unsloth", True):
        from unsloth import FastLanguageModel, is_bfloat16_supported

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_cfg["name"],
            max_seq_length=model_cfg["max_seq_length"],
            load_in_4bit=model_cfg.get("load_in_4bit", True),
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=config["lora"]["r"],
            target_modules=config["lora"]["target_modules"],
            lora_alpha=config["lora"]["alpha"],
            lora_dropout=config["lora"]["dropout"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=int(config.get("project", {}).get("seed", 42)),
        )
        return model, tokenizer, is_bfloat16_supported(), True

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], device_map="auto")
    return model, tokenizer, False, False


def build_grpo_config(config: dict) -> GRPOConfig:
    from trl import GRPOConfig

    training = config["training"]
    report_to = ["wandb"] if is_wandb_enabled(config) else []
    if is_wandb_enabled(config):
        os.environ.setdefault("WANDB_PROJECT", wandb_project_name(config))
    reward_cfg = config.get("rewards", {})
    reward_weights = [
        reward_cfg.get("correctness", 1.0),
        reward_cfg.get("integer_answer", 1.0),
        reward_cfg.get("strict_format", 1.0),
        reward_cfg.get("soft_format", 1.0),
        reward_cfg.get("reasoning_tags", 1.0),
    ]
    return GRPOConfig(
        output_dir=training["output_dir"],
        max_steps=training["max_steps"],
        learning_rate=training["learning_rate"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        generation_batch_size=training.get("generation_batch_size"),
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        num_generations=training["num_generations"],
        max_prompt_length=training["max_prompt_length"],
        max_completion_length=training["max_completion_length"],
        temperature=training["temperature"],
        top_p=training["top_p"],
        logging_steps=training["logging_steps"],
        save_steps=training["save_steps"],
        save_total_limit=training["save_total_limit"],
        optim=training.get("optim", "paged_adamw_8bit"),
        bf16=bool(training.get("bf16", False)),
        fp16=bool(training.get("fp16", True)),
        seed=int(config.get("project", {}).get("seed", 42)),
        report_to=report_to,
        run_name=wandb_run_name(config),
        log_completions=bool(config.get("tracking", {}).get("log_samples", True)),
        reward_weights=reward_weights,
        push_to_hub=False,
    )


def train(config_path: str) -> None:
    if load_config(config_path)["model"].get("use_unsloth", True):
        import unsloth  # noqa: F401

    config = load_config(config_path)
    train_dataset = prepare_train_dataset(config)
    model, tokenizer, bf16_supported, model_already_has_peft = load_policy_model(config)
    config["training"]["bf16"] = bf16_supported
    config["training"]["fp16"] = not bf16_supported
    trainer_kwargs = {}
    if not model_already_has_peft:
        trainer_kwargs["peft_config"] = build_lora_config(config)
    from trl import GRPOTrainer

    trainer = GRPOTrainer(
        model=model,
        args=build_grpo_config(config),
        reward_funcs=reward_functions(),
        train_dataset=train_dataset,
        processing_class=tokenizer,
        **trainer_kwargs,
    )
    trainer.train()

    export_cfg = config["export"]
    if export_cfg.get("save_adapter", True):
        adapter_dir = Path(export_cfg["adapter_dir"])
        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(adapter_dir)

    if export_cfg.get("publish_adapter", False):
        raise RuntimeError("Adapter publication is intentionally disabled in v1.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
