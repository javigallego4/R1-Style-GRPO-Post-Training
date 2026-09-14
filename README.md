# R1-Style GRPO Post-Training on Kaggle T4x2

This repository contains a reproducible project for post-training a small instruction-tuned language model with GRPO on Kaggle's dual-T4 GPU environment.

The project will be developed using a spec-driven workflow:

1. Define the product requirements in a PRD.
2. Split the work into implementation specs under `specs/`.
3. Implement each component from the approved specs.
4. Run training and evaluation on Kaggle.
5. Publish clear results, limitations, and reproducibility notes.

The goal is not to replicate DeepSeek-R1 at scale. The goal is to build a small, rigorous, reproducible demonstration of modern LLM post-training using TRL, GRPO, LoRA/QLoRA, reward functions, and constrained GPU infrastructure.

Planned scope:

- small open-source instruction model;
- rule-based reward functions for reasoning tasks;
- GRPO post-training pipeline;
- Kaggle T4x2-compatible training configuration;
- pre/post-training evaluation;
- exported LoRA adapter;
- experiment tracking with Weights & Biases.

## Default Setup

- Model: `unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit`
- Dataset: `openai/gsm8k`
- Training: GRPO-only
- Tracking: Weights & Biases
- Adapter policy: save locally, do not publish by default
- Runtime: Unsloth training path enabled; vLLM fast inference disabled by default for Kaggle compatibility

## Output Format

The model is trained to answer with:

```text
<reasoning>
...
</reasoning>
<answer>
42
</answer>
```

## Local Smoke Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
python scripts/validate_project.py
```

## Kaggle Training

Install dependencies in the Kaggle environment, configure W&B through Kaggle secrets or environment variables, then run:

```bash
python scripts/train_kaggle.py --config configs/default.yaml
```

To push this repository as a Kaggle script kernel from local, create metadata first:

```bash
python scripts/create_kaggle_metadata.py --username YOUR_KAGGLE_USERNAME --private --internet
kaggle kernels push -p . --accelerator NvidiaTeslaT4
```

The Kaggle script entrypoint is:

```bash
python kaggle_entry.py
```

By default it uses a strict Unsloth setup for Kaggle: it refreshes the CUDA PyTorch/vLLM stack, installs Unsloth, and then launches the configured GRPO training run. Set `INSTALL_DEPS=0` if dependencies are already available, or `KAGGLE_STRICT_UNSLOTH_SETUP=0` to install from `requirements.txt` instead.

Before a long run, check the Kaggle environment:

```bash
python scripts/check_environment.py
python scripts/inspect_data.py --config configs/smoke.yaml
python scripts/train_kaggle.py --config configs/smoke.yaml
```

The Kaggle entrypoint defaults to `configs/kaggle_smoke.yaml`, which keeps the run short and runs `train_then_evaluate`: it trains a 10-step smoke adapter, then compares the base model and adapter on the configured held-out GSM8K subset.
If a CLI-pushed version cannot see `WANDB_API_KEY`, open the Kaggle notebook UI, enable the secret again in Add-ons > Secrets, and rerun the same version from Kaggle.
When the key is unavailable, the entrypoint falls back to disabled tracking and still writes local training and evaluation outputs.

To test W&B without starting training, run this from the Kaggle notebook UI after enabling the `WANDB_API_KEY` secret:

```bash
WANDB_PROBE_ONLY=1 python kaggle_entry.py
```

The expected successful log includes `Direct Kaggle secret bootstrap: WANDB_API_KEY loaded`, `api_key_available: True`, and a W&B probe result with `ok: True`.

The Kaggle entrypoint can also be controlled with:

```bash
RUN_MODE=train python kaggle_entry.py
RUN_MODE=evaluate ADAPTER_PATH=outputs/kaggle-smoke-adapter python kaggle_entry.py
RUN_MODE=train_then_evaluate python kaggle_entry.py
CONFIG_PATH=configs/kaggle_pilot.yaml python kaggle_entry.py
```

For a first non-smoke pilot run, use `configs/kaggle_pilot.yaml`: 100 GRPO steps on 256 GSM8K training examples, adapter export, and a 16-example base-vs-adapter evaluation. From the Kaggle UI, set `CONFIG_PATH=configs/kaggle_pilot.yaml` before running `kaggle_entry.py`; for CLI-pushed kernels, use a one-off local default override before pushing.

Once the smoke run finishes, switch to the default configuration:

```bash
python scripts/train_kaggle.py --config configs/default.yaml
```

For evaluation:

```bash
python scripts/evaluate.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml --adapter-path outputs/adapter
```

To compare the base model and a trained adapter with the same held-out examples:

```bash
python scripts/evaluate_comparison.py \
  --config configs/kaggle_smoke.yaml \
  --adapter-path outputs/kaggle-smoke-adapter
```

This writes JSON, CSV, and Markdown outputs under the configured `evaluation.output_dir`, including base metrics, adapter metrics, metric deltas, reward-component deltas, and qualitative examples. Add `--final` to use the final evaluation size, or `--no-wandb` to skip W&B evaluation logging.

## Configuration

All important choices live in `configs/default.yaml`, including model, dataset size, GRPO settings, reward weights, W&B settings, and adapter export behavior.

W&B is controlled from the `tracking` section. Keep `tracking.enabled: false` for local smoke tests without login, use `tracking.mode: offline` when you want local W&B files only, and use `tracking.mode: online` on Kaggle when the `WANDB_API_KEY` secret is available.

Each training run also writes a sanitized `run_manifest.json` inside the configured training output directory, so partial Kaggle runs leave local status and config metadata even if external tracking is unavailable.

## Credentials

Credential setup is documented in `docs/credentials.md`. Real tokens and private keys must stay outside the repository.
