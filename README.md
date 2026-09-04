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
kaggle kernels push -p .
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

Once the smoke run finishes, switch to the default configuration:

```bash
python scripts/train_kaggle.py --config configs/default.yaml
```

For evaluation:

```bash
python scripts/evaluate.py --config configs/default.yaml
python scripts/evaluate.py --config configs/default.yaml --adapter-path outputs/adapter
```

## Configuration

All important choices live in `configs/default.yaml`, including model, dataset size, GRPO settings, reward weights, W&B settings, and adapter export behavior.

## Credentials

Credential setup is documented in `docs/credentials.md`. Real tokens and private keys must stay outside the repository.
