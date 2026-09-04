# PRD: R1-Style GRPO Post-Training on Kaggle T4x2

## Status

Draft

## Objective

Build a reproducible project that post-trains a small instruction-tuned language model with GRPO on Kaggle's dual-T4 GPU environment.

The project should demonstrate a realistic modern LLM post-training workflow under constrained compute:

- loading a small open-source base model;
- preparing a reasoning-focused dataset;
- defining rule-based reward functions;
- training a LoRA/QLoRA adapter with GRPO;
- tracking experiments with Weights & Biases;
- evaluating behavior before and after post-training;
- exporting artifacts in a reusable and documented format.

The project is intended to be technically defensible, reproducible, and suitable for public GitHub presentation.

## Background And Motivation

Modern LLM engineering increasingly depends on post-training techniques rather than only prompt engineering or supervised fine-tuning. GRPO is a practical algorithm for reasoning-oriented post-training because it can optimize model outputs using automatically computed rewards without requiring a separate critic model.

This project will use a small model and a constrained GPU setup to show the core mechanics of that workflow without pretending to reproduce frontier-scale training.

## Target User

The primary user is a developer or ML practitioner who wants to understand, run, and inspect a compact GRPO post-training pipeline.

Secondary users include technical recruiters, hiring managers, and engineers reviewing the repository as evidence of hands-on LLM engineering experience.

## Scope

The project will include:

- a local repository structure for development;
- Kaggle-compatible training entrypoints;
- configuration files for model, dataset, training, rewards, logging, and evaluation;
- data preparation code for reasoning tasks;
- reward functions for correctness, output format, and reasoning structure;
- GRPO training using TRL;
- efficient fine-tuning through LoRA/QLoRA;
- optional acceleration through Unsloth when compatible with the selected model and runtime;
- experiment tracking with Weights & Biases;
- evaluation scripts for base model and trained adapter;
- saved metrics and representative examples;
- documentation explaining setup, execution, results, and limitations.

## Out Of Scope

The project will not include:

- full pretraining of an LLM;
- large-scale reproduction of frontier reasoning models;
- multi-node training;
- paid GPU infrastructure as a required path;
- production serving infrastructure;
- a web application;
- CV-specific writing, publication strategy, or LinkedIn content inside technical specs;
- references to private inspiration material.

## Success Criteria

The project is successful when:

- the repository can be understood from the README without private context;
- a fresh Kaggle environment can run the documented training path;
- the training path fits within Kaggle T4x2 constraints for the selected default configuration;
- Weights & Biases captures training metrics and run configuration;
- evaluation can compare the base model against the trained adapter;
- results include quantitative metrics and qualitative samples;
- the adapter can be saved and reused for inference;
- the adapter is not published by default;
- limitations are clearly documented.

## Functional Requirements

### Model

- The default model must be small enough to run on Kaggle T4x2 with LoRA/QLoRA.
- The v1 default model is `unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit`.
- The project must allow changing the model from configuration without editing core training code.

### Dataset

- The v1 default dataset is `openai/gsm8k`.
- The default task should allow automatic reward computation.
- Dataset preparation must use the training split for post-training and leave the validation/test split for later evaluation.
- The initial training configuration should use a capped subset of 1,000-2,000 GSM8K training examples.
- Quick evaluation should use 100-200 held-out examples.
- Final evaluation should support 500+ held-out examples when runtime allows.

### Reward Functions

- The reward layer must be modular.
- At minimum, the project should support:
  - correctness reward;
  - output format reward;
  - reasoning structure reward.
- Rewards should be logged separately where possible, not only as a combined scalar.
- Reward functions must be deterministic for the same model output and target.
- The default output template is:

```text
<reasoning>
...
</reasoning>
<answer>
42
</answer>
```

### Training

- Training must use GRPO through TRL.
- The v1 training path is GRPO-only, with no supervised fine-tuning warm-up stage.
- Fine-tuning should use LoRA or QLoRA rather than updating all model weights.
- Training must support configuration-driven hyperparameters.
- The training entrypoint must be suitable for Kaggle execution.
- Checkpointing or adapter saving must be supported.
- Adapter publication must be controlled by configuration and disabled by default.

### Experiment Tracking

- The project must integrate with Weights & Biases.
- The default W&B project name is `r1-grpo-kaggle`.
- W&B should record:
  - run name;
  - model name;
  - dataset name/version;
  - training configuration;
  - reward metrics;
  - loss/training metrics exposed by the trainer;
  - evaluation metrics;
  - selected generated samples.
- The project must not hardcode private W&B credentials.
- W&B should make it easy to diagnose whether total reward is improving because of correctness or only because of formatting rewards.

### Evaluation

- The project must include pre-training and post-training evaluation.
- Evaluation must measure task correctness where possible.
- Evaluation should include a small set of representative examples.
- Results must be saved in a machine-readable format and summarized for humans.

### Reproducibility

- The repository must include clear setup instructions.
- Configuration should be versioned.
- Random seeds should be controlled where practical.
- Known environment limitations should be documented.

## Non-Functional Requirements

### Compute Constraints

- The default training path must be designed for Kaggle T4x2.
- Memory usage should be controlled through quantization, LoRA/QLoRA, small batch sizes, and gradient accumulation.
- The default run should avoid assumptions that require A100/H100-class hardware.

### Maintainability

- Core logic should be split into focused modules:
  - data loading/preparation;
  - reward functions;
  - training;
  - evaluation;
  - configuration handling;
  - experiment tracking.
- The primary interface should be script-first.
- Kaggle execution should run repository scripts directly where practical.

### Portability

- Local execution should support smoke tests on CPU or small samples where feasible.
- Kaggle execution should be the main GPU path.
- The project should avoid environment-specific hacks unless documented.

### Public-Repo Cleanliness

- The repository must not include private tokens, credentials, API keys, or local-only paths.
- Documentation should be self-contained.
- External inspiration should not be cited or linked unless intentionally part of the public project narrative.

## Proposed Specs

The implementation should be split into these specs:

- `specs/data-and-tasks.md`
- `specs/reward-functions.md`
- `specs/grpo-training-pipeline.md`
- `specs/kaggle-runtime.md`
- `specs/evaluation-and-results.md`
- `specs/experiment-tracking.md`

## Likely Repository Structure

```text
r1-grpo-kaggle/
  README.md
  PRD.md
  configs/
    default.yaml
  specs/
    data-and-tasks.md
    reward-functions.md
    grpo-training-pipeline.md
    kaggle-runtime.md
    evaluation-and-results.md
    experiment-tracking.md
  src/
    data.py
    rewards.py
    train_grpo.py
    evaluate.py
    tracking.py
    config.py
  results/
    .gitkeep
```

## Technical Assumptions

- Python will be the implementation language.
- TRL will provide the GRPO trainer.
- Hugging Face Transformers will provide model/tokenizer loading.
- PEFT will provide LoRA/QLoRA adapter training.
- Unsloth will be used for the default model path where compatible with Kaggle.
- Weights & Biases will be used for experiment tracking.
- Kaggle will be the default GPU execution environment.
- The first complete version should favor reliability over model size.

## Confirmed Decisions

- Default model: `unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit`.
- Default dataset: `openai/gsm8k`.
- Dataset split policy: train on the training split and reserve the validation/test split for evaluation.
- Dataset size policy: start with 1,000-2,000 training examples, 100-200 quick evaluation examples, and 500+ final evaluation examples when runtime allows.
- Training approach: GRPO-only for v1.
- Execution style: script-first, with Kaggle used as the remote GPU runtime.
- Experiment tracking: Weights & Biases is required from v1.
- Default W&B project name: `r1-grpo-kaggle`.
- Default output format: `<reasoning>...</reasoning><answer>...</answer>`.
- Adapter policy: save adapter artifacts locally, but do not publish them by default.
- Public documentation must not reference private inspiration material.

## Training Signal Expectations

The project should evaluate training progress with component-level metrics, not only aggregate reward.

Required tracked signals include:

- total reward;
- reward standard deviation;
- completion length;
- KL divergence;
- correctness reward;
- integer answer reward;
- format rewards;
- reasoning-structure rewards;
- evaluation accuracy before and after training.

Healthy training should show improvement in correctness or evaluation accuracy, not only improvement in formatting rewards. A run where format-related rewards improve but correctness remains flat should be treated as partial learning rather than a successful final result.

## Risks And Constraints

- Kaggle GPU availability and quotas may vary.
- Kaggle sessions have runtime and persistence limits.
- T4x2 memory may constrain model size, sequence length, batch size, and number of generations per prompt.
- GRPO can be sensitive to reward design; poorly shaped rewards may optimize format without improving reasoning.
- Evaluation results may be noisy on small datasets.
- Some acceleration libraries may have compatibility constraints in Kaggle.

## Open Questions

- Should optional adapter export support Hugging Face, Kaggle Models, or both as a later disabled-by-default feature?
- What exact accuracy gain should count as a strong v1 result after observing the first real run?

## Milestones

### Milestone 1: Planning

- Approve PRD.
- Approve implementation specs.
- Decide default model and dataset.
- Decide Kaggle execution shape.

### Milestone 2: Minimal Pipeline

- Implement dataset preparation.
- Implement reward functions.
- Implement GRPO training entrypoint.
- Add W&B tracking.
- Run a small smoke test.

### Milestone 3: Kaggle Training

- Run the default training configuration on Kaggle T4x2.
- Save adapter and logs.
- Capture W&B run outputs.

### Milestone 4: Evaluation

- Evaluate base model.
- Evaluate trained adapter.
- Save metrics and representative samples.
- Document limitations.

### Milestone 5: Public Release

- Clean README.
- Freeze default config.
- Add reproducibility notes.
- Ensure no private credentials or local paths are committed.

## Change Log

- 2026-09-03: Initial draft.
