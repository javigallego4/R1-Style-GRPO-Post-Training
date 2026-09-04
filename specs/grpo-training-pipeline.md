# GRPO Training Pipeline

## GitHub Links

- Issue: TBD

## Status

Implemented for smoke training; full default-run calibration remains pending.

## Objective

Define the script-first GRPO training pipeline for post-training a small instruction model with LoRA/QLoRA on Kaggle T4x2.

## Current Behavior

The repository contains script-first GRPO training code with configurable Unsloth model loading, LoRA adapter setup, TRL GRPO trainer construction, smoke configuration, W&B integration, and local adapter saving.

## Desired Behavior

The project should expose a reproducible training script that loads the default Unsloth 4-bit model, prepares LoRA/QLoRA adapters, trains with TRL GRPO, logs progress, and saves the resulting adapter.

## Acceptance Criteria

- [x] Training is launched from a script, not from notebook-only logic.
- [x] The default model is `unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit`.
- [x] Model name, LoRA settings, sequence length, batch size, learning rate, and GRPO parameters are configurable.
- [x] Training uses TRL GRPO.
- [x] Fine-tuning updates adapter weights rather than full model weights.
- [x] The training script supports a smoke-test mode.
- [x] A smoke configuration runs a tiny GRPO job before the default run.
- [x] Adapter artifacts are saved to a predictable output directory.
- [x] Training can resume or at least avoid losing final adapter artifacts when a run completes.
- [x] Adapter publication is controlled by configuration and disabled by default.

## Technical Notes

- The default path should use Unsloth where compatible with Kaggle.
- The implementation should keep model loading separate from trainer construction.
- Configuration should be centralized in `configs/default.yaml`.
- Conservative defaults should be used for Kaggle reliability.
- The training loop should expose enough metrics for W&B and local result files.
- Export behavior should be controlled from `configs/default.yaml`, not hardcoded constants.

## Likely Affected Areas

- `src/train_grpo.py`
- `src/config.py`
- `src/data.py`
- `src/rewards.py`
- `src/tracking.py`
- `configs/default.yaml`
- `scripts/train_kaggle.py`

## Validation Plan

- Run a local import/config smoke test.
- Run a tiny dataset smoke test with minimal steps where hardware allows.
- Verify adapter output directory is created.
- Verify W&B can be disabled for local dry runs.
- Verify training logs include reward, KL, completion length, and loss-like metrics exposed by TRL.

## Confirmed Decisions

- v1 uses GRPO-only, with no supervised fine-tuning warm-up.
- The implementation is script-first.
- Kaggle T4x2 is the target GPU environment.
- Adapter artifacts should be saved locally but not published by default.

## Accepted Assumptions

- Full local GPU training is not required.
- Local CPU execution only needs to support lightweight smoke checks where feasible.

## Open Questions

- What should the default number of training steps be for the first non-smoke Kaggle run after observing W&B training curves?
- What should the default number of generations per prompt be on T4x2 after observing memory/runtime?
- Which optional adapter publication destinations should be supported later?

## Change Log

- 2026-09-03: Initial draft.
- 2026-09-05: Marked script-first GRPO smoke training, LoRA adapter export, and W&B training integration as implemented.
