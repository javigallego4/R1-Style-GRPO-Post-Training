# GRPO Training Pipeline

## GitHub Links

- Issue: TBD

## Status

Draft

## Objective

Define the script-first GRPO training pipeline for post-training a small instruction model with LoRA/QLoRA on Kaggle T4x2.

## Current Behavior

The repository does not yet contain model loading, adapter setup, or GRPO training code.

## Desired Behavior

The project should expose a reproducible training script that loads the default Unsloth 4-bit model, prepares LoRA/QLoRA adapters, trains with TRL GRPO, logs progress, and saves the resulting adapter.

## Acceptance Criteria

- [ ] Training is launched from a script, not from notebook-only logic.
- [ ] The default model is `unsloth/Llama-3.2-1B-Instruct-unsloth-bnb-4bit`.
- [ ] Model name, LoRA settings, sequence length, batch size, learning rate, and GRPO parameters are configurable.
- [ ] Training uses TRL GRPO.
- [ ] Fine-tuning updates adapter weights rather than full model weights.
- [ ] The training script supports a smoke-test mode.
- [ ] A smoke configuration runs a tiny GRPO job before the default run.
- [ ] Adapter artifacts are saved to a predictable output directory.
- [ ] Training can resume or at least avoid losing final adapter artifacts when a run completes.
- [ ] Adapter publication is controlled by configuration and disabled by default.

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

- What should the default number of training steps be for the first Kaggle run?
- What should the default number of generations per prompt be on T4x2?
- Should the first version support checkpoint resume or only final adapter export?
- Which optional adapter publication destinations should be supported later?

## Change Log

- 2026-09-03: Initial draft.
