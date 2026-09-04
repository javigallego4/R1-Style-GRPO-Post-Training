# Experiment Tracking

## GitHub Links

- Issue: TBD

## Status

Implemented for training and Kaggle smoke runs; evaluation-specific logging remains a later slice.

## Objective

Define how the project uses Weights & Biases to track GRPO runs, diagnose training quality, and compare experiments.

## Current Behavior

The repository contains optional W&B tracking controlled by YAML configuration, Kaggle Secret bootstrap support, local run manifests, and a lightweight W&B connectivity probe.

## Desired Behavior

The project should initialize W&B from configuration, log run metadata and training metrics, record reward components separately, and attach evaluation results and selected samples.

## Acceptance Criteria

- [x] W&B can be enabled or disabled from configuration.
- [x] The W&B project name is configurable and defaults to `r1-grpo-kaggle`.
- [x] Private credentials are never hardcoded.
- [x] Run config includes model, dataset, split, LoRA settings, GRPO settings, and seed.
- [x] Training logs include total reward, reward standard deviation, KL, completion length, and trainer metrics.
- [x] Reward components are logged separately.
- [ ] Evaluation metrics are logged after evaluation.
- [ ] Selected generated examples can be logged as a table or artifact.
- [x] Failed or partial runs still leave useful local logs when possible.

## Technical Notes

- W&B should help distinguish real reasoning improvement from format-only reward improvement.
- Run names should be stable and descriptive enough for comparison.
- The tracking layer should be thin and optional so local smoke tests do not require W&B login.
- Local result files should remain the source of truth even when W&B is enabled.
- Adapter/checkpoint artifact logging must be controlled by configuration and disabled by default.

## Likely Affected Areas

- `src/tracking.py`
- `src/train_grpo.py`
- `src/evaluate.py`
- `configs/default.yaml`
- `README.md`

## Validation Plan

- Unit test config parsing for tracking settings.
- Smoke test disabled W&B mode.
- Smoke test W&B initialization with environment-based authentication.
- Confirm no credentials are printed or saved.
- Confirm reward components appear as separate metrics.
- Confirm the Kaggle UI can create a W&B probe run from the `WANDB_API_KEY` secret before launching training.

## Confirmed Decisions

- Weights & Biases is required from v1.
- Reward components must be tracked separately.
- W&B credentials must be provided through environment variables or Kaggle secrets.
- Default W&B project name: `r1-grpo-kaggle`.
- Adapter artifacts should not be published or uploaded by default.

## Accepted Assumptions

- W&B is used for experiment observability, not as the only storage location for results.
- Local result files should still be generated for reproducibility.

## Open Questions

- Should generated samples be logged every N steps, only after evaluation, or both?
- Should adapters/checkpoints be logged as W&B artifacts in a later optional mode?

## Change Log

- 2026-09-03: Initial draft.
- 2026-09-05: Added W&B configuration, Kaggle Secret bootstrap, run manifest tracking, and W&B probe workflow.
