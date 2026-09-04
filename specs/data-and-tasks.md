# Data And Tasks

## GitHub Links

- Issue: TBD

## Status

Draft

## Objective

Define how the project loads, normalizes, splits, and exposes the reasoning task used for GRPO post-training and evaluation.

## Current Behavior

The repository does not yet contain data loading or task preparation code.

## Desired Behavior

The project should load `openai/gsm8k`, use the training split for GRPO post-training, and reserve the validation/test split for later evaluation. The internal data format should make prompts, reference answers, and extracted numeric targets explicit.

## Acceptance Criteria

- [ ] The dataset can be loaded from configuration.
- [ ] The default dataset is `openai/gsm8k`.
- [ ] Training data and evaluation data are kept separate.
- [ ] Each sample exposes the original question, reference answer, and parsed final numeric answer.
- [ ] A small subset mode exists for smoke tests.
- [ ] Dataset preparation is deterministic when a seed is provided.
- [ ] Malformed or unparsable samples are counted and reported.
- [ ] Default configuration supports 1,000-2,000 training examples.
- [ ] Default configuration supports 100-200 quick evaluation examples.
- [ ] Final evaluation supports 500+ held-out examples when runtime allows.

## Technical Notes

- The expected default task is grade-school mathematical reasoning.
- GSM8K answers commonly include a final answer marker; parsing should extract the final numeric result robustly.
- The project should avoid hardcoding dataset paths so Kaggle and local execution can share the same code.
- Prompt construction should be centralized so training and evaluation use consistent input formatting.
- Prompts should ask for the default reasoning and answer format used by the reward layer.

## Likely Affected Areas

- `src/data.py`
- `src/config.py`
- `configs/default.yaml`
- `tests/test_data.py`

## Validation Plan

- Unit test final-answer extraction on representative GSM8K answer strings.
- Unit test deterministic subset selection.
- Smoke test loading a small number of training and evaluation samples.
- Confirm train/evaluation splits do not overlap.

## Confirmed Decisions

- Default dataset: `openai/gsm8k`.
- Training uses the training split.
- Evaluation uses a reserved split.
- The first version should prioritize a small, reliable pipeline over maximum dataset size.
- v1 starts with a capped training subset of 1,000-2,000 examples.
- v1 supports quick evaluation on 100-200 examples and larger final evaluation on 500+ examples.

## Accepted Assumptions

- The default GSM8K split naming will be handled by the dataset-loading layer.
- A small subset mode is acceptable for local smoke tests.

## Open Questions

- Should the default training subset be 1,000, 1,500, or 2,000 examples?
- Should the first final report use exactly 500 held-out examples or more if runtime allows?

## Change Log

- 2026-09-03: Initial draft.
