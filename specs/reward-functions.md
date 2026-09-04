# Reward Functions

## GitHub Links

- Issue: TBD

## Status

Implemented for v1 rule-based GRPO rewards.

## Objective

Define the reward functions used by GRPO and the validation needed to ensure they measure useful behavior rather than only superficial formatting.

## Current Behavior

The repository contains deterministic reward functions for correctness, integer answer parsing, strict/soft XML-like format compliance, reasoning tag coverage, and component summaries for diagnostics.

## Desired Behavior

The project should provide modular, deterministic reward functions for mathematical correctness, final-answer format, and reasoning structure. Each reward component should be logged separately so training behavior can be diagnosed.

## Acceptance Criteria

- [x] Correctness reward compares the model's final parsed answer against the reference answer.
- [x] Integer answer reward detects whether the model produces a parseable integer answer.
- [x] Format rewards validate the requested output structure.
- [x] Reasoning-structure rewards validate that the model separates reasoning from the final answer.
- [x] Reward functions handle malformed, empty, and overlong completions safely.
- [x] Reward outputs are deterministic for identical inputs.
- [x] Each reward has unit tests covering positive and negative examples.
- [x] Reward components are available for W&B logging.
- [x] The default expected output format is validated consistently.

## Technical Notes

- Correctness should be the most important success signal.
- Format rewards should support training stability, but must not dominate the interpretation of model quality.
- The reward layer should expose both individual rewards and a combined reward path compatible with TRL.
- Parsing should be tolerant enough to handle normal generations but strict enough to avoid false positives.
- The default output template is:

```text
<reasoning>
...
</reasoning>
<answer>
42
</answer>
```

## Likely Affected Areas

- `src/rewards.py`
- `src/evaluate.py`
- `src/tracking.py`
- `tests/test_rewards.py`

## Validation Plan

- Unit test answer extraction and correctness comparisons.
- Unit test format matching against valid and invalid completions.
- Unit test reward behavior on empty strings, non-numeric answers, and multiple candidate answers.
- Run reward diagnostics on generated samples before training.

## Confirmed Decisions

- Reward components must be logged separately.
- Training success requires correctness or evaluation accuracy improvement, not only format reward improvement.
- The default output format uses separate `<reasoning>` and `<answer>` tags.

## Accepted Assumptions

- Rule-based rewards are sufficient for v1.
- No learned reward model is needed for the initial project.

## Open Questions

- Should correctness remain binary after the first non-smoke evaluation, or should numeric equivalence/partial credit be introduced?
- Should invalid format suppress correctness reward in a later ablation, or remain tracked independently?

## Change Log

- 2026-09-03: Initial draft.
- 2026-09-05: Marked v1 reward functions, parsing helpers, component summaries, and tests as implemented.
