# Reward Functions

## GitHub Links

- Issue: TBD

## Status

Draft

## Objective

Define the reward functions used by GRPO and the validation needed to ensure they measure useful behavior rather than only superficial formatting.

## Current Behavior

The repository does not yet contain reward functions.

## Desired Behavior

The project should provide modular, deterministic reward functions for mathematical correctness, final-answer format, and reasoning structure. Each reward component should be logged separately so training behavior can be diagnosed.

## Acceptance Criteria

- [ ] Correctness reward compares the model's final parsed answer against the reference answer.
- [ ] Integer answer reward detects whether the model produces a parseable integer answer.
- [ ] Format rewards validate the requested output structure.
- [ ] Reasoning-structure rewards validate that the model separates reasoning from the final answer.
- [ ] Reward functions handle malformed, empty, and overlong completions safely.
- [ ] Reward outputs are deterministic for identical inputs.
- [ ] Each reward has unit tests covering positive and negative examples.
- [ ] Reward components are available for W&B logging.
- [ ] The default expected output format is validated consistently.

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

- Should correctness reward be binary, partially graded, or weighted with format rewards?
- Should invalid format suppress correctness reward or be tracked independently?

## Change Log

- 2026-09-03: Initial draft.
