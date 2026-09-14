# Evaluation And Results

## GitHub Links

- Issue: TBD

## Status

Implemented for v1 base-vs-adapter comparison, W&B evaluation logging, and Kaggle `train_then_evaluate` execution; first real Kaggle comparison output is running/next to verify.

## Objective

Define how the project evaluates the base model and trained adapter, saves results, and determines whether training produced meaningful reasoning improvement.

## Current Behavior

The repository contains evaluation scripts that load the configured model, optionally apply a LoRA adapter, evaluate a reserved GSM8K split, compute exact numeric correctness, summarize reward components, write JSON/CSV/Markdown outputs, compare base-vs-adapter metrics, and log evaluation summaries plus examples to W&B when tracking is enabled. The Kaggle entrypoint supports `train`, `evaluate`, and `train_then_evaluate` modes; the Kaggle smoke config runs training and evaluation in the same session so the adapter does not need to be published or committed.

## Desired Behavior

The project should run the same evaluation protocol on the base model and on the trained adapter, then save quantitative metrics and representative qualitative examples.

## Acceptance Criteria

- [x] Evaluation can run on a reserved GSM8K split.
- [x] Base model and adapter evaluation use the same prompts and parsing rules.
- [x] Accuracy or exact-match correctness is reported.
- [x] Evaluation records parse failure rate.
- [x] Evaluation records format compliance where relevant.
- [x] Representative examples are saved for human inspection.
- [x] Metrics are saved in a machine-readable file.
- [x] Evaluation results are logged to W&B when enabled.
- [x] Quick evaluation supports 100-200 held-out examples.
- [x] Final evaluation supports 500+ held-out examples when runtime allows.

## Technical Notes

- The primary success metric should be correctness on held-out GSM8K examples.
- Qualitative examples should include question, model response, parsed answer, reference answer, and correctness.
- Evaluation should support small subsets for quick checks and larger runs for final reporting.
- The same answer extraction logic should be shared with reward functions where possible.

## Likely Affected Areas

- `src/evaluate.py`
- `src/data.py`
- `src/rewards.py`
- `src/tracking.py`
- `results/`
- `configs/default.yaml`

## Validation Plan

- Unit test metric computation.
- Unit test response parsing consistency.
- Run evaluation on a small fixed subset.
- Compare base model and adapter outputs in the same result schema.
- Log evaluation metrics and selected examples to W&B when tracking is enabled.
- Run Kaggle smoke in `train_then_evaluate` mode and download the generated comparison outputs.

## Confirmed Decisions

- Evaluation must compare pre-training and post-training behavior.
- Correctness improvement matters more than format-only improvement.
- Results should include both metrics and examples.
- A successful v1 means the full pipeline runs and produces interpretable base-vs-adapter results, even if the first training run exposes weak learning.

## Accepted Assumptions

- Exact numeric answer matching is acceptable for v1.
- More advanced mathematical equivalence checking can be deferred.

## Open Questions

- What held-out subset size should be used for the first public result?
- What accuracy gain should count as a strong v1 result after observing the first real run?
- Should qualitative samples include failed examples deliberately?

## Change Log

- 2026-09-03: Initial draft.
- 2026-09-05: Marked basic JSON evaluation as partially implemented and identified base-vs-adapter/W&B reporting gaps.
- 2026-09-14: Added base-vs-adapter comparison outputs, parse-failure metrics, CSV/Markdown reports, and W&B evaluation logging.
- 2026-09-14: Added Kaggle entrypoint support for `train_then_evaluate` smoke runs.
