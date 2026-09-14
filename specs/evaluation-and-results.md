# Evaluation And Results

## GitHub Links

- Issue: TBD

## Status

Implemented for v1 base-vs-adapter comparison, W&B evaluation logging, and Kaggle `train_then_evaluate` execution. Kaggle smoke comparison has been validated on version 16.

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

## Latest Smoke Result

- Kaggle version: 16
- Run mode: `train_then_evaluate`
- Training: 10 GRPO steps on 32 GSM8K training examples
- Evaluation: 4 held-out GSM8K examples, base vs saved adapter
- Base accuracy: 0.25
- Adapter accuracy: 0.25
- Accuracy delta: +0.00
- Base parse failure rate: 0.00
- Adapter parse failure rate: 0.00
- Strict/soft XML format rates: 0.00 for both base and adapter
- W&B: disabled by runtime fallback because CLI-launched Kaggle could not access the configured secret

Interpretation: the pipeline is now validated end to end, but the smoke run is too small to demonstrate learning. The result should be treated as infrastructure validation, not as model-quality evidence.

## Latest Pilot Result

- Kaggle version: 19
- Run mode: `train_then_evaluate`
- Training: 50 GRPO steps on 128 GSM8K training examples
- Evaluation: 8 held-out GSM8K examples, base vs saved adapter
- Base accuracy: 0.125
- Adapter accuracy: 0.125
- Accuracy delta: +0.00
- Base parse failure rate: 0.00
- Adapter parse failure rate: 0.00
- Strict/soft XML format rates: 0.00 for both base and adapter
- W&B: disabled by runtime fallback because CLI-launched Kaggle could not access the configured secret

Interpretation: the first stable pilot run validates a longer train/evaluate loop and adapter export, but it still does not show held-out quality improvement. Training metrics show intermittent reward variation and occasional soft-format reward, but held-out outputs remain format-poor. The next iteration should focus on stronger format induction and/or a slightly longer run from Kaggle UI with W&B enabled.

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
- 2026-09-14: Validated Kaggle version 16 end to end with train, adapter export, base-vs-adapter evaluation, and downloaded comparison outputs.
- 2026-09-14: Validated Kaggle version 19 pilot end to end with 50 GRPO steps, adapter export, checkpoints, and base-vs-adapter evaluation.
