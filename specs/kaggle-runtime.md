# Kaggle Runtime

## GitHub Links

- Issue: TBD

## Status

Implemented for Kaggle script execution and smoke validation; W&B online validation is supported through Kaggle UI secrets.

## Objective

Define how the repository should run on Kaggle's dual-T4 GPU environment from scripts while keeping local development and public documentation clean.

## Current Behavior

The repository contains a Kaggle script entrypoint, kernel metadata generation, T4 accelerator metadata, runtime environment setup, smoke configuration, W&B Secret bootstrap, and documented CLI/UI execution notes.

## Desired Behavior

The project should provide a clear script-based workflow that can be launched in Kaggle, install or verify required dependencies, load configuration, authenticate external services through environment variables, and save outputs in Kaggle-compatible paths.

## Acceptance Criteria

- [x] Kaggle execution is documented in the README.
- [x] Runtime setup does not require private local paths.
- [x] W&B authentication uses environment variables or Kaggle secrets.
- [x] Output directories are compatible with Kaggle persistence rules.
- [x] The training script can run with Kaggle T4x2 constraints.
- [x] The project documents expected memory/runtime constraints.
- [x] The project includes a minimal command sequence for Kaggle execution.
- [x] The repository can generate Kaggle kernel metadata without hardcoding a user account.
- [x] The Kaggle entrypoint is a Python script.
- [x] A smoke configuration exists for low-cost Kaggle validation before a full run.

## Technical Notes

- Kaggle is the default GPU execution target.
- Scripts should detect or accept configured paths for input data, outputs, and adapters.
- The project should avoid assuming paid GPUs or custom containers.
- Dependencies should be pinned or constrained enough to reduce runtime surprises.
- A local helper should generate `kernel-metadata.json` from a username/slug instead of committing user-specific metadata.

## Likely Affected Areas

- `README.md`
- `configs/default.yaml`
- `scripts/train_kaggle.py`
- `scripts/evaluate_kaggle.py`
- `requirements.txt` or `pyproject.toml`
- `kaggle_entry.py`
- `scripts/create_kaggle_metadata.py`
- `scripts/check_environment.py`
- `scripts/inspect_data.py`
- `configs/smoke.yaml`

## Validation Plan

- Validate that setup instructions are executable in a fresh environment.
- Run a short Kaggle smoke test.
- Confirm output artifacts are saved under a predictable directory.
- Confirm no secret values are written to logs or committed files.

## Confirmed Decisions

- Kaggle T4x2 is the intended GPU target.
- The project should be script-first.
- W&B should be integrated from v1.

## Accepted Assumptions

- Some Kaggle-side setup may still require a minimal cell or shell command to launch repository scripts.
- GPU availability and weekly quota are outside the repository's control.

## Open Questions

- Should outputs be exported as Kaggle Dataset/Model artifacts in v1?
- Should dependency installation be pinned more tightly after the first full successful run?

## Change Log

- 2026-09-03: Initial draft.
- 2026-09-05: Marked Kaggle script runtime, T4 metadata, smoke config, and UI-based W&B validation as implemented.
