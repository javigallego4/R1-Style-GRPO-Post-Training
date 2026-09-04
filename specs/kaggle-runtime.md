# Kaggle Runtime

## GitHub Links

- Issue: TBD

## Status

Draft

## Objective

Define how the repository should run on Kaggle's dual-T4 GPU environment from scripts while keeping local development and public documentation clean.

## Current Behavior

The repository does not yet contain Kaggle-specific execution instructions or scripts.

## Desired Behavior

The project should provide a clear script-based workflow that can be launched in Kaggle, install or verify required dependencies, load configuration, authenticate external services through environment variables, and save outputs in Kaggle-compatible paths.

## Acceptance Criteria

- [ ] Kaggle execution is documented in the README.
- [ ] Runtime setup does not require private local paths.
- [ ] W&B authentication uses environment variables or Kaggle secrets.
- [ ] Output directories are compatible with Kaggle persistence rules.
- [ ] The training script can run with Kaggle T4x2 constraints.
- [ ] The project documents expected memory/runtime constraints.
- [ ] The project includes a minimal command sequence for Kaggle execution.
- [ ] The repository can generate Kaggle kernel metadata without hardcoding a user account.
- [ ] The Kaggle entrypoint is a Python script.
- [ ] A smoke configuration exists for low-cost Kaggle validation before a full run.

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

- Should the repo include Kaggle metadata files for automated notebook/kernel creation?
- Should outputs be exported as Kaggle Dataset/Model artifacts in v1?
- Should dependency installation prefer `pip`, `uv`, or a plain requirements file for Kaggle?

## Change Log

- 2026-09-03: Initial draft.
