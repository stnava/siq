---
name: release
description: >-
  Standardized procedure for version bumping, staging, committing, tagging,
  and pushing releases in the siq repository. Use whenever the user asks to
  run the release skill, commit, bump version, tag, release, or push changes.
---

# siq Release Workflow: Commit, Tag, Version Bump, Push

This runbook guides the complete release cycle to ensure version synchronization, clean git history, and proper tag propagation in `siq`.

## Phase 1: Determine Version Bump Level

1. Inspect current version strings:
   - `pyproject.toml`: `version = "X.Y.Z"`
   - `siq/version.py`: `__version__ = "X.Y.Z"`
   - Latest git tag: `git tag -l --sort=-v:refname | head -n 5`
2. Determine increment:
   - **Patch (`X.Y.Z+1`)**: Bug fixes, metric improvements, weight transfer helpers, training scripts, documentation without breaking changes.
   - **Minor (`X.Y+1.0`)**: New architectures, new perceptual backends, major pipeline additions.
   - **Major (`X+1.0.0`)**: Breaking API or coordinate convention changes.

## Phase 2: Synchronize Version Across Repository

Update the version in both files simultaneously:
1. `pyproject.toml`:
   ```toml
   version = "X.Y.Z"
   ```
2. `siq/version.py`:
   ```python
   __version__ = "X.Y.Z"
   ```
3. Verify synchronization:
   ```bash
   python -c "import tomllib; f=open('pyproject.toml','rb'); print('pyproject:', tomllib.load(f)['project']['version']); import siq; print('init:', siq.__version__)"
   ```

## Phase 3: Staging & Committing

1. Inspect modified and untracked files:
   ```bash
   git status
   ```
2. Stage intentional files (code, tests, documentation, scripts, configs). Do **NOT** stage heavy temporary scratch caches or non-champion model weights unless intended:
   ```bash
   git add pyproject.toml siq/version.py siq/... tests/... scripts/...
   ```
3. Commit using the project's standard release format:
   ```bash
   git commit -m "release: vX.Y.Z - <Summary of primary changes>"
   ```

## Phase 4: Create Annotated Git Tag

1. Tag the release commit matching the exact version string (`vX.Y.Z`):
   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z: <Summary>"
   ```
2. Verify tag points to the HEAD commit:
   ```bash
   git describe --tags --exact-match
   ```

## Phase 5: Push Branch and Tags to Remote

1. Push current branch and tags:
   ```bash
   git push origin main && git push origin vX.Y.Z
   ```
2. Verify remote status:
   ```bash
   git status
   ```

## Invariants & Safety Checks
- **Dual-file consistency**: Never bump `pyproject.toml` without bumping `siq/version.py`.
- **Tag prefix**: Git tags must always use lowercase `v` prefix (`vX.Y.Z`).
- **Never force push tags**: Do not use `git push --force --tags` unless explicitly instructed.
