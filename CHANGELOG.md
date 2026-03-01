# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- Added `RELEASE_TODO.md` to track 14-item public release checklist completion.

### Changed
- Finalized rank-ablation summary in README after rank-64 evaluation completed.

### Research Snapshot Updates
- Rank ablation (`8/16/32/64`) final means on labeled easy/hard strata:
  - Rank 8: easy `-0.5790`, hard `-0.7552`, selectivity `-0.1762`, gate discrimination `+0.6927`.
  - Rank 16: easy `-0.6368`, hard `-0.7790`, selectivity `-0.1422`, gate discrimination `+0.6939`.
  - Rank 32: easy `-0.6161`, hard `-0.7794`, selectivity `-0.1633`, gate discrimination `+0.6982`.
  - Rank 64: easy `-0.6288`, hard `-0.8187`, selectivity `-0.1899`, gate discrimination `+0.7020`.
- Sweep conclusion: rank 64 showed the strongest hard-stratum gain and highest gate discrimination in this run.

## [0.1.0] - 2026-02-28

### Added
- Public repository maintenance guide (`AGENTS.md`).
- Open license file (`CC BY 4.0`).
- Packaging metadata via `pyproject.toml`.
- Contributor guide (`CONTRIBUTING.md`).
- Minimal test coverage (`tests/test_analysis.py`).
- Basic GitHub Actions CI workflow.
- Package version export via `spectral_microscope.__version__`.

### Changed
- README synchronized with current public research snapshot.
- Public docs wording normalized for open-source release.
- Optional notebook dependency cleanup (removed unused `seaborn`).
- `SpectralMicroscope` now exposes `streaming_cov_alpha` for EMA tuning.
- Quickstart notebook now installs/uses the package API (`pip install -e .`).
- Core package dependencies were minimized; notebook stack moved to optional `notebook` extras.
