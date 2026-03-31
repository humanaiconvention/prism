# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [0.2.0] - 2026-03-30

### Added
- **14-module `src/prism/` package** replacing `spectral_microscope` as primary entry point:
  - `prism.telemetry` — PRISM adapters, snapshot logic, settlement proofs (`EntropyDeltaProof`, `GeometricHealthScore`)
  - `prism.causal` — Activation patching, Attribution Patching (AtP), knockout circuits
  - `prism.discovery` — `CircuitScout` automated circuit extraction, graph export
  - `prism.eval` — Grounding metrics, calibration/diversity metrics, early warning detection, geometric drift extension, temporal collapse analysis
  - `prism.geometry` / `prism.geometry.hessian` — TurboQuant outlier-geometry, Hessian landscape diagnostics
  - `prism.attention` — Induction head detection, weight-space SVD, attention entropy maps
  - `prism.lens` — Logit lens, tuned lens, prediction entropy tracking
  - `prism.probing` — Concept probing, causal steering (CIP), CKA drift
  - `prism.sae` — TopK Sparse Autoencoder training, feature attribution
  - `prism.mlp` — Rank restoration profile, key-value neuron mapping
  - `prism.arch` — Recurrent attractors, positional sensitivity, linear/softmax fingerprinting
  - `prism.phase` — Hilbert phase extraction, cross-layer PLV, FFT telemetry, phase clustering
  - `prism.entropy` — Expansion/pruning profiles, Rényi entropy sweeps, spectral-semantic coupling
  - `prism.runs` — Run management and logging utilities
- `SpectralMicroscope.full_scan()` high-level API (logit lens, rank restoration, static circuit SVD, positional sensitivity).
- `has_outlier_geometry` boolean flag on `PrismRunSummary`.
- Genesis-152M mechanistic interpretability replication suite (`experiments/genesis/`) covering Phases 0A–12F.
- Comprehensive test suite (`tests/`) covering all 14 modules.
- Run scripts (`run_*.py`) and CI workflow (`.github/workflows/`).
- `spectral_microscope` compatibility shim for backwards-compatible imports.

### Changed
- `pyproject.toml` version bumped to `0.2.0`.
- README overhauled with full module reference, quickstart, and Genesis-152M research snapshot.
- CHANGELOG updated to reflect all additions since 0.1.0.
- Research snapshot: Phase 10 findings (early-mid corridor L7-L11, OOD family sensitivity), Phase 11 decomposition, rank-64 ablation sweep finalized.
- Rank ablation (`8/16/32/64`) final means on labeled easy/hard strata:
  - Rank 8: easy `-0.5790`, hard `-0.7552`, selectivity `-0.1762`, gate discrimination `+0.6927`.
  - Rank 64: easy `-0.6288`, hard `-0.8187`, selectivity `-0.1899`, gate discrimination `+0.7020`.
  - Rank 64 showed strongest hard-stratum gain and highest gate discrimination.

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

