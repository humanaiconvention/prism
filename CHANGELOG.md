# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.2.0] - 2026-05-11

### Added
- **`prism.provenance` submodule** — wraps Cisco's Model Provenance Kit
  (Apache-2.0, https://github.com/cisco-ai-defense/model-provenance-kit,
  released 2026-05-04) to surface model lineage detection alongside
  PRISM's geometry/NLA scans.
  - `compare_models(candidate, parent)` — pairwise lineage comparison.
  - `scan_model_provenance(model, top_k=...)` — database lookup against
    the ~150-model MPK reference catalogue.
  - `ProvenanceResult` / `ProvenanceMatch` / `ProvenanceSignals` typed
    dataclasses.  Every result carries `not_cryptographic=True`, which
    propagates verbatim into the audit-dict serialisation so consumers
    cannot silently drop Cisco's "strong evidence, not absolute proof"
    caveat.
  - `MPKBackend` adapter with lazy import of `provenancekit` and a
    pluggable `scanner=` argument for tests.
  - `mock_compare` / `mock_scan` deterministic offline backend — no MPK
    install or 908 MB dataset download required.
- `docs/PROVENANCE.md` — design doc covering the five MPK signals
  (EAS/END/NLF/LEP/WVC), the 0.70 threshold calibration, when to use
  provenance vs geometry vs NLA, failure modes, and the HAIC connection
  (PRISM model-side primitive + HAIC data-side Merkle receipts).
- 16 new tests in `tests/test_provenance.py`.  No network, no MPK install.

### Changed
- README "Model Provenance" section + toolkit table row.
- `pyproject.toml` version bumped to `1.2.0`.

### Notes
- `provenancekit` is an *optional* runtime dependency.  PRISM does not
  ship it; users who want the real backend run `pip install provenancekit`.
- MPK is v1.0.0 (one week old at this release).  Treat its ground-truth
  claims with appropriate deference until it has had adversarial
  scrutiny.

## [1.1.0] - 2026-05-11

### Added
- **`prism.nla` submodule** — integration with Anthropic's Natural Language
  Autoencoders (May 2026, https://transformer-circuits.pub/2026/nla/index.html):
  - `NLAExplanation` / `NLABatchResult` typed dataclasses.
  - `NLACheckpoint` registry with four released NLAs from
    [kitft/nla-models](https://github.com/kitft/natural_language_autoencoders)
    (Qwen2.5-7B layer 20, Gemma-3-12B layer 32, Gemma-3-27B layer 41,
    Llama-3.3-70B layer 53). Explicit non-entry: no NLA exists for
    `google/gemma-4-e2b-it`.
  - `NLAExplainer` HTTP client over kitft's SGLang serving contract,
    with a pluggable transport so tests need no network.
  - `MockNLAExplainer` / `mock_explainer()` for deterministic offline tests.
  - `summarize_layer()` aggregation into a single per-layer result.
- **`scan_model_geometry(..., nla_explainer=)` parameter** — when supplied,
  the scanner verbalises one layer's activations and attaches an `nla`
  block (`layer_idx`, `n_samples`, `explanations`, `summary`, `mean_fve`,
  `fve_std`). Refuses cross-architecture scans via `d_model` validation.
- `docs/NLA.md` — full design doc covering the technique, the inference
  contract, what is and isn't released, when to use NLA vs geometry alone,
  cost transparency, and Anthropic's four disclosed limitations verbatim.
- README "Natural Language Autoencoders" section in the Methodology block.
- 28 new tests across `tests/test_nla_inference.py` and
  `tests/test_nla_geometry.py`. No GPU and no network required.

### Changed
- README API reference now lists `prism.nla` alongside the other 14 modules.
- `pyproject.toml` version bumped to `1.1.0`.

### Notes
- PRISM remains under CC BY 4.0. The kitft inference wire format is
  Apache-2.0; only that format is referenced here, not any of its code.
- The `nla` block deliberately does not echo raw activation vectors —
  NLA's purpose is to surface human-readable text.

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

