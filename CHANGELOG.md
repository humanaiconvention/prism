# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

## [1.2.2] - 2026-05-11

### Fixed
- **Add `pandas>=2.0.0` to install requirements.**  `prism.eval.temporal`
  imports `pandas as pd` at module load, and `prism.eval.__init__`
  chains to it, so `import prism.eval` (and any submodule sweep) fails
  without pandas.  Surfaced during the post-1.2.1 import-sweep audit;
  scipy was the larger blocker and was fixed in 1.2.1, but pandas was
  hiding behind it.
- Removed `pandas` from the `notebook` optional extras group — it is
  now a base dependency, so the duplicate listing would be confusing.
  `notebook` retains matplotlib + jupyter for users who actually run
  the analysis notebooks.

### Notes
- 1.2.0 and 1.2.1 exist on TestPyPI as failed verification runs; neither
  was published to real PyPI.  This release supersedes both.

## [1.2.1] - 2026-05-11

### Fixed
- **Add `scipy>=1.10.0` to install requirements.**  `prism.phase.coherence`
  imports `scipy.signal.hilbert` at module load time, which transitively
  blocks `import prism` itself on a clean install.  The dev environment
  has scipy as a transitive of pandas/sklearn so `pip install -e .`
  worked locally, but the 1.2.0 TestPyPI wheel could not be imported on
  a fresh venv.  Surfaced during TestPyPI verification, not after a real
  PyPI publish.
- Declare `sentence-transformers` under a new `dense-eval` optional extra
  (`pip install 'humanaiconvention-prism[dense-eval]'`).  It is lazily
  imported inside `prism.eval.metrics` so its absence does not block
  `import prism`, but users running dense-embedding grounding metrics
  now have a discoverable install path.

### Notes
- 1.2.0 exists on TestPyPI but was never published to real PyPI.  This
  release supersedes it; do not deploy 1.2.0.

## [1.2.0] - 2026-05-11

> First public release on PyPI.
>
> Earlier numbered entries below (`0.2.0`, `0.1.0`) document internal
> development milestones that were never published to PyPI.  The
> package is starting its public version history at 1.2.0 because the
> internal `__version__` reached 1.0.0 during the 14-module
> consolidation pass and 1.1.0 during the NLA work earlier today;
> rather than fabricate separate PyPI releases for those internal
> milestones, this entry consolidates everything new since 0.2.0 into
> one honest first-publish record.

### Added

#### `prism.nla` — Natural Language Autoencoder integration

Wraps Anthropic's Natural Language Autoencoders ([Anthropic May 2026
paper](https://transformer-circuits.pub/2026/nla/index.html)) so an
NLA can be attached to PRISM's geometry scan to produce human-readable
per-layer activation explanations.

- `NLAExplanation` / `NLABatchResult` typed dataclasses.
- `NLACheckpoint` registry with the four released NLAs from
  [kitft/nla-models](https://github.com/kitft/natural_language_autoencoders)
  (Qwen2.5-7B layer 20, Gemma-3-12B layer 32, Gemma-3-27B layer 41,
  Llama-3.3-70B layer 53).  Explicit non-entry: no NLA exists for
  `google/gemma-4-e2b-it` — the d_model guard refuses cross-architecture
  scans.
- `NLAExplainer` HTTP client over a wrapper-API contract, with a
  pluggable transport so tests need no network.  `docs/NLA.md` §8
  documents the bridge to kitft's raw SGLang serving format.
- `MockNLAExplainer` / `mock_explainer()` deterministic offline backend.
- `summarize_layer()` aggregation into a single per-layer result.
- `scan_model_geometry(..., nla_explainer=)` parameter: when supplied,
  the scanner verbalises one layer's activations and attaches an
  `nla` block (`layer_idx`, `n_samples`, `explanations`, `summary`,
  `mean_fve`, `fve_std`) alongside the existing geometry metrics.
  Refuses cross-architecture scans via `d_model` validation.
- `docs/NLA.md` — full design doc covering the technique, the inference
  contract, what is and isn't released, when to use NLA vs geometry
  alone, cost transparency, and Anthropic's four disclosed limitations
  (confabulation, blackbox by construction, excessive expressivity,
  cost) carried verbatim.
- 28 new tests across `tests/test_nla_inference.py` and
  `tests/test_nla_geometry.py`. No GPU and no network required.

#### `prism.provenance` — Cisco Model Provenance Kit wrapper

Wraps [Cisco's Model Provenance Kit](https://github.com/cisco-ai-defense/model-provenance-kit)
(Apache-2.0, released 2026-05-04) to surface model lineage detection
("is this artifact derived from what its producer claims?") alongside
the geometry / NLA scans.

- `compare_models(candidate, parent)` — pairwise lineage comparison.
- `scan_model_provenance(model, top_k=...)` — database lookup against
  the ~150-model MPK reference catalogue.
- `ProvenanceResult` / `ProvenanceMatch` / `ProvenanceSignals` typed
  dataclasses.  Every result carries `not_cryptographic=True`, which
  propagates verbatim into the audit-dict serialisation so consumers
  cannot silently drop Cisco's "strong evidence, not absolute proof"
  caveat.
- `MPKBackend` adapter with lazy import of `provenancekit` (the PyPI
  distribution is `cisco-ai-provenance-kit`) and a pluggable
  `scanner=` argument for tests.
- `mock_compare` / `mock_scan` deterministic offline backend — no MPK
  install or 908 MB dataset download required.
- Live-validated against `cisco-ai-provenance-kit==1.0.0` on the BEAST
  on 2026-05-11; the smoke test surfaced a schema mismatch in the
  initial duck-typed adapter (the real composite score lives at
  `result.scores.pipeline_score`, not at `result.composite_score`),
  which was fixed and locked in by `TestMPKRealSchema` regression
  tests.  See `docs/PROVENANCE.md` §9 for the full smoke-test log.
- `docs/PROVENANCE.md` — design doc covering the five MPK signals
  (EAS/END/NLF/LEP/WVC), the 0.70 threshold calibration, when to use
  provenance vs geometry vs NLA, failure modes, the live-backend log,
  and the HAIC connection (PRISM model-side primitive + HAIC data-side
  Merkle receipts).
- 19 new tests in `tests/test_provenance.py`. No network, no MPK
  install required.

#### Earlier internal work (rolled into this first release)

- 14-module package (`prism.telemetry`, `prism.causal`,
  `prism.discovery`, `prism.eval`, `prism.geometry`,
  `prism.geometry.hessian`, `prism.attention`, `prism.lens`,
  `prism.probing`, `prism.sae`, `prism.mlp`, `prism.arch`,
  `prism.phase`, `prism.entropy`, `prism.runs`) replacing the
  deprecated `spectral_microscope` entry point.  See the 0.2.0 entry
  below for the full module list.
- `SpectralMicroscope.full_scan()` high-level API (logit lens, rank
  restoration, static circuit SVD, positional sensitivity).
- `has_outlier_geometry` boolean flag on `PrismRunSummary`.
- Genesis-152M mechanistic interpretability replication suite
  (`experiments/genesis/`) covering Phases 0A–12F.
- `spectral_microscope` compatibility shim for backwards-compatible
  imports from pre-0.2.0 callers.

### Changed
- README adds "Natural Language Autoencoders" and "Model Provenance"
  sections in the Methodology block, plus two new rows in the toolkit
  table.
- `pyproject.toml` version → `1.2.0`.

### Optional dependencies (unchanged install surface)
- `prism.nla` lazily imports `requests` only when the real HTTP
  transport is constructed.  Mock backend has no extra dependency.
- `prism.provenance` lazily imports `provenancekit` (PyPI:
  `cisco-ai-provenance-kit`) only when the real backend is used
  without an injected scanner.  Mock backend has no extra dependency.
- Neither submodule changes the base install surface — `pip install
  humanaiconvention-prism` pulls torch + transformers + numpy as
  before; the new external integrations are opt-in extras.

### Notes
- PRISM remains under CC BY 4.0.  The kitft inference wire format
  (Apache-2.0) and Cisco's MPK (Apache-2.0) are referenced via their
  public APIs; no vendored code from either project.
- MPK is v1.0.0 (released 2026-05-04, one week before this release).
  Treat its ground-truth claims with appropriate deference until it
  has had broader adversarial scrutiny.
- The `nla` block in `scan_model_geometry` output deliberately does
  not echo raw activation vectors — NLA's purpose is to surface
  human-readable text.

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

