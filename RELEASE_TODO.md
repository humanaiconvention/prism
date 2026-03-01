# Public Release Checklist (14-item plan)

Status legend: [x] done, [~] in progress, [ ] pending.

## Must-do Before Public Release
- [x] 1. Add a LICENSE file (CC BY 4.0).
- [x] 2. Add a `.gitignore`.
- [x] 3. Add `pyproject.toml`.
- [x] 4. Resolve git state (README updates committed, AGENTS decision documented, `reproduce_paper.py` removal documented).

## High Priority (Polish, Credibility)
- [x] 5. Add `__version__` to `src/spectral_microscope/__init__.py`.
- [x] 6. Add citation block to README.
- [~] 7. Update README Active Work with final rank ablation conclusion after rank-64 eval completes.
- [x] 8. Add example telemetry output to README.
- [x] 9. Add `CONTRIBUTING.md`.

## Optional (Long-term Polish)
- [x] 10. Expose EMA alpha as `streaming_cov_alpha` in `SpectralMicroscope`.
- [x] 11. Add minimal tests (`tests/test_analysis.py`).
- [x] 12. Add CI workflow (`.github/workflows/test.yml`).
- [x] 13. Resolve seaborn dependency mismatch (removed from requirements; notebook extras retained in `pyproject.toml`).
- [x] 14. Add `CHANGELOG.md`.

## Next action when rank-64 finishes
1. Pull final `rank_ablation_rank64_stratified.csv`.
2. Compute mean Easy/Hard Delta NLL + selectivity + gate discrimination across all ranks.
3. Replace interim table in README with final table + one-sentence conclusion.
4. Add final result summary in `CHANGELOG.md` under `[Unreleased]`.
