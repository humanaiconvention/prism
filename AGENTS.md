# Agent Instructions for Public Repo Maintenance

## Mission
Maintain `spectral_microscope_public` as a clean, public, open-source repository for Spectral Microscope.

## Scope and Boundaries
1. Only edit files inside this repository root unless the user explicitly approves broader scope.
2. Treat external research docs as reference-only sources.
3. Do not modify primary research files outside this public folder.

## Public-Safe Documentation Rules
1. Do not publish primary infrastructure names, machine nicknames, hostnames, or SSH commands.
2. Do not publish primary absolute local paths from development machines.
3. Do not include internal run identifiers unless explicitly approved.
4. Never include secrets, tokens, credentials, or API keys.

## Repository Content Policy
1. `README.md` is the public contract. Keep it accurate and neutral.
2. Keep examples aligned with files that actually exist in this repo.
3. If a script uses synthetic/demo data, label it clearly as a template and not as validated reproduction.
4. Keep dependency declarations minimal and justified in `requirements.txt`.

## Update Protocol (when syncing from primary research)
1. Extract only public-safe findings (metrics and conclusions).
2. Rewrite infrastructure-specific wording into neutral terms (for example: "local run", "scaled cloud run").
3. Keep claims evidence-bound and version/date stamped where relevant.
4. Verify no primary environment references remain.

## Quality Checklist Before Finalizing Changes
- [ ] All edits are inside this repository root.
- [ ] No primary infra names, primary hostnames, or primary absolute paths appear.
- [ ] Public docs are consistent with current public files and API.
- [ ] Notebook and README usage snippets are still valid.
- [ ] No debug-only artifacts are left in committed content.

## Style and Communication
1. Use concise, technical language.
2. Prefer explicit status labels: `complete`, `running`, `planned`.
3. Avoid overstated claims; report measured outcomes with context.

## Current Public Priorities
1. Keep `README.md` aligned with current validated findings.
2. Keep onboarding smooth via `quickstart.ipynb`.
3. Preserve API stability in `src/spectral_microscope/` unless explicitly requested.
4. When evaluating a new model, follow `FUTURE_MODEL_INTAKE.md` so the run,
   analysis, and comparison steps stay consistent.
5. Use `MODEL_RUN_TEMPLATE.md` as the starting structure for per-model notes,
   and evolve it only when the same gap appears repeatedly.

