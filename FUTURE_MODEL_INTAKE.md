# Future Model Intake

Status: reusable checklist  
Last updated: 2026-03-22  
Scope: using PRISM to evaluate future models before wider promotion  
Not authoritative for: benchmark rules or submission state

## Goal

Keep future model testing consistent, bounded, and easy to compare.

## Standard Flow

1. Note the model name, source, and why it matters.
2. Run a bounded PRISM scan or shared test batch.
3. Capture the raw run output and the Prism model-run bundle path.
4. Fill in `MODEL_RUN_TEMPLATE.md` with the run metadata and analysis notes.
5. Compare the result against the previous model or baseline.
6. Decide whether to stay local, iterate, or move to a bigger GPU / cloud lane.

## What To Record

- model name
- source or checkpoint location
- prompt / task family
- run command
- hardware used
- raw output path
- Prism log bundle path
- Prism observations
- Prism lessons to carry forward
- semantic grounding observations
- keep / revise / escalate decision
- template file used

## Good Defaults

- keep logs bounded
- keep one analysis note per model or batch
- use the same section order every time
- compare against the immediate prior run, not just a vague memory
- evolve the template if the same missing field shows up more than once

## Where To Use This

- the PRISM repo itself
- HAIC Prism analysis lanes
- any future model that needs a repeatable interpretability pass
