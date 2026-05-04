#!/usr/bin/env bash
# v9.3 morning workflow — runs steps 2-4 of the docx prep sequence in order.
#
# Prerequisites:
#   - Kaggle zip (runs_0p5b.zip) extracted to runs/0p5b/ (15 JSONs expected)
#   - .venv set up (uv venv with requirements.lock.txt installed); see RUNBOOK_COLAB.md
#
# Usage from D:\humanai-convention\prism\experiments\sgt\ on Windows bash:
#   ./morning_workflow.sh             # uses runs/0p5b/
#   ./morning_workflow.sh runs/0p5b runs/1p5b   # multiple --runs dirs (e.g. include 1.5B)
#
# Outputs:
#   analysis/v1.json           — sgt_analysis.py output, source of truth for §4
#   analysis/figure2_bands.png — v9.3 §4 Figure 2 (and .pdf)
#   analysis/v9_3_section_4_5.md — §4.5 outcome table + narrative + headline
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/Scripts/python.exe
[ -x "$PY" ] || PY=.venv/bin/python  # linux fallback

if [ "$#" -eq 0 ]; then
  RUN_DIRS=(runs/0p5b)
else
  RUN_DIRS=("$@")
fi

echo "==[1/3]== sgt_analysis.py"
ARGS=()
for d in "${RUN_DIRS[@]}"; do
  if [ ! -d "$d" ]; then
    echo "!! run dir not found: $d"
    exit 1
  fi
  N=$(ls "$d"/*.json 2>/dev/null | wc -l)
  echo "  $d: $N run JSONs"
  ARGS+=(--runs "$d")
done
"$PY" sgt_analysis.py "${ARGS[@]}" --out analysis/v1.json
echo

echo "==[2/3]== plot_bands.py"
"$PY" plot_bands.py "${ARGS[@]}" --out analysis/figure2_bands.png
echo

echo "==[3/3]== format_outcome_table.py"
"$PY" format_outcome_table.py --in analysis/v1.json --out analysis/v9_3_section_4_5.md
echo

echo "============================================================"
echo "morning workflow complete. outputs:"
echo "  analysis/v1.json"
echo "  analysis/figure2_bands.png + .pdf"
echo "  analysis/v9_3_section_4_5.md"
echo
echo "next: open Semantic_Grounding_Recursive_Systems_v9_2.docx,"
echo "      save-as v9_3.docx, apply the diff from"
echo "      C:/Users/benja/Documents/sgt_v9_3_outline.md,"
echo "      pasting numbers from analysis/v9_3_section_4_5.md."
echo "============================================================"
