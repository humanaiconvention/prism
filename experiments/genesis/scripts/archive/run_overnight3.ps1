# Genesis Phase 2 — Gap-Filling Diagnostics (overnight batch)
# Estimated total: ~8-10 hours
#
# Run 1: W_o Probe (20 prompts × 32 tokens, ~2h)
# Run 2: Matched-N TTT control (60 prompts × 64 tokens with TTT-off, ~4h)  
# Run 3: Prompt-diversity test (matched N, different prompt/token ratios, ~3h)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONIOENCODING = "utf-8"

Set-Location D:\Genesis

$startTime = Get-Date
Write-Host "========================================"
Write-Host "GENESIS PHASE 2 — GAP-FILLING RUN"
Write-Host "Started: $startTime"
Write-Host "Expected completion: ~8-10 hours"
Write-Host "========================================"

# ============================================================
# RUN 1: Pre-W_o vs Post-W_o Probe (20 prompts × 32 tokens)
# Answers: Is W_o the compressor, or are heads already low-rank?
# Expected: ~2 hours
# ============================================================
Write-Host ""
Write-Host "[RUN 1/3] W_o Probe — 60 prompts × 64 tokens"
Write-Host "  Output: logs/wo_probe/"
Write-Host "  Log: logs/phase2_run1_wo_probe.log"
$t1 = Get-Date

python scripts/run_wo_probe.py --max-prompts 60 --max-tokens 64 --prompts prompts/prompts_200.json --output-dir logs/wo_probe 2>&1 | Tee-Object -FilePath logs/phase2_run1_wo_probe.log

$elapsed1 = (Get-Date) - $t1
Write-Host ""
Write-Host "[RUN 1/3] DONE in $($elapsed1.TotalMinutes.ToString('F1')) min"

# ============================================================
# RUN 2: Matched-N TTT Control (60 prompts × 64 tokens, TTT-off)
# N=3840, matching the normal run exactly
# Answers: True TTT contribution without sample-size confound
# Expected: ~4 hours
# ============================================================
Write-Host ""
Write-Host "[RUN 2/3] Matched-N TTT Control — 60 prompts × 64 tokens (TTT-off)"
Write-Host "  Output: logs/corrected_ttt_off_matched/"
Write-Host "  Log: logs/phase2_run2_ttt_matched.log"
$t2 = Get-Date

python scripts/run_corrected_er.py --disable-ttt --max-prompts 60 --max-tokens 64 --output-dir logs/corrected_ttt_off_matched 2>&1 | Tee-Object -FilePath logs/phase2_run2_ttt_matched.log

$elapsed2 = (Get-Date) - $t2
Write-Host ""
Write-Host "[RUN 2/3] DONE in $($elapsed2.TotalMinutes.ToString('F1')) min"

# ============================================================
# RUN 3: Prompt Diversity Test — matched N, different ratios
# 3a: 60 prompts × 64 tokens = 3840 samples (many prompts, fewer tokens)
# 3b: 30 prompts × 128 tokens = 3840 samples (few prompts, many tokens)
# Answers: Is prompt diversity or token diversity more important for ER?
# Expected: ~3 hours total
# ============================================================
Write-Host ""
Write-Host "[RUN 3/3] Prompt Diversity Test — matched N=3840"
Write-Host "  Output: logs/diversity_many_prompts/ and logs/diversity_many_tokens/"
Write-Host "  Log: logs/phase2_run3_diversity.log"
$t3 = Get-Date

Write-Host "  [3a] 60 prompts × 64 tokens (many prompts)"
python scripts/run_corrected_er.py --max-prompts 60 --max-tokens 64 --prompts prompts/prompts_200.json --output-dir logs/diversity_many_prompts 2>&1 | Tee-Object -FilePath logs/phase2_run3a_diversity.log

Write-Host "  [3b] 30 prompts × 128 tokens (many tokens)"
python scripts/run_corrected_er.py --max-prompts 30 --max-tokens 128 --prompts prompts/prompts_200.json --output-dir logs/diversity_many_tokens 2>&1 | Tee-Object -FilePath logs/phase2_run3b_diversity.log

$elapsed3 = (Get-Date) - $t3
Write-Host ""
Write-Host "[RUN 3/3] DONE in $($elapsed3.TotalMinutes.ToString('F1')) min"

# ============================================================
# SUMMARY
# ============================================================
$totalElapsed = (Get-Date) - $startTime
$endTime = Get-Date

$summary = "========================================`n"
$summary += "PHASE 2 GAP-FILLING COMPLETE`n"
$summary += "========================================`n"
$summary += "Started:  $startTime`n"
$summary += "Finished: $endTime`n"
$summary += "Total:    $($totalElapsed.TotalHours.ToString('F1')) hours`n`n"
$summary += "Run 1 (W_o Probe 60x64):          $($elapsed1.TotalMinutes.ToString('F1')) min - logs/wo_probe/`n"
$summary += "Run 2 (TTT-matched 60x64):        $($elapsed2.TotalMinutes.ToString('F1')) min - logs/corrected_ttt_off_matched/`n"
$summary += "Run 3 (Diversity 60x64 vs 30x128): $($elapsed3.TotalMinutes.ToString('F1')) min - logs/diversity_*/`n"
$summary += "========================================`n"

Write-Host $summary
$summary | Out-File -FilePath logs/phase2_summary.log -Encoding utf8
Write-Host "All results saved. Check logs/phase2_summary.log for recap."
