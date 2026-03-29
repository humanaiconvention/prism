# ============================================================
# PHASE 3: Deep Architectural Probing — Sequential Runner
# ============================================================
# Experiment 1: CKA Norm Diagnostic         (~12 min)
# Experiment 2: Higher-N Convergence Run     (~66 min)
# Experiment 3: Prompt-Type ER Breakdown     (~60 min)
# ============================================================
# Total estimated runtime: ~2.5 hours
# ============================================================

$env:PYTHONIOENCODING="utf-8"

Write-Host "========================================"
Write-Host "STARTING PHASE 3: DEEP ARCHITECTURAL PROBING"
Write-Host "Started: $(Get-Date)"
Write-Host "Expected total time: ~2.5 hours"
Write-Host "========================================"

$t_total = Get-Date

# ============================================================
# RUN 1: CKA Norm Diagnostic
# 60 prompts × 64 tokens = 3840 samples
# Linear CKA(block_input, post_norm) per layer
# Expected: ~12 min
# ============================================================
Write-Host ""
Write-Host "[RUN 1/3] CKA Norm Diagnostic — 60 prompts × 64 tokens"
Write-Host "  Output: logs/cka_norm/"
Write-Host "  Log: logs/phase3_run1_cka.log"
$t1 = Get-Date

python scripts/run_cka_norm.py --max-prompts 60 --max-tokens 64 --prompts prompts/prompts_200.json --output-dir logs/cka_norm 2>&1 | Tee-Object -FilePath logs/phase3_run1_cka.log

$elapsed1 = (Get-Date) - $t1
Write-Host ""
Write-Host "  CKA Norm complete: $($elapsed1.TotalMinutes.ToString('F1')) min"

# ============================================================
# RUN 2: Higher-N Convergence Run
# 200 prompts × 144 tokens = 28,800 samples (N/d ≈ 50x)
# Standard Welford ER measurement at high N
# Expected: ~66 min
# ============================================================
Write-Host ""
Write-Host "[RUN 2/3] Higher-N Convergence — 200 prompts × 144 tokens (N/d≈50)"
Write-Host "  Output: logs/high_n_converged/"
Write-Host "  Log: logs/phase3_run2_high_n.log"
$t2 = Get-Date

python scripts/run_corrected_er.py --max-prompts 200 --max-tokens 144 --prompts prompts/prompts_200.json --output-dir logs/high_n_converged 2>&1 | Tee-Object -FilePath logs/phase3_run2_high_n.log

$elapsed2 = (Get-Date) - $t2
Write-Host ""
Write-Host "  Higher-N complete: $($elapsed2.TotalMinutes.ToString('F1')) min"

# ============================================================
# RUN 3: Prompt-Type ER Re-measurement
# 6 categories × 30 prompts × 64 tokens = ~11,520 samples total
# Independent Welford per category
# Expected: ~60 min
# ============================================================
Write-Host ""
Write-Host "[RUN 3/3] Prompt-Type ER — 6 categories × 30 prompts × 64 tokens"
Write-Host "  Output: logs/prompt_types/"
Write-Host "  Log: logs/phase3_run3_prompt_types.log"
$t3 = Get-Date

python scripts/run_prompt_types.py --max-prompts-per-cat 30 --max-tokens 64 --prompts prompts/prompts_200.json --output-dir logs/prompt_types 2>&1 | Tee-Object -FilePath logs/phase3_run3_prompt_types.log

$elapsed3 = (Get-Date) - $t3
Write-Host ""
Write-Host "  Prompt-Types complete: $($elapsed3.TotalMinutes.ToString('F1')) min"

# ============================================================
# SUMMARY
# ============================================================
$totalElapsed = (Get-Date) - $t_total

$summary = "========================================`n"
$summary += "PHASE 3 DEEP PROBING COMPLETE`n"
$summary += "========================================`n"
$summary += "Started:  $($t_total.ToString())`n"
$summary += "Finished: $(Get-Date)`n"
$summary += "Total:    $($totalElapsed.TotalHours.ToString('F1')) hours`n`n"
$summary += "Run 1 (CKA Norm 60x64):           $($elapsed1.TotalMinutes.ToString('F1')) min - logs/cka_norm/`n"
$summary += "Run 2 (Higher-N 200x144):          $($elapsed2.TotalMinutes.ToString('F1')) min - logs/high_n_converged/`n"
$summary += "Run 3 (Prompt-Types 6x30x64):     $($elapsed3.TotalMinutes.ToString('F1')) min - logs/prompt_types/`n"
$summary += "========================================`n"

Write-Host $summary

$summary | Out-File -FilePath logs/phase3_summary.log -Encoding utf8

Write-Host ""
Write-Host "All results saved. Check logs/phase3_summary.log for recap."
Write-Host ""
Write-Host "========================================"
Write-Host "ALL PHASE 3 RUNS COMPLETED."
Write-Host "Finished: $(Get-Date)"
Write-Host "Total Time: $($totalElapsed.TotalHours.ToString('F2')) hours ✔"
Write-Host "Logs saved to D:\Genesis\logs\"
Write-Host "========================================"
