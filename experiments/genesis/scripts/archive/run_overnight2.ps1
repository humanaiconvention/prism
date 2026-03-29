# Genesis Phase 1 Run - 6 hour batch
# Run 1: High-N stabilization (60 prompts x 32 tokens = 1920, ~1.5h)
# Run 2: Per-head ablation on FoX layers (20 prompts x 32 tokens x 63 conditions, ~4h)

$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONIOENCODING = "utf-8"

Set-Location D:\Genesis

$startTime = Get-Date
Write-Host "========================================"
Write-Host "GENESIS PHASE 1 RUN"
Write-Host "Started: $startTime"
Write-Host "Expected completion: ~6 hours"
Write-Host "========================================"

# ============================================================
# RUN 1: High-N stabilization (60 prompts x 32 tokens)
# N=1920 but from prompts_200 for more diversity
# Expected: ~1.5 hours
# ============================================================
Write-Host ""
Write-Host "[RUN 1/2] High-N stabilization - 60 prompts x 32 tokens"
Write-Host "  Output: logs/high_n/"
Write-Host "  Log: logs/phase1_run1_high_n.log"
$t1 = Get-Date

python scripts/run_head_rank.py --mode high-n --max-prompts 60 --max-tokens 64 --prompts prompts/prompts_200.json --output-dir logs/high_n 2>&1 | Tee-Object -FilePath logs/phase1_run1_high_n.log

$elapsed1 = (Get-Date) - $t1
Write-Host ""
Write-Host "[RUN 1/2] DONE in $($elapsed1.TotalMinutes.ToString('F1')) min"

# ============================================================
# RUN 2: Per-head ablation on FoX layers
# 7 FoX layers x 9 heads = 63 conditions + baseline
# 20 prompts x 32 tokens per condition
# Expected: ~4 hours
# ============================================================
Write-Host ""
Write-Host "[RUN 2/2] Per-head ablation - 20 prompts x 32 tokens x 64 conditions"
Write-Host "  Output: logs/head_rank/"
Write-Host "  Log: logs/phase1_run2_head_rank.log"
$t2 = Get-Date

python scripts/run_head_rank.py --mode ablation --max-prompts 60 --max-tokens 64 --prompts prompts/prompts_200.json --output-dir logs/head_rank 2>&1 | Tee-Object -FilePath logs/phase1_run2_head_rank.log

$elapsed2 = (Get-Date) - $t2
Write-Host ""
Write-Host "[RUN 2/2] DONE in $($elapsed2.TotalMinutes.ToString('F1')) min"

# ============================================================
# SUMMARY
# ============================================================
$totalElapsed = (Get-Date) - $startTime
$endTime = Get-Date

$summary = "========================================`n"
$summary += "PHASE 1 RUN COMPLETE`n"
$summary += "========================================`n"
$summary += "Started:  $startTime`n"
$summary += "Finished: $endTime`n"
$summary += "Total:    $($totalElapsed.TotalHours.ToString('F1')) hours`n`n"
$summary += "Run 1 (High-N 60x32):     $($elapsed1.TotalMinutes.ToString('F1')) min - logs/high_n/`n"
$summary += "Run 2 (Head ablation):    $($elapsed2.TotalMinutes.ToString('F1')) min - logs/head_rank/`n"
$summary += "========================================`n"

Write-Host $summary
$summary | Out-File -FilePath logs/phase1_summary.log -Encoding utf8
Write-Host "All results saved. Check logs/phase1_summary.log for recap."
