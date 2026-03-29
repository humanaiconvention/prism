# Genesis Overnight Run - 6.5 hour batch
# Chains 3 experiments with full logging
# Each experiment logs stdout+stderr to its own log file

$ErrorActionPreference = "Continue"
$env:TRITON_INTERPRET = "1"
$env:PYTHONIOENCODING = "utf-8"

Set-Location D:\Genesis

$startTime = Get-Date
Write-Host "========================================"
Write-Host "GENESIS OVERNIGHT RUN"
Write-Host "Started: $startTime"
Write-Host "Expected completion: ~5:00 AM"
Write-Host "========================================"

# ============================================================
# RUN 1: Corrected ER - Normal (60 prompts x 64 tokens)
# N=3840 samples, N/d=6.7x
# Expected: ~3.5 hours
# ============================================================
Write-Host ""
Write-Host "[RUN 1/3] Corrected ER (Normal) - 60 prompts x 64 tokens"
Write-Host "  Output: logs/corrected/"
Write-Host "  Log: logs/overnight_run1_normal.log"
$t1 = Get-Date

python scripts/run_corrected_er.py --max-prompts 60 --max-tokens 64 --output-dir logs/corrected 2>&1 | Tee-Object -FilePath logs/overnight_run1_normal.log

$elapsed1 = (Get-Date) - $t1
Write-Host ""
Write-Host "[RUN 1/3] DONE in $($elapsed1.TotalMinutes.ToString('F1')) min"
Write-Host "  Exit code: $LASTEXITCODE"

# ============================================================
# RUN 2: Corrected ER - TTT Disabled (30 prompts x 64 tokens)
# N=1920, N/d=3.3x
# Expected: ~1.5 hours
# ============================================================
Write-Host ""
Write-Host "[RUN 2/3] Corrected ER (TTT Disabled) - 30 prompts x 64 tokens"
Write-Host "  Output: logs/corrected_ttt_off/"
Write-Host "  Log: logs/overnight_run2_ttt_off.log"
$t2 = Get-Date

python scripts/run_corrected_er.py --max-prompts 30 --max-tokens 64 --output-dir logs/corrected_ttt_off --disable-ttt 2>&1 | Tee-Object -FilePath logs/overnight_run2_ttt_off.log

$elapsed2 = (Get-Date) - $t2
Write-Host ""
Write-Host "[RUN 2/3] DONE in $($elapsed2.TotalMinutes.ToString('F1')) min"
Write-Host "  Exit code: $LASTEXITCODE"

# ============================================================
# RUN 3: Corrected ER - Extended tokens (20 prompts x 128 tokens)
# N=2560, N/d=4.4x
# Expected: ~1.5 hours
# ============================================================
Write-Host ""
Write-Host "[RUN 3/3] Corrected ER (Extended) - 20 prompts x 128 tokens"
Write-Host "  Output: logs/corrected_extended/"
Write-Host "  Log: logs/overnight_run3_extended.log"
$t3 = Get-Date

python scripts/run_corrected_er.py --max-prompts 20 --max-tokens 128 --output-dir logs/corrected_extended 2>&1 | Tee-Object -FilePath logs/overnight_run3_extended.log

$elapsed3 = (Get-Date) - $t3
Write-Host ""
Write-Host "[RUN 3/3] DONE in $($elapsed3.TotalMinutes.ToString('F1')) min"
Write-Host "  Exit code: $LASTEXITCODE"

# ============================================================
# SUMMARY
# ============================================================
$totalElapsed = (Get-Date) - $startTime
$endTime = Get-Date

$summary = "========================================`n"
$summary += "OVERNIGHT RUN COMPLETE`n"
$summary += "========================================`n"
$summary += "Started:  $startTime`n"
$summary += "Finished: $endTime`n"
$summary += "Total:    $($totalElapsed.TotalHours.ToString('F1')) hours`n`n"
$summary += "Run 1 (Normal 60x64):     $($elapsed1.TotalMinutes.ToString('F1')) min - logs/corrected/`n"
$summary += "Run 2 (TTT-off 30x64):    $($elapsed2.TotalMinutes.ToString('F1')) min - logs/corrected_ttt_off/`n"
$summary += "Run 3 (Extended 20x128):  $($elapsed3.TotalMinutes.ToString('F1')) min - logs/corrected_extended/`n"
$summary += "========================================`n"

Write-Host $summary
$summary | Out-File -FilePath logs/overnight_summary.log -Encoding utf8
Write-Host "All results saved. Check logs/overnight_summary.log for recap."
