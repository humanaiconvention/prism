# Genesis GPU Overnight Batch
# Runs Phase 1 (Head Ablation) and Phase 2 (Gap-Filling Diagnostics)

Set-Location D:\Genesis
$ErrorActionPreference = "Continue"

$startTime = Get-Date
Write-Host "========================================"
Write-Host "STARTING FULL GPU OVERNIGHT BATCH"
Write-Host "Started: $startTime"
Write-Host "Expected total time: ~4-5 hours"
Write-Host "========================================"

Write-Host "`n>>> RUNNING PHASE 1 (High-N & Head Ablation)"
.\scripts\run_overnight2.ps1

Write-Host "`n>>> RUNNING PHASE 2 (W_o Probe, TTT Control, Diversity)"
.\scripts\run_overnight3.ps1

$endTime = Get-Date
$totalElapsed = $endTime - $startTime
Write-Host "`n========================================"
Write-Host "ALL RUNS COMPLETED"
Write-Host "Finished: $endTime"
Write-Host "Total Time: $($totalElapsed.TotalHours.ToString('F2')) hours"
Write-Host "Logs saved to D:\Genesis\logs\"
Write-Host "========================================"
