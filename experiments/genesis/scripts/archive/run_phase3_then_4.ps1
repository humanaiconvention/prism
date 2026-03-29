# ============================================================
# Phase 3 → Phase 4 Chainer
# Waits for Phase 3 to finish, cleans up logs, launches Phase 4
# ============================================================

$env:PYTHONIOENCODING="utf-8"

Write-Host "========================================"
Write-Host "WAITING FOR PHASE 3 TO COMPLETE..."
Write-Host "Started monitor: $(Get-Date)"
Write-Host "========================================"

# Wait for Phase 3 to finish by checking for the phase3_summary.log
# Phase 3 writes to logs/ directly (original paths), so we watch for its summary
while (-not (Test-Path "logs\phase3_summary.log")) {
    Start-Sleep -Seconds 30
    Write-Host "  $(Get-Date -Format 'HH:mm:ss') — Phase 3 still running..."
}

Write-Host ""
Write-Host "Phase 3 complete! Moving new logs to phase3/ directory..."

# Move any newly created Phase 3 outputs
$phase3_dirs = @("high_n_converged", "prompt_types")
foreach ($d in $phase3_dirs) {
    if (Test-Path "logs\$d") {
        Move-Item -Path "logs\$d" -Destination "logs\phase3\$d" -Force
        Write-Host "  Moved $d -> phase3/$d"
    }
}
$phase3_files = @("phase3_run2_high_n.log", "phase3_run3_prompt_types.log", "phase3_summary.log")
foreach ($f in $phase3_files) {
    if (Test-Path "logs\$f") {
        Move-Item -Path "logs\$f" -Destination "logs\phase3\$f" -Force
        Write-Host "  Moved $f -> phase3/$f"
    }
}

Write-Host ""
Write-Host "Launching Phase 4..."
Write-Host "========================================"

# Launch Phase 4
& .\scripts\run_phase4.ps1
