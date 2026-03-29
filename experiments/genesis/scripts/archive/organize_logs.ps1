# ============================================================
# Log Directory Reorganization — Move to phase-based layout
# ============================================================

$logsDir = "D:\Genesis\logs"

# Create phase directories
New-Item -ItemType Directory -Force -Path "$logsDir\phase0"
New-Item -ItemType Directory -Force -Path "$logsDir\phase1"
New-Item -ItemType Directory -Force -Path "$logsDir\phase2"
New-Item -ItemType Directory -Force -Path "$logsDir\phase3"
New-Item -ItemType Directory -Force -Path "$logsDir\phase4"

Write-Host "=== Organizing logs by phase ==="

# --- PHASE 0: Initial measurements & corrections ---
$phase0_dirs = @("corrected", "corrected_ttt_off", "corrected_extended", "high_n")
$phase0_files = @(
    "overnight_run1_normal.log", "overnight_run2_ttt_off.log",
    "overnight_run3_extended.log", "overnight_summary.log",
    "energy_ratio.csv", "energy_ratio_summary.txt",
    "layer_geometry_normal.csv", "subblock_analysis.txt", "subblock_rank.csv",
    "analysis_output.txt", "analysis_overnight.txt",
    "smoke_spectral.csv", "smoke_test_v2.csv", "spectral_profile_v2.csv",
    "wsl_traceback.txt"
)

foreach ($d in $phase0_dirs) {
    if (Test-Path "$logsDir\$d") {
        Move-Item -Path "$logsDir\$d" -Destination "$logsDir\phase0\$d" -Force
        Write-Host "  Moved $d -> phase0/$d"
    }
}
foreach ($f in $phase0_files) {
    if (Test-Path "$logsDir\$f") {
        Move-Item -Path "$logsDir\$f" -Destination "$logsDir\phase0\$f" -Force
        Write-Host "  Moved $f -> phase0/$f"
    }
}

# --- PHASE 1: Per-head ablation ---
$phase1_dirs = @("head_rank", "head_rank_optimized")
$phase1_files = @(
    "phase1_run1_high_n.log", "phase1_run2_head_rank.log", "phase1_summary.log"
)

foreach ($d in $phase1_dirs) {
    if (Test-Path "$logsDir\$d") {
        Move-Item -Path "$logsDir\$d" -Destination "$logsDir\phase1\$d" -Force
        Write-Host "  Moved $d -> phase1/$d"
    }
}
foreach ($f in $phase1_files) {
    if (Test-Path "$logsDir\$f") {
        Move-Item -Path "$logsDir\$f" -Destination "$logsDir\phase1\$f" -Force
        Write-Host "  Moved $f -> phase1/$f"
    }
}

# --- PHASE 2: Gap-filling diagnostics ---
$phase2_dirs = @("wo_probe", "wo_probe_smoke", "corrected_ttt_off_matched",
                 "diversity_many_prompts", "diversity_many_tokens")
$phase2_files = @(
    "phase2_run1_wo_probe.log", "phase2_run2_ttt_matched.log",
    "phase2_run3a_diversity.log", "phase2_run3b_diversity.log",
    "phase2_summary.log"
)

foreach ($d in $phase2_dirs) {
    if (Test-Path "$logsDir\$d") {
        Move-Item -Path "$logsDir\$d" -Destination "$logsDir\phase2\$d" -Force
        Write-Host "  Moved $d -> phase2/$d"
    }
}
foreach ($f in $phase2_files) {
    if (Test-Path "$logsDir\$f") {
        Move-Item -Path "$logsDir\$f" -Destination "$logsDir\phase2\$f" -Force
        Write-Host "  Moved $f -> phase2/$f"
    }
}

# --- PHASE 3: Deep probing (some may still be in progress) ---
$phase3_dirs = @("cka_norm", "cka_norm_smoke", "high_n_converged",
                 "prompt_types", "prompt_types_smoke")
$phase3_files = @(
    "phase3_run1_cka.log", "phase3_run2_high_n.log", "phase3_run3_prompt_types.log",
    "phase3_summary.log"
)

foreach ($d in $phase3_dirs) {
    if (Test-Path "$logsDir\$d") {
        Move-Item -Path "$logsDir\$d" -Destination "$logsDir\phase3\$d" -Force
        Write-Host "  Moved $d -> phase3/$d"
    }
}
foreach ($f in $phase3_files) {
    if (Test-Path "$logsDir\$f") {
        Move-Item -Path "$logsDir\$f" -Destination "$logsDir\phase3\$f" -Force
        Write-Host "  Moved $f -> phase3/$f"
    }
}

Write-Host ""
Write-Host "=== Log reorganization complete ==="
Write-Host ""
Write-Host "Directory structure:"
Get-ChildItem "$logsDir" -Directory | ForEach-Object {
    $count = (Get-ChildItem $_.FullName -Recurse -File).Count
    Write-Host "  $($_.Name)/  ($count files)"
}
