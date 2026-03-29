# ============================================================
# PHASE 4: Extended Robustness Testing — Sequential Runner
# ============================================================
# 4A: ER Convergence Curve         (~45 min)
# 4B: Bootstrap CI (10 reps)       (~60 min)
# 4C: GLA vs FoX full profile      (~12 min)
# 4D: Per-Prompt ER Variance       (~12 min)
# ============================================================
# Total estimated runtime: ~2.5 hours
# ============================================================

$env:PYTHONIOENCODING="utf-8"

Write-Host "========================================"
Write-Host "STARTING PHASE 4: EXTENDED ROBUSTNESS"
Write-Host "Started: $(Get-Date)"
Write-Host "Expected total time: ~2.5 hours"
Write-Host "========================================"

$t_total = Get-Date

# ============================================================
# RUN 4A: ER Convergence Curve
# Multiple N values to plot ER convergence
# ============================================================
Write-Host ""
Write-Host "[RUN 4A] ER Convergence Curve"
$t4a = Get-Date

# N/d=1: 9 prompts x 64 tokens = 576 samples
Write-Host "  [4A-1] N=576 (N/d=1)"
python scripts/run_corrected_er.py --max-prompts 9 --max-tokens 64 --prompts prompts/prompts_200.json --output-dir logs/phase4/convergence/n576 2>&1 | Tee-Object -FilePath logs/phase4/convergence_n576.log

# N/d=5: 45 prompts x 64 tokens = 2880 samples
Write-Host "  [4A-2] N=2880 (N/d=5)"
python scripts/run_corrected_er.py --max-prompts 45 --max-tokens 64 --prompts prompts/prompts_200.json --output-dir logs/phase4/convergence/n2880 2>&1 | Tee-Object -FilePath logs/phase4/convergence_n2880.log

# N/d=10: 90 prompts x 64 tokens = 5760 samples
Write-Host "  [4A-3] N=5760 (N/d=10)"
python scripts/run_corrected_er.py --max-prompts 90 --max-tokens 64 --prompts prompts/prompts_200.json --output-dir logs/phase4/convergence/n5760 2>&1 | Tee-Object -FilePath logs/phase4/convergence_n5760.log

# N/d=20: 180 prompts x 64 tokens = 11520 samples
Write-Host "  [4A-4] N=11520 (N/d=20)"
python scripts/run_corrected_er.py --max-prompts 180 --max-tokens 64 --prompts prompts/prompts_200.json --output-dir logs/phase4/convergence/n11520 2>&1 | Tee-Object -FilePath logs/phase4/convergence_n11520.log

$elapsed4a = (Get-Date) - $t4a
Write-Host "  Convergence curve complete: $($elapsed4a.TotalMinutes.ToString('F1')) min"

# ============================================================
# RUN 4B: Bootstrap Confidence Intervals
# 10 replicates at N=3840
# ============================================================
Write-Host ""
Write-Host "[RUN 4B] Bootstrap CI — 10 replicates × 60 prompts × 64 tokens"
$t4b = Get-Date

python scripts/run_bootstrap_er.py --n-replicates 10 --max-prompts 60 --max-tokens 64 --output-dir logs/phase4/bootstrap_ci 2>&1 | Tee-Object -FilePath logs/phase4/phase4_bootstrap.log

$elapsed4b = (Get-Date) - $t4b
Write-Host "  Bootstrap CI complete: $($elapsed4b.TotalMinutes.ToString('F1')) min"

# ============================================================
# RUN 4C: GLA vs FoX Full Sub-Block Profile
# All 30 layers, full sub-block analysis
# ============================================================
Write-Host ""
Write-Host "[RUN 4C] GLA vs FoX — Full 30-layer sub-block profile"
$t4c = Get-Date

python scripts/run_gla_vs_fox.py --max-prompts 60 --max-tokens 64 --output-dir logs/phase4/gla_vs_fox 2>&1 | Tee-Object -FilePath logs/phase4/phase4_gla_fox.log

$elapsed4c = (Get-Date) - $t4c
Write-Host "  GLA vs FoX complete: $($elapsed4c.TotalMinutes.ToString('F1')) min"

# ============================================================
# RUN 4D: Per-Prompt ER Variance
# Independent ER per prompt
# ============================================================
Write-Host ""
Write-Host "[RUN 4D] Per-Prompt Variance — 60 prompts × 64 tokens each"
$t4d = Get-Date

python scripts/run_per_prompt_er.py --max-prompts 60 --max-tokens 64 --output-dir logs/phase4/per_prompt_variance 2>&1 | Tee-Object -FilePath logs/phase4/phase4_per_prompt.log

$elapsed4d = (Get-Date) - $t4d
Write-Host "  Per-Prompt Variance complete: $($elapsed4d.TotalMinutes.ToString('F1')) min"

# ============================================================
# SUMMARY
# ============================================================
$totalElapsed = (Get-Date) - $t_total

$summary = "========================================`n"
$summary += "PHASE 4 EXTENDED ROBUSTNESS COMPLETE`n"
$summary += "========================================`n"
$summary += "Started:  $($t_total.ToString())`n"
$summary += "Finished: $(Get-Date)`n"
$summary += "Total:    $($totalElapsed.TotalHours.ToString('F1')) hours`n`n"
$summary += "4A Convergence Curve:     $($elapsed4a.TotalMinutes.ToString('F1')) min`n"
$summary += "4B Bootstrap CI (10x):    $($elapsed4b.TotalMinutes.ToString('F1')) min`n"
$summary += "4C GLA vs FoX:            $($elapsed4c.TotalMinutes.ToString('F1')) min`n"
$summary += "4D Per-Prompt Variance:   $($elapsed4d.TotalMinutes.ToString('F1')) min`n"
$summary += "========================================`n"

Write-Host $summary
$summary | Out-File -FilePath logs/phase4/phase4_summary.log -Encoding utf8

Write-Host "ALL PHASE 4 RUNS COMPLETED: $(Get-Date)"
