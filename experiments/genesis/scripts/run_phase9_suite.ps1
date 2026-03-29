# ============================================================
# PHASE 9: Semantic Manifold Mapping and Steering — Orchestrator
# ============================================================
# 9A: Semantic Extraction           (~30 min)
# 9B: Manifold Mapping              (~10 min)
# 9C: Semantic Direction Isolation   (~15 min)
# 9D: Layerwise Semantic Evolution   (~15 min)
# 9E: Semantic Rotational Dynamics   (~20 min)
# 9F: Causal Steering Quantification (~30 min)
# 9G: Activation Patching / Swap     (~30 min)
# 9I: Causal Triangle                (~35 min)
# 9J: Energy Landscape               (~25 min)
# 9K: Counterfactual Trajectory      (~25 min)
# 9L: Closed-Loop Control            (~20 min)
# 9M: Token Window Patching          (~35 min)
# 9N: Directional Patching           (~15 min)
# 9O: Sign-Aware Readout Refinement  (~10 min)
# 9P: Readout-Localized Patching     (~15 min)
# ============================================================
# Total estimated runtime: ~5.4 hours
# ============================================================

$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUNBUFFERED="1"

$logDir = "logs/phase9"
if (!(Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir }

$statusFile = "$logDir/phase9_status.log"
"PHASE 9 STARTED: $(Get-Date)" | Out-File -FilePath $statusFile -Encoding utf8

Write-Host "========================================"
Write-Host "STARTING PHASE 9: SEMANTIC MANIFOLD SUITE"
Write-Host "Started: $(Get-Date)"
Write-Host "Expected total time: ~5.0 hours"
Write-Host "========================================"

$t_total = Get-Date

function Invoke-Phase {
    param(
        [string]$Name,
        [string]$Script,
        [string]$ScriptArgs,
        [string]$LogFile,
        [string]$Marker
    )
    
    if (Test-Path "$logDir/$Marker") {
        Write-Host "`n[SKIP] $Name (Marker $Marker exists)" -ForegroundColor Cyan
        return
    }

    Write-Host "`n[$Name] Running $Script $ScriptArgs" -ForegroundColor Yellow
    $startTime = Get-Date
    
    # Run python with unbuffered output and tee to log
    $argList = if ([string]::IsNullOrWhiteSpace($ScriptArgs)) { @() } else { $ScriptArgs -split ' ' }
    & python -u $Script @argList 2>&1 | Tee-Object -FilePath "$logDir/$LogFile"
    
    if ($LASTEXITCODE -eq 0) {
        $elapsed = (Get-Date) - $startTime
        $msg = "  $Name SUCCESS - $($elapsed.TotalMinutes.ToString('F1')) min"
        Write-Host $msg -ForegroundColor Green
        $msg | Out-File -FilePath $statusFile -Append -Encoding utf8
        "COMPLETE" | Out-File -FilePath "$logDir/$Marker" -Encoding utf8
    } else {
        $msg = "  $Name FAILED with exit code $LASTEXITCODE"
        Write-Host $msg -ForegroundColor Red
        $msg | Out-File -FilePath $statusFile -Append -Encoding utf8
        exit $LASTEXITCODE
    }
}

# ------------------------------------------------------------
# 9A: Extraction
# ------------------------------------------------------------
Invoke-Phase -Name "9A: Extraction" `
          -Script "scripts/run_phase9_extract.py" `
          -ScriptArgs "--max-prompts 120 --max-tokens 64" `
          -LogFile "9a_extract.log" `
          -Marker "SUCCESS_9A"

# ------------------------------------------------------------
# 9B: Manifold Mapping
# ------------------------------------------------------------
Invoke-Phase -Name "9B: Manifold Mapping" `
          -Script "scripts/run_phase9_manifold.py" `
          -ScriptArgs "" `
          -LogFile "9b_manifold.log" `
          -Marker "SUCCESS_9B"

# ------------------------------------------------------------
# 9C: Semantic Direction Isolation
# ------------------------------------------------------------
Invoke-Phase -Name "9C: Semantic Direction Isolation" `
          -Script "scripts/run_phase9_semantic_dirs.py" `
          -ScriptArgs "--k-bulk 70 --min-retained-fraction 0.25" `
          -LogFile "9c_semantic_dirs.log" `
          -Marker "SUCCESS_9C"

# ------------------------------------------------------------
# 9D: Layerwise Semantic Evolution
# ------------------------------------------------------------
Invoke-Phase -Name "9D: Layerwise Semantic Evolution" `
          -Script "scripts/run_phase9_layerwise_semantics.py" `
          -ScriptArgs "" `
          -LogFile "9d_layerwise.log" `
          -Marker "SUCCESS_9D"

# ------------------------------------------------------------
# 9E: Semantic Rotational Dynamics
# ------------------------------------------------------------
Invoke-Phase -Name "9E: Semantic Rotational Dynamics" `
          -Script "scripts/run_phase9_semantic_rotation.py" `
          -ScriptArgs "" `
          -LogFile "9e_rotation.log" `
          -Marker "SUCCESS_9E"

# ------------------------------------------------------------
# 9F: Causal Steering Quantification
# ------------------------------------------------------------
Invoke-Phase -Name "9F: Causal Steering Quantification" `
          -Script "scripts/run_phase9_semantic_steering.py" `
          -ScriptArgs "--lambda-sweep 0.0,5.0,12.5 --vector-key delta_perp --eval-json prompts/phase9_shared_eval_heldout.json" `
          -LogFile "9f_steering.log" `
          -Marker "SUCCESS_9F"

# ------------------------------------------------------------
# 9G: Activation Patching / Representation Swap
# ------------------------------------------------------------
Invoke-Phase -Name "9G: Activation Patching / Representation Swap" `
          -Script "scripts/run_phase9_activation_patching.py" `
          -ScriptArgs "--layer 15 --alpha-sweep 0.0,0.5,1.0 --eval-json prompts/phase9_shared_eval_heldout.json" `
          -LogFile "9g_activation_patching.log" `
          -Marker "SUCCESS_9G"

# ------------------------------------------------------------
# 9I: Causal Triangle Diagnostic
# ------------------------------------------------------------
Invoke-Phase -Name "9I: Causal Triangle Diagnostic" `
          -Script "scripts/run_phase9_causal_triangle.py" `
          -ScriptArgs "--layers 13,14,15,16,17 --eval-json prompts/phase9_shared_eval_heldout.json --semantic-directions logs/phase9/semantic_directions.json" `
          -LogFile "9i_causal_triangle.log" `
          -Marker "SUCCESS_9I"

# ------------------------------------------------------------
# 9J: Residual Stream Energy Landscape
# ------------------------------------------------------------
Invoke-Phase -Name "9J: Residual Stream Energy Landscape" `
          -Script "scripts/run_phase9_energy_landscape.py" `
          -ScriptArgs "--layers 15 --eval-json prompts/phase9_shared_eval_heldout.json --semantic-directions logs/phase9/semantic_directions.json" `
          -LogFile "9j_energy_landscape.log" `
          -Marker "SUCCESS_9J"

# ------------------------------------------------------------
# 9K: Counterfactual Trajectory Patching
# ------------------------------------------------------------
Invoke-Phase -Name "9K: Counterfactual Trajectory Patching" `
          -Script "scripts/run_phase9_counterfactual_trajectory.py" `
          -ScriptArgs "--layers 13,14,15,16,17 --start-layer 15 --eval-json prompts/phase9_shared_eval_heldout.json --semantic-directions logs/phase9/semantic_directions.json" `
          -LogFile "9k_counterfactual_trajectory.log" `
          -Marker "SUCCESS_9K"

# ------------------------------------------------------------
# 9L: Closed-Loop Behavioral Control
# ------------------------------------------------------------
Invoke-Phase -Name "9L: Closed-Loop Behavioral Control" `
          -Script "scripts/run_phase9_closed_loop_control.py" `
          -ScriptArgs "--layer 15 --target-levels=-3,-1,0,1,3 --eval-json prompts/phase9_shared_eval_heldout.json --semantic-directions logs/phase9/semantic_directions.json" `
          -LogFile "9l_closed_loop_control.log" `
          -Marker "SUCCESS_9L"

# ------------------------------------------------------------
# 9M: Token Window Patching
# ------------------------------------------------------------
Invoke-Phase -Name "9M: Token Window Patching" `
          -Script "scripts/run_phase9_token_window_patching.py" `
          -ScriptArgs "--layers 13,14,15,16,17 --window-sizes 1,2,3,4 --eval-json prompts/phase9_shared_eval_heldout.json" `
          -LogFile "9m_token_window_patching.log" `
          -Marker "SUCCESS_9M"

# ------------------------------------------------------------
# 9N: Directional Patching
# ------------------------------------------------------------
Invoke-Phase -Name "9N: Directional Patching" `
          -Script "scripts/run_phase9_directional_patching.py" `
          -ScriptArgs "--layers 15,16,17 --alpha-sweep 0.5,1.0,2.0 --eval-json prompts/phase9_shared_eval_heldout.json --semantic-directions logs/phase9/semantic_directions.json" `
          -LogFile "9n_directional_patching.log" `
          -Marker "SUCCESS_9N"

# ------------------------------------------------------------
# 9O: Sign-Aware Readout Refinement
# ------------------------------------------------------------
Invoke-Phase -Name "9O: Sign-Aware Readout Refinement" `
          -Script "scripts/run_phase9_semantic_readout_refinement.py" `
          -ScriptArgs "--layer 17 --alpha-sweep 0.5,1.0,2.0 --eval-json prompts/phase9_shared_eval_heldout.json --semantic-directions logs/phase9/semantic_directions.json" `
          -LogFile "9o_semantic_readout_refinement.log" `
          -Marker "SUCCESS_9O"

# ------------------------------------------------------------
# 9P: Readout-Localized Patching
# ------------------------------------------------------------
Invoke-Phase -Name "9P: Readout-Localized Patching" `
          -Script "scripts/run_phase9_readout_localized_patching.py" `
          -ScriptArgs "--layer 17 --alpha-sweep 0.5,1.0,2.0 --eval-json prompts/phase9_shared_eval_heldout.json --semantic-directions logs/phase9/semantic_directions.json" `
          -LogFile "9p_readout_localized_patching.log" `
          -Marker "SUCCESS_9P"

# ------------------------------------------------------------
# 9P-Check: Readout-Localized Artifact Validation
# ------------------------------------------------------------
Invoke-Phase -Name "9P-Check: Readout-Localized Artifact Validation" `
          -Script "scripts/run_phase9_readout_localized_patching.py" `
          -ScriptArgs "--output-csv logs/phase9/readout_localized_patching_results.csv --validate-existing" `
          -LogFile "9p_readout_localized_patching_validation.log" `
          -Marker "SUCCESS_9P_VALIDATE"

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------
$totalElapsed = (Get-Date) - $t_total
$summary = "`n========================================`n"
$summary += "PHASE 9 COMPLETE`n"
$summary += "Total Time: $($totalElapsed.TotalHours.ToString('F2')) hours`n"
$summary += "========================================`n"

Write-Host $summary -ForegroundColor Green
$summary | Out-File -FilePath $statusFile -Append -Encoding utf8

Write-Host "Logs and data saved to $logDir/"
