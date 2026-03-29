"""Batch runner for ~2 hour experimental suite.

Runs three experiments sequentially:
1. Layer geometry audit (normal) — 10 prompts × 64 tokens (~35 min)
2. Layer geometry audit (TTT disabled) — 10 prompts × 64 tokens (~35 min)  
3. Updated spectral profile — 10 prompts × 128 tokens (~50 min)

Total estimated: ~120 minutes
"""

import os
os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import subprocess
import sys
import time
from datetime import datetime

PYTHON = sys.executable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

def run_experiment(name, script, args):
    """Run a single experiment and report timing."""
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {name}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")
    
    cmd = [PYTHON, os.path.join(SCRIPT_DIR, script)] + args
    t0 = time.time()
    
    result = subprocess.run(
        cmd, cwd=PROJECT_DIR,
        env={**os.environ},
    )
    
    elapsed = time.time() - t0
    status = "✓ PASSED" if result.returncode == 0 else f"✗ FAILED (exit {result.returncode})"
    print(f"\n  {status} in {elapsed/60:.1f} minutes")
    return result.returncode == 0, elapsed


def main():
    print("="*60)
    print("  GENESIS 2-HOUR EXPERIMENTAL SUITE")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = []
    total_t0 = time.time()
    
    # Experiment 1: Layer geometry audit (normal)
    ok, t = run_experiment(
        "Layer Geometry Audit (GLA vs FoX comparison)",
        "run_layer_geometry.py",
        ["--max-prompts", "10", "--max-tokens", "64",
         "--output", "logs/layer_geometry_normal.csv"],
    )
    results.append(("Layer Geometry (normal)", ok, t))
    
    # Experiment 2: Layer geometry with TTT disabled
    ok, t = run_experiment(
        "Layer Geometry Audit (TTT DISABLED — control)",
        "run_layer_geometry.py",
        ["--max-prompts", "10", "--max-tokens", "64",
         "--disable-ttt",
         "--output", "logs/layer_geometry_ttt_off.csv"],
    )
    results.append(("Layer Geometry (TTT off)", ok, t))
    
    # Experiment 3: Updated spectral profile
    ok, t = run_experiment(
        "Spectral Profile (with Shannon EffRank)",
        "run_spectral_profile.py",
        ["--max-prompts", "10", "--max-tokens", "128",
         "--greedy",
         "--output", "logs/spectral_profile_v2.csv"],
    )
    results.append(("Spectral Profile v2", ok, t))
    
    # Summary
    total_elapsed = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"  SUITE COMPLETE — {total_elapsed/60:.1f} minutes total")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    
    for name, ok, t in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {name}: {t/60:.1f} min")
    
    print(f"\nOutputs:")
    print(f"  logs/layer_geometry_normal.csv")
    print(f"  logs/layer_geometry_ttt_off.csv")
    print(f"  logs/spectral_profile_v2.csv")


if __name__ == "__main__":
    main()
