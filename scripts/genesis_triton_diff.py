"""
Head-to-head Genesis-v3 validation: triton primary vs no-triton fallback,
same Python process, same seed (config.seed=42, pinned in validate_genesis),
same exp config, same Wanda guard patch.

Reports val_loss/ppl/train_loss for each backend and the delta, and counts
how many prune_weights() calls hit the no-stats branch on each backend so
we know whether the triton path's slightly worse ppl is pure backend or
also "fewer experts got proper Wanda".
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

_genesis_root_env = os.environ.get("GENESIS_ROOT")
if not _genesis_root_env:
    raise RuntimeError(
        "GENESIS_ROOT environment variable is not set. "
        "Set it to the path of your Genesis v3 model root, e.g.:\n"
        "  export GENESIS_ROOT=/path/to/orchOSModel-genesis-v3"
    )
GENESIS_ROOT = Path(_genesis_root_env)
if str(GENESIS_ROOT) not in sys.path:
    sys.path.insert(0, str(GENESIS_ROOT))

# Instrument prune_weights to count no-stats fallback hits per call.
import genesis.moe.experts.pruning as pruning_mod  # noqa: E402

_orig_prune = pruning_mod._PruningMixin.prune_weights  # type: ignore[attr-defined]
_counter = {"none_stats_calls": 0, "with_stats_calls": 0}


def _instrumented_prune(self, sparsity: float = 0.2, activation_stats: dict | None = None):
    has_any = False
    if activation_stats is not None:
        for k in ("up_input", "down_input"):
            if activation_stats.get(k) is not None:
                has_any = True
                break
    if has_any:
        _counter["with_stats_calls"] += 1
    else:
        _counter["none_stats_calls"] += 1
    return _orig_prune(self, sparsity=sparsity, activation_stats=activation_stats)


pruning_mod._PruningMixin.prune_weights = _instrumented_prune  # type: ignore[attr-defined]

import validate_genesis as vg  # noqa: E402

# Mirror prism harness flags exactly (model_runs.py:693)
BASE_FLAGS = dict(
    hopmoe=True,
    mod=True,
    mom=True,
    ms=True,
    ortho=True,
    variance=True,
    entropy=False,
    expert_proj=True,
    baldwin=True,
    adaptive=True,
    n_experts=8,
    prune=True,
)

MAX_ITERS = 50
BATCH_SIZE = 8
BLOCK_SIZE = 64


def run_one(label: str, use_triton: bool) -> dict:
    _counter["none_stats_calls"] = 0
    _counter["with_stats_calls"] = 0

    print(f"\n{'='*60}\n  {label}  (use_triton={use_triton})\n{'='*60}", flush=True)

    config = vg.Config()
    config.batch_size = BATCH_SIZE
    config.block_size = BLOCK_SIZE
    config.max_iters = MAX_ITERS
    config.eval_interval = max(10, min(config.eval_interval, MAX_ITERS))
    train_loader, val_loader, vocab = vg.load_data(config)
    config.vocab_size = vocab

    flags = dict(BASE_FLAGS)
    flags["use_triton"] = use_triton

    tempdir = tempfile.TemporaryDirectory(prefix=f"genesis-diff-{label.replace(' ', '_')}-")
    original_cwd = Path.cwd()
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    t0 = time.perf_counter()
    try:
        os.chdir(tempdir.name)
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            result = vg.run_exp(
                "Baseline_MoC_v2.9a",
                config,
                train_loader,
                val_loader,
                **flags,
            )
    finally:
        os.chdir(original_cwd)
        tempdir.cleanup()
    elapsed = time.perf_counter() - t0

    result = result or {}
    val_loss = float(result.get("val_loss", 0.0))
    ppl = float(result.get("ppl", 0.0))
    train_loss = float(result.get("train_loss", 0.0))

    print(f"  val_loss={val_loss:.6f}  ppl={ppl:.6f}  train_loss={train_loss:.6f}  ({elapsed:.1f}s)")
    print(
        f"  prune_weights calls: with_stats={_counter['with_stats_calls']}  "
        f"none_stats={_counter['none_stats_calls']}"
    )

    return {
        "label": label,
        "use_triton": use_triton,
        "val_loss": val_loss,
        "ppl": ppl,
        "train_loss": train_loss,
        "elapsed_s": elapsed,
        "prune_with_stats": _counter["with_stats_calls"],
        "prune_none_stats": _counter["none_stats_calls"],
    }


def main() -> int:
    triton = run_one("triton primary", use_triton=True)
    notriton = run_one("no-triton fallback", use_triton=False)

    delta_val = triton["val_loss"] - notriton["val_loss"]
    delta_ppl = triton["ppl"] - notriton["ppl"]

    print("\n" + "=" * 60)
    print("  Genesis-v3 backend diff (same seed=42, same exp, same flags)")
    print("=" * 60)
    print(f"  triton    : val_loss={triton['val_loss']:.6f}  ppl={triton['ppl']:.6f}")
    print(f"  no-triton : val_loss={notriton['val_loss']:.6f}  ppl={notriton['ppl']:.6f}")
    print(f"  delta     : val_loss={delta_val:+.6f}  ppl={delta_ppl:+.6f}")
    print(
        f"  prune_weights none_stats counts: triton={triton['prune_none_stats']}  "
        f"no-triton={notriton['prune_none_stats']}"
    )
    if triton["prune_none_stats"] != notriton["prune_none_stats"]:
        print(
            "  >>> CONFOUNDER: backends invoke the no-stats Wanda branch "
            "different numbers of times, so the ppl gap is not pure backend."
        )
    elif triton["prune_none_stats"] == 0 and notriton["prune_none_stats"] == 0:
        print("  >>> Clean comparison: full Wanda on both backends, gap is pure kernel.")
    else:
        print(
            "  >>> Equal no-stats counts on both backends: gap is pure kernel "
            "but both miss the same experts."
        )

    out_dir = Path(r"D:\prism\logs\model-runs\genesis-v3-backend-diff")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d_%H-%MZ", time.gmtime())
    report = {
        "started": ts,
        "config": {
            "exp": "Baseline_MoC_v2.9a",
            "seed": 42,
            "max_iters": MAX_ITERS,
            "batch_size": BATCH_SIZE,
            "block_size": BLOCK_SIZE,
            "flags": BASE_FLAGS,
        },
        "triton": triton,
        "no_triton": notriton,
        "delta": {"val_loss": delta_val, "ppl": delta_ppl},
    }
    out_path = out_dir / f"{ts}_diff.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"\n  Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
