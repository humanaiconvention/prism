#!/usr/bin/env python3
"""Run bounded Prism analyses for Genesis v3 and TRM."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prism.runs.model_runs import (
    analyze_genesis_v3,
    analyze_trm_physics_validation,
    write_run_bundle,
)
from prism.runs.gemma4 import analyze_gemma4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded model analyses and write Prism logs.")
    parser.add_argument("--prism-root", type=Path, default=ROOT)
    parser.add_argument("--genesis-root", type=Path, default=None,
                        help="Path to the Genesis v3 model root. Required when --target genesis is used.")
    parser.add_argument("--trm-root", type=Path, default=None,
                        help="Path to the TRM model root. Required when --target trm is used.")
    parser.add_argument("--genesis-max-iters", type=int, default=50)
    parser.add_argument("--genesis-batch-size", type=int, default=8)
    parser.add_argument("--genesis-block-size", type=int, default=64)
    parser.add_argument("--genesis-bench-steps", type=int, default=2)
    parser.add_argument("--trm-epochs", type=int, default=5)
    parser.add_argument("--trm-examples", type=int, default=18)
    parser.add_argument("--trm-hidden-size", type=int, default=64)
    parser.add_argument("--trm-arch-seq-len", type=int, default=8)
    parser.add_argument(
        "--gemma4-model-id",
        type=str,
        default="google/gemma-4-12b-it",
        help="HuggingFace hub id or local path for the Gemma 4 analyze run.",
    )
    parser.add_argument(
        "--gemma4-demo",
        action="store_true",
        help="Run the Gemma 4 analysis in demo mode (numpy stubs, no GPU/torch).",
    )
    parser.add_argument(
        "--gemma4-4bit",
        action="store_true",
        help="Load the Gemma 4 model in 4-bit NF4 (bitsandbytes) on CUDA. Required for 26B/31B variants on 8GB VRAM; recommended for E4B on 32GB-RAM hosts where bf16 CPU loads can OOM.",
    )
    parser.add_argument(
        "--target",
        action="append",
        choices=["genesis", "trm", "gemma4"],
        help="Select which analyses to run. If --target is given at least once, only those run; otherwise defaults to genesis+trm (gemma4 is opt-in).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = args.target or ["genesis", "trm"]
    prism_root = args.prism_root.resolve()

    runs: list[tuple[str, object]] = []
    if "genesis" in targets:
        if args.genesis_root is None:
            print("error: --genesis-root is required when --target genesis is used", file=sys.stderr)
            return 1
        run = analyze_genesis_v3(
            args.genesis_root,
            prism_root=prism_root,
            max_iters=args.genesis_max_iters,
            batch_size=args.genesis_batch_size,
            block_size=args.genesis_block_size,
            bench_steps=args.genesis_bench_steps,
        )
        bundle = write_run_bundle(run, prism_root=prism_root, mirror_root=args.genesis_root)
        runs.append(("Genesis v3", bundle))

    if "trm" in targets:
        if args.trm_root is None:
            print("error: --trm-root is required when --target trm is used", file=sys.stderr)
            return 1
        run = analyze_trm_physics_validation(
            args.trm_root,
            prism_root=prism_root,
            physics_epochs=args.trm_epochs,
            physics_examples=args.trm_examples,
            hidden_size=args.trm_hidden_size,
            arch_seq_len=args.trm_arch_seq_len,
        )
        bundle = write_run_bundle(run, prism_root=prism_root, mirror_root=args.trm_root)
        runs.append(("TRM", bundle))

    if "gemma4" in targets:
        # analyze_gemma4 handles its own artifact writing (run.json under
        # D:\prism\logs\model-runs\gemma4\<ts>\ and a maestro ledger record).
        # It does NOT use write_run_bundle, so we just print the run summary.
        gemma_run = analyze_gemma4(
            model_id=args.gemma4_model_id,
            demo=args.gemma4_demo,
            config={"load_in_4bit": args.gemma4_4bit},
        )
        runs.append(
            (
                f"Gemma 4 ({'demo' if args.gemma4_demo else 'real'})",
                gemma_run,
            )
        )

    for label, bundle in runs:
        if hasattr(bundle, "run_md"):
            print(f"{label}: {bundle.run_md}")
        else:
            # Gemma4Run has its own console summary already; just confirm status.
            status = getattr(bundle, "status", "unknown")
            model = getattr(bundle, "model_id", "")
            print(f"{label}: status={status}  model={model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
