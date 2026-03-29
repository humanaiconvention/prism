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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run bounded model analyses and write Prism logs.")
    parser.add_argument("--prism-root", type=Path, default=Path(r"D:\prism"))
    parser.add_argument("--genesis-root", type=Path, default=Path(r"D:\Genesis\external\orchOSModel-genesis-v3"))
    parser.add_argument("--trm-root", type=Path, default=Path(r"D:\TRM"))
    parser.add_argument("--genesis-max-iters", type=int, default=50)
    parser.add_argument("--genesis-batch-size", type=int, default=8)
    parser.add_argument("--genesis-block-size", type=int, default=64)
    parser.add_argument("--genesis-bench-steps", type=int, default=2)
    parser.add_argument("--trm-epochs", type=int, default=5)
    parser.add_argument("--trm-examples", type=int, default=18)
    parser.add_argument("--trm-hidden-size", type=int, default=64)
    parser.add_argument("--trm-arch-seq-len", type=int, default=8)
    parser.add_argument(
        "--target",
        action="append",
        choices=["genesis", "trm"],
        help="Select which analyses to run. Defaults to both.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = args.target or ["genesis", "trm"]
    prism_root = args.prism_root.resolve()

    runs: list[tuple[str, object]] = []
    if "genesis" in targets:
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

    for label, bundle in runs:
        print(f"{label}: {bundle.run_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
