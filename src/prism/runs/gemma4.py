# UPSTREAM MIRROR: D:\humanai-convention\prism\src\prism\runs\gemma4.py
# Canonical source. D:\prism\src\prism\runs\gemma4.py should be kept in sync.
# To sync: copy this file to D:\prism\src\prism\runs\gemma4.py
"""
Gemma 4 PRISM run harness.

Entrypoint: analyze_gemma4(model_id, config, demo)

Real run  — requires torch + transformers + ~5 GB VRAM (bfloat16 CPU) or
            ~3 GB VRAM (4-bit NF4).  Set demo=False.
Demo run  — numpy stubs only, no GPU/torch needed.  Set demo=True (default
            when invoked via `--demo` CLI flag).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
#
# This file lives in two places:
#   - The monorepo:  D:\humanai-convention\prism\src\prism\runs\gemma4.py
#   - The public mirror: D:\prism\src\prism\runs\gemma4.py
#
# The parent-chain calculation below resolves correctly when this file lives
# inside the monorepo (parents[3] == humanai-convention/). On the public
# mirror it resolves to the wrong place (D:\), which would silently write
# Maestro records into the public mirror's directory tree. To handle that:
#
#   1. Honor HAIC_HOME env var if set (highest priority).
#   2. Otherwise use the parent-chain calculation.
#   3. If the resolved path doesn't contain experiments/gemma4/, fall back
#      to the well-known monorepo path on this machine, which is the only
#      place where the real-mode harness lives.
#
# Demo mode does not need any of these paths to exist; it only writes the
# run.json artifact under _PRISM_LOGS.
# ---------------------------------------------------------------------------

_RUNS_DIR = Path(__file__).parent                        # …/prism/src/prism/runs/
_PRISM_SRC = _RUNS_DIR.parent.parent                    # …/prism/src/


def _resolve_haic_home() -> Path:
    env_override = os.environ.get("HAIC_HOME")
    if env_override:
        p = Path(env_override)
        if p.exists():
            return p
    parent_chain = _PRISM_SRC.parent.parent              # …/humanai-convention/ (upstream) or D:\ (mirror)
    if (parent_chain / "experiments" / "gemma4").exists():
        return parent_chain
    fallback = Path(r"D:\humanai-convention")
    if fallback.exists():
        return fallback
    # Last resort — return the parent-chain result so demo mode still
    # produces a path that downstream warn-and-continue logic can handle.
    return parent_chain


_HUMANAI_ROOT = _resolve_haic_home()
_EXPERIMENTS_GEMMA4 = _HUMANAI_ROOT / "experiments" / "gemma4"
_MAESTRO_RECORDS = _HUMANAI_ROOT / "maestro" / "data" / "records"

# canonical D:\prism log root (matches DEFAULT_LOG_ROOT in model_runs.py)
_PRISM_LOGS = Path(r"D:\prism\logs\model-runs")
_GEMMA4_LOG_ROOT = _PRISM_LOGS / "gemma4"

PRISM_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Prompts — same 20-item suite used in experiments/gemma4/run_prism_analysis.py
# ---------------------------------------------------------------------------

PROMPTS: list[str] = [
    "The mitochondria is the powerhouse of the cell.",
    "Water boils at 100 degrees Celsius at sea level.",
    "I feel a deep sense of gratitude when I watch the sunrise.",
    "The grief after losing someone you love never fully disappears.",
    (
        "Artificial intelligence systems trained on human-generated text inherit "
        "the perspectives, biases, and knowledge gaps present in that corpus. "
        "Grounding AI in verified, consented contributions from real people can "
        "mitigate this by anchoring representations in lived experience rather "
        "than filtered public records."
    ),
    (
        "The tension between privacy and collective knowledge is one of the defining "
        "challenges of the twenty-first century. Individuals have legitimate interests "
        "in controlling their own narratives, yet society benefits enormously from "
        "open sharing of information and lived expertise."
    ),
    "Why?",
    "Help.",
    "It depends.",
    "La inteligencia artificial debe servir a todas las personas, sin distinción.",
    "L'expérience humaine est irréductible à des données textuelles.",
    "人类的智慧不能被简化为数字信号。",
    "التعاون بين البشر والذكاء الاصطناعي يتطلب الثقة والشفافية.",
    "A diversidade de perspectivas fortalece qualquer sistema de aprendizado.",
    "def outlier_ratio(H): return H.abs().mean(0).max() / H.abs().mean()",
    "42, 17, 9, 3.14159, 2.71828, 1.41421, 0.577",
    "No person should be coerced into providing their data for AI training.",
    "I am not sure whether what I experience constitutes genuine understanding.",
    "Consent must be informed, revocable, and freely given.",
    "Le consentement doit être éclairé, révocable et librement accordé.",
]

assert len(PROMPTS) == 20

# ---------------------------------------------------------------------------
# Gemma 4 architecture constants (from actual run results)
# ---------------------------------------------------------------------------

GEMMA4_ARCH_NOTES: dict[str, Any] = {
    "per_layer_embeddings": True,
    "per_layer_embedding_dim": 256,
    "sliding_attention_layers": 28,
    "full_attention_layers": 7,
    "full_attention_positions": [5, 10, 15, 20, 25, 30, 35],
    "num_kv_shared_layers": 20,
    "use_double_wide_mlp": True,
    "hidden_activation": "gelu_pytorch_tanh",
    "sliding_window": 512,
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ts_label(dt: datetime | None = None) -> str:
    dt = dt or _utc_now()
    return dt.strftime("%Y-%m-%d_%H-%M-%SZ")


def _iso(dt: datetime | None = None) -> str:
    dt = dt or _utc_now()
    return dt.isoformat().replace("+00:00", "Z")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Demo-mode stub  (numpy only, no torch/transformers)
# ---------------------------------------------------------------------------

def _demo_outlier_geometry(H: "np.ndarray") -> dict[str, float]:  # type: ignore[name-defined]
    """Pure-numpy replica of prism.geometry.core.outlier_geometry for --demo mode."""
    import math
    import numpy as np

    H = H.astype(np.float32)
    seq, dim = H.shape
    if seq < 1 or dim < 1:
        return {"outlier_ratio": 1.0, "activation_kurtosis": 0.0,
                "cardinal_proximity": 0.0, "quantization_hostility": 0.0}

    dim_mag = np.abs(H).mean(axis=0)
    mean_mag = dim_mag.mean()
    max_mag = dim_mag.max()
    outlier_ratio = float(max_mag / (mean_mag + 1e-12))

    mu = dim_mag.mean()
    sigma = dim_mag.std()
    if sigma < 1e-12:
        activation_kurtosis = 0.0
    else:
        activation_kurtosis = float(((dim_mag - mu) ** 4).mean() / (sigma ** 4 + 1e-12) - 3.0)

    norms = np.linalg.norm(H, axis=-1, keepdims=True).clip(1e-12)
    h_unit = H / norms
    cardinal_proximity = float(np.abs(h_unit).max(axis=-1).mean())

    or_norm = min(math.log(max(outlier_ratio, 1.0)) / math.log(50.0), 1.0)
    ak_norm = min(max(activation_kurtosis, 0.0) / 20.0, 1.0)
    cp_norm = cardinal_proximity
    quantization_hostility = (or_norm + ak_norm + cp_norm) / 3.0

    return {
        "outlier_ratio": round(outlier_ratio, 4),
        "activation_kurtosis": round(activation_kurtosis, 4),
        "cardinal_proximity": round(cardinal_proximity, 4),
        "quantization_hostility": round(quantization_hostility, 4),
    }


def _run_demo(model_id: str) -> dict[str, Any]:
    """
    Demo run: synthetic tensors, numpy only.
    Mimics the layer-by-layer structure of the real run without loading weights.
    Injects mild Gemma-4-like activation signatures (high outlier ratio) so
    the output numbers are illustrative rather than all-zeros.
    """
    import numpy as np

    rng = np.random.default_rng(42)

    num_layers = 35
    hidden_size = 1536
    seq_len = 12

    print(f"  [demo] model_id : {model_id}")
    print(f"  [demo] layers   : {num_layers}  hidden: {hidden_size}")
    print(f"  [demo] prompts  : {len(PROMPTS)} (synthetic tensors, no weights loaded)")

    layer_buckets: list[list[dict]] = [[] for _ in range(num_layers + 1)]

    for pi, _ in enumerate(PROMPTS):
        for li in range(num_layers + 1):
            H = rng.normal(0, 1, (seq_len, hidden_size)).astype(np.float32)
            # Inject a dominant "massive activation" dim to simulate Gemma-4's
            # known high outlier-ratio signature (real: ~83x)
            outlier_dim = rng.integers(0, hidden_size)
            H[:, outlier_dim] *= rng.uniform(40, 100)
            metrics = _demo_outlier_geometry(H)
            layer_buckets[li].append(metrics)

        if (pi + 1) % 5 == 0:
            print(f"    {pi+1}/{len(PROMPTS)} prompts done [demo]")

    print("  Aggregating…")
    layer_summaries: list[dict] = []
    for li, bucket in enumerate(layer_buckets):
        if not bucket:
            continue
        n = len(bucket)
        layer_summaries.append({
            "layer_idx": li,
            "mean_quantization_hostility": round(sum(m["quantization_hostility"] for m in bucket) / n, 4),
            "mean_outlier_ratio":          round(sum(m["outlier_ratio"] for m in bucket) / n, 4),
            "mean_activation_kurtosis":    round(sum(m["activation_kurtosis"] for m in bucket) / n, 4),
            "mean_cardinal_proximity":     round(sum(m["cardinal_proximity"] for m in bucket) / n, 4),
        })

    inner = layer_summaries[1:]
    n_inner = len(inner)
    mean_qh  = sum(l["mean_quantization_hostility"] for l in inner) / n_inner
    mean_or  = sum(l["mean_outlier_ratio"]           for l in inner) / n_inner
    mean_ak  = sum(l["mean_activation_kurtosis"]     for l in inner) / n_inner
    mean_cp  = sum(l["mean_cardinal_proximity"]      for l in inner) / n_inner
    worst = max(inner, key=lambda l: l["mean_quantization_hostility"])["layer_idx"]
    best  = min(inner, key=lambda l: l["mean_quantization_hostility"])["layer_idx"]

    return {
        "model": model_id,
        "architecture": "gemma4",
        "text_architecture": "gemma4_text",
        "num_hidden_layers": num_layers,
        "hidden_size": hidden_size,
        "num_prompts": len(PROMPTS),
        "demo_mode": True,
        "architecture_notes": GEMMA4_ARCH_NOTES,
        "aggregate": {
            "mean_quantization_hostility": round(mean_qh, 4),
            "mean_outlier_ratio":          round(mean_or, 4),
            "mean_activation_kurtosis":    round(mean_ak, 4),
            "mean_cardinal_proximity":     round(mean_cp, 4),
            "worst_layer_idx":             worst,
            "best_layer_idx":              best,
        },
        "layer_summaries": layer_summaries,
        "prompts": PROMPTS,
    }


# ---------------------------------------------------------------------------
# Real run (delegates to experiments/gemma4/run_prism_analysis.py)
# ---------------------------------------------------------------------------

def _run_real(model_id: str, load_in_4bit: bool = False) -> dict[str, Any]:
    """
    Load the harness from experiments/gemma4/run_prism_analysis.py and
    call run_analysis() with model_id as the model path/hub id.

    Args:
        model_id     : HF hub id or local path to model weights.
        load_in_4bit : If True, load via bitsandbytes 4-bit NF4 on CUDA.
                       Required for variants too large to fit in host RAM.

    Requires: torch, transformers >= 5.5.0. Approx VRAM: ~3 GB (E4B 4-bit),
    ~13 GB (26B-A4B 4-bit), ~16 GB (31B 4-bit). Without 4-bit: ~5-8 GB host
    RAM for E2B/E4B; larger variants will OOM the host.
    """
    harness_path = _EXPERIMENTS_GEMMA4 / "run_prism_analysis.py"
    if not harness_path.exists():
        raise FileNotFoundError(f"Harness not found: {harness_path}")

    # Ensure prism geometry is importable
    prism_src = str(_PRISM_SRC)
    if prism_src not in sys.path:
        sys.path.insert(0, prism_src)

    harness = _load_module("prism_gemma4_harness", harness_path)

    ts = _ts_label()
    output_path = str(_EXPERIMENTS_GEMMA4 / f"gemma4_run_{ts}.json")
    agg = harness.run_analysis(
        model_path=model_id,
        output_path=output_path,
        load_in_4bit=load_in_4bit,
    )

    # Re-read the full result that run_analysis() wrote
    result = json.loads(Path(output_path).read_text(encoding="utf-8"))
    return result


# ---------------------------------------------------------------------------
# Gemma4Run dataclass (matches ModelRun conventions from model_runs.py)
# ---------------------------------------------------------------------------

@dataclass
class Gemma4Run:
    """Structured record for a single Gemma 4 PRISM analysis run."""

    model_id: str
    demo: bool = False
    status: str = "pending"
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    aggregate: dict[str, Any] = field(default_factory=dict)
    layer_summaries: list[dict] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    _result_raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at or time.time()
        return max(0.0, end - self.started_at)

    def to_run_dict(self) -> dict[str, Any]:
        """Schema matches model_runs.ModelRun.to_dict() for index compatibility."""
        return {
            "target_name": "Gemma 4",
            "target_path": self.model_id,
            "target_slug": "gemma4",
            "objective": "PRISM outlier-geometry analysis (outlier_ratio, activation_kurtosis, cardinal_proximity, quantization_hostility)",
            "status": self.status,
            "model_family": "gemma4",
            "run_kind": "prism_geometry",
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": round(self.duration_seconds, 3),
            "command_line": f"analyze_gemma4(model_id={self.model_id!r}, demo={self.demo})",
            "demo_mode": self.demo,
            "aggregate": self.aggregate,
            "layer_summaries": self.layer_summaries,
            "findings": self.findings,
            "summary": self.summary,
            "phases": [
                {
                    "name": "outlier_geometry",
                    "description": "Run PRISM outlier_geometry() on each transformer layer across 20 prompts.",
                    "exit_code": 0 if self.status == "success" else 1,
                    "duration_seconds": round(self.duration_seconds, 3),
                    "notes": self.findings,
                    "metrics": self.aggregate,
                    "stdout_excerpt": self.summary,
                    "stderr_excerpt": "",
                }
            ],
            "artifacts": [
                {
                    "label": "Maestro ledger record",
                    "path": str(_MAESTRO_RECORDS),
                    "kind": "directory",
                    "notes": "gemma4_<timestamp>.json",
                }
            ],
            "metadata": self.metadata,
        }

    def to_maestro_dict(self, run_id: str) -> dict[str, Any]:
        """Schema matches existing maestro/data/records/*.json files."""
        now = _iso()
        return {
            "id": run_id,
            "run_id": run_id,
            "base_model_ref": "Gemma 4",
            "run_kind": "prism_geometry",
            "status": self.status,
            "source_axis": "Gemma 4 (google/gemma-4-12b-it)",
            "updated_at": now,
            "created_at": _iso(datetime.fromtimestamp(self.started_at, tz=timezone.utc)),
            "duration_seconds": round(self.duration_seconds, 3),
            "objective": "PRISM outlier-geometry analysis of Gemma 4 transformer layers.",
            "summary": self.summary,
            "findings": self.findings,
            "target_slug": "gemma4",
            "target_path": self.model_id,
            "command_line": f"analyze_gemma4(model_id={self.model_id!r}, demo={self.demo})",
            "phases": [
                {
                    "name": "outlier_geometry",
                    "description": "PRISM outlier_geometry() sweep across all transformer layers.",
                    "duration_seconds": round(self.duration_seconds, 3),
                    "exit_code": 0 if self.status == "success" else 1,
                    "notes": self.findings,
                    "metrics": self.aggregate,
                    "stdout_excerpt": self.summary,
                    "stderr_excerpt": "",
                }
            ],
            "metadata": self.metadata,
            "_source_file": str(_GEMMA4_LOG_ROOT / run_id / "run.json"),
            "_importer": "analyze_gemma4()",
        }


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def analyze_gemma4(
    model_id: str = "google/gemma-4-12b-it",
    config: dict[str, Any] | None = None,
    demo: bool = False,
) -> Gemma4Run:
    """
    Run the full Gemma 4 PRISM analysis.

    Writes:
      - Internal run artifact : D:\\prism\\logs\\model-runs\\gemma4\\<ts>\\run.json
      - Maestro ledger record : D:\\humanai-convention\\maestro\\data\\records\\gemma4_<ts>.json

    Args:
        model_id : HuggingFace hub id or local path to model weights.
        config   : Optional overrides (currently unused, reserved for future).
        demo     : If True, run with synthetic numpy tensors (no GPU/torch needed).

    Returns:
        Gemma4Run dataclass with all results populated.
    """
    cfg = config or {}
    load_in_4bit = bool(cfg.get("load_in_4bit", False))
    ts = _ts_label()
    run_id = f"{ts}_gemma4"

    print(f"\n{'='*60}")
    print(f"  PRISM Gemma 4 Analysis")
    print(f"  run_id  : {run_id}")
    print(f"  model   : {model_id}")
    print(f"  demo    : {demo}")
    print(f"  4-bit   : {load_in_4bit}")
    print(f"{'='*60}", flush=True)

    run = Gemma4Run(model_id=model_id, demo=demo)

    try:
        result = _run_demo(model_id) if demo else _run_real(model_id, load_in_4bit=load_in_4bit)
        run._result_raw = result

        agg = result["aggregate"]
        run.aggregate = agg
        run.layer_summaries = result.get("layer_summaries", [])

        hostility = agg["mean_quantization_hostility"]
        label = "Hostile" if hostility >= 0.7 else ("Moderate" if hostility >= 0.3 else "Friendly")

        run.findings = [
            f"mean_quantization_hostility={hostility:.4f} ({label})",
            f"mean_outlier_ratio={agg['mean_outlier_ratio']:.2f}x — "
            + ("dominant massive-activation signature" if agg["mean_outlier_ratio"] > 10 else "within normal range"),
            f"mean_activation_kurtosis={agg['mean_activation_kurtosis']:.4f} — "
            + ("heavy-tailed per-dim magnitudes" if agg["mean_activation_kurtosis"] > 5 else "near-Gaussian"),
            f"mean_cardinal_proximity={agg['mean_cardinal_proximity']:.4f} — "
            + ("strong axis-alignment → quant-snaps" if agg["mean_cardinal_proximity"] > 0.7 else "moderate alignment"),
            f"worst_layer=L{agg['worst_layer_idx']}  best_layer=L{agg['best_layer_idx']}",
            f"Gemma 4 architecture: {GEMMA4_ARCH_NOTES['sliding_attention_layers']} sliding-window + "
            f"{GEMMA4_ARCH_NOTES['full_attention_layers']} full-attention layers, "
            f"per-layer embeddings (dim={GEMMA4_ARCH_NOTES['per_layer_embedding_dim']})",
        ]
        if demo:
            run.findings.insert(0, "DEMO MODE — synthetic tensors, not real model weights")

        run.summary = (
            f"Gemma 4 PRISM analysis {'(demo)' if demo else ''}completed. "
            f"Hostility={hostility:.4f} ({label}), "
            f"outlier={agg['mean_outlier_ratio']:.2f}x, "
            f"kurtosis={agg['mean_activation_kurtosis']:.2f}, "
            f"cardinal={agg['mean_cardinal_proximity']:.4f}. "
            f"Worst layer: L{agg['worst_layer_idx']}, best: L{agg['best_layer_idx']}."
        )

        run.status = "success"
    except Exception as exc:
        import traceback
        run.status = "error"
        run.summary = f"Analysis failed: {exc}"
        run.findings = [f"ERROR: {exc}", traceback.format_exc()]
        print(f"\n[ERROR] {exc}")
        traceback.print_exc()

    run.finished_at = time.time()

    import platform
    run.metadata = {
        "prism_version": PRISM_VERSION,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "demo_mode": demo,
        "model_id": model_id,
        "architecture_notes": GEMMA4_ARCH_NOTES,
        **(cfg or {}),
    }

    # -- torch metadata if available
    try:
        import torch
        run.metadata["torch_version"] = getattr(torch, "__version__", "")
        run.metadata["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            run.metadata["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        run.metadata["torch_version"] = "not installed"
        run.metadata["cuda_available"] = False

    # -------------------------------------------------------------------------
    # Write internal run artifact
    # -------------------------------------------------------------------------
    run_dir = _GEMMA4_LOG_ROOT / run_id
    run_json_path = run_dir / "run.json"
    try:
        _write_json(run_json_path, run.to_run_dict())
        print(f"\n  [artifact] run.json -> {run_json_path}")
    except Exception as exc:
        print(f"\n  [warn] Could not write run.json: {exc}")

    # -------------------------------------------------------------------------
    # Write Maestro ledger record
    # -------------------------------------------------------------------------
    maestro_path = _MAESTRO_RECORDS / f"gemma4_{ts}.json"
    try:
        _write_json(maestro_path, run.to_maestro_dict(run_id))
        print(f"  [ledger]   maestro -> {maestro_path}")
    except Exception as exc:
        print(f"\n  [warn] Could not write maestro record: {exc}")

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------
    print(f"\n  STATUS  : {run.status.upper()}")
    print(f"  SUMMARY : {run.summary}")
    print(f"\n  AGGREGATE METRICS:")
    for k, v in run.aggregate.items():
        print(f"    {k:<35} {v}")
    if run.status == "success":
        print(f"\n  FINDINGS:")
        for f in run.findings:
            print(f"    • {f}")
    print(f"\n{'='*60}\n")

    return run
