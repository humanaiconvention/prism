"""Structured model-run logging for Prism.

This module keeps the logging schema stable so the same run can be rendered
later in the Console, on the website, or in a local markdown archive.
"""

from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import os
import platform
import re
import sys
import tempfile
import time
import traceback
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from prism import __version__ as PRISM_VERSION


DEFAULT_LOG_ROOT = Path(__file__).parents[3] / "logs" / "model-runs"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp_label(dt: datetime | None = None) -> str:
    dt = dt or _utc_now()
    return dt.strftime("%Y-%m-%d_%H-%M-%SZ")


def _iso_timestamp(dt: datetime | None = None) -> str:
    dt = dt or _utc_now()
    return dt.isoformat().replace("+00:00", "Z")


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-")
    return value or "run"


def _safe_excerpt(text: str, limit: int = 4000) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head - 40
    return text[:head].rstrip() + "\n\n...[truncated]...\n\n" + text[-tail:].lstrip()


def _line_excerpt(text: str, max_lines: int = 40) -> str:
    lines = [line.rstrip() for line in (text or "").splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    head = max_lines // 2
    tail = max_lines - head
    return "\n".join(lines[:head] + ["...[truncated]..."] + lines[-tail:])


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, text: str) -> None:
    _ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _runtime_metadata() -> dict[str, Any]:
    return {
        "prism_version": PRISM_VERSION,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "torch_version": getattr(torch, "__version__", ""),
        "cuda_available": bool(torch.cuda.is_available()),
        "cwd": str(Path.cwd()),
    }


def _docx_paragraphs(path: Path, limit: int = 12) -> list[str]:
    if not path.exists():
        return []
    try:
        with zipfile.ZipFile(path) as zf:
            xml_bytes = zf.read("word/document.xml")
    except Exception:
        return []

    try:
        root = ET.fromstring(xml_bytes)
    except Exception:
        return []

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paras: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        parts: list[str] = []
        for text_node in paragraph.findall(".//w:t", namespace):
            if text_node.text:
                parts.append(text_node.text)
        text = "".join(parts).strip()
        if text:
            paras.append(text)
        if len(paras) >= limit:
            break
    return paras


def _stringify_command(command: Sequence[str] | None) -> str:
    if not command:
        return ""
    return " ".join(command)


def _phase_excerpt(text: str) -> str:
    return _safe_excerpt(_line_excerpt(text, max_lines=60), limit=6000)


@dataclass
class ArtifactRef:
    label: str
    path: str
    kind: str = "file"
    size_bytes: int = 0
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "path": self.path,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "notes": self.notes,
        }


@dataclass
class RunPhase:
    name: str
    description: str = ""
    command: list[str] = field(default_factory=list)
    cwd: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    exit_code: int = 0
    stdout: str = field(default="", repr=False)
    stderr: str = field(default="", repr=False)
    stdout_path: str = ""
    stderr_path: str = ""
    stdout_excerpt: str = ""
    stderr_excerpt: str = ""
    notes: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        return max(0.0, float(self.finished_at) - float(self.started_at))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "command": list(self.command),
            "cwd": self.cwd,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "exit_code": self.exit_code,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
            "stdout_excerpt": self.stdout_excerpt,
            "stderr_excerpt": self.stderr_excerpt,
            "notes": list(self.notes),
            "metrics": self.metrics,
        }

    def to_markdown(self, phase_dir: Path | None = None) -> str:
        lines = [f"### {self.name}"]
        if self.description:
            lines.append(f"- Description: {self.description}")
        if self.command:
            lines.append(f"- Command: `{_stringify_command(self.command)}`")
        if self.cwd:
            lines.append(f"- CWD: `{self.cwd}`")
        lines.append(f"- Exit code: `{self.exit_code}`")
        lines.append(f"- Duration: `{self.duration_seconds:.2f}s`")
        if self.notes:
            lines.append("- Notes:")
            for note in self.notes:
                lines.append(f"  - {note}")
        if self.metrics:
            lines.append("- Metrics:")
            lines.append("```json")
            lines.append(json.dumps(self.metrics, indent=2, ensure_ascii=False))
            lines.append("```")
        if self.stdout_excerpt:
            lines.append("- Stdout excerpt:")
            lines.append("```text")
            lines.append(self.stdout_excerpt)
            lines.append("```")
        if self.stderr_excerpt:
            lines.append("- Stderr excerpt:")
            lines.append("```text")
            lines.append(self.stderr_excerpt)
            lines.append("```")
        if phase_dir is not None:
            lines.append(f"- Phase folder: `{phase_dir}`")
        return "\n".join(lines)


@dataclass
class ModelRun:
    target_name: str
    target_path: str
    objective: str
    status: str = "pending"
    model_family: str = ""
    run_kind: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    command_line: str = ""
    comparison: dict[str, Any] = field(default_factory=dict)
    lessons: list[str] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    summary: str = ""
    phases: list[RunPhase] = field(default_factory=list)
    artifacts: list[ArtifactRef] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        end = self.finished_at if self.finished_at else time.time()
        return max(0.0, float(end) - float(self.started_at))

    @property
    def target_slug(self) -> str:
        return _slugify(self.target_name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_name": self.target_name,
            "target_path": self.target_path,
            "target_slug": self.target_slug,
            "objective": self.objective,
            "status": self.status,
            "model_family": self.model_family,
            "run_kind": self.run_kind,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "command_line": self.command_line,
            "comparison": self.comparison,
            "lessons": list(self.lessons),
            "findings": list(self.findings),
            "summary": self.summary,
            "phases": [phase.to_dict() for phase in self.phases],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "metadata": self.metadata,
        }

    def to_markdown(self) -> str:
        lines = [
            f"# {self.target_name}",
            "",
            f"- Status: `{self.status}`",
            f"- Objective: {self.objective}",
            f"- Model family: `{self.model_family or 'n/a'}`",
            f"- Run kind: `{self.run_kind or 'n/a'}`",
            f"- Target path: `{self.target_path}`",
            f"- Started: `{_iso_timestamp(datetime.fromtimestamp(self.started_at, tz=timezone.utc))}`",
            f"- Finished: `{_iso_timestamp(datetime.fromtimestamp(self.finished_at, tz=timezone.utc)) if self.finished_at else 'n/a'}`",
            f"- Duration: `{self.duration_seconds:.2f}s`",
        ]
        if self.command_line:
            lines.extend(["- Command line:", f"  `{self.command_line}`"])
        if self.summary:
            lines.extend(["", "## Summary", self.summary])
        if self.lessons:
            lines.extend(["", "## Lessons"])
            for lesson in self.lessons:
                lines.append(f"- {lesson}")
        if self.findings:
            lines.extend(["", "## Findings"])
            for finding in self.findings:
                lines.append(f"- {finding}")
        if self.comparison:
            lines.extend(["", "## Comparison"])
            for key, value in self.comparison.items():
                if isinstance(value, list):
                    lines.append(f"- {key}:")
                    for item in value:
                        lines.append(f"  - {item}")
                else:
                    lines.append(f"- {key}: {value}")
        if self.metadata:
            lines.extend(["", "## Metadata", "```json", json.dumps(self.metadata, indent=2, ensure_ascii=False), "```"])
        if self.phases:
            lines.extend(["", "## Phases"])
            for phase in self.phases:
                lines.append(phase.to_markdown())
                lines.append("")
        if self.artifacts:
            lines.extend(["", "## Artifacts"])
            for artifact in self.artifacts:
                note = f" ({artifact.notes})" if artifact.notes else ""
                lines.append(f"- [{artifact.label}]({artifact.path}){note}")
        return "\n".join(lines).rstrip() + "\n"


@dataclass
class RunBundlePaths:
    run_id: str
    canonical_dir: Path
    mirror_dir: Path | None
    run_json: Path
    run_md: Path
    index_json: Path
    index_md: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "canonical_dir": str(self.canonical_dir),
            "mirror_dir": str(self.mirror_dir) if self.mirror_dir else "",
            "run_json": str(self.run_json),
            "run_md": str(self.run_md),
            "index_json": str(self.index_json),
            "index_md": str(self.index_md),
        }


def _write_phase_files(phase_dir: Path, phase: RunPhase, max_inline_text_bytes: int = 262_144) -> None:
    phase_dir.mkdir(parents=True, exist_ok=True)
    phase_json = phase_dir / "phase.json"
    phase_md = phase_dir / "phase.md"
    stdout_path = phase_dir / "stdout.txt"
    stderr_path = phase_dir / "stderr.txt"
    stdout_excerpt_path = phase_dir / "stdout_excerpt.txt"
    stderr_excerpt_path = phase_dir / "stderr_excerpt.txt"

    if phase.stdout:
        _write_text(stdout_path, phase.stdout)
        phase.stdout_path = str(stdout_path)
        if len(phase.stdout.encode("utf-8")) > max_inline_text_bytes:
            _write_text(stdout_excerpt_path, _phase_excerpt(phase.stdout))
        else:
            _write_text(stdout_excerpt_path, phase.stdout)
    else:
        _write_text(stdout_path, "")
        _write_text(stdout_excerpt_path, "")
        phase.stdout_path = str(stdout_path)

    if phase.stderr:
        _write_text(stderr_path, phase.stderr)
        phase.stderr_path = str(stderr_path)
        if len(phase.stderr.encode("utf-8")) > max_inline_text_bytes:
            _write_text(stderr_excerpt_path, _phase_excerpt(phase.stderr))
        else:
            _write_text(stderr_excerpt_path, phase.stderr)
    else:
        _write_text(stderr_path, "")
        _write_text(stderr_excerpt_path, "")
        phase.stderr_path = str(stderr_path)

    phase.stdout_excerpt = _phase_excerpt(phase.stdout)
    phase.stderr_excerpt = _phase_excerpt(phase.stderr)
    _write_json(phase_json, phase.to_dict())
    _write_text(phase_md, phase.to_markdown(phase_dir=phase_dir))


def _write_index(index_path: Path, entries: list[dict[str, Any]]) -> None:
    _write_json(index_path, {"generated_at": _iso_timestamp(), "runs": entries})


def _inline_preview(text: str, limit: int = 120) -> str:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""
    if len(cleaned) > limit:
        cleaned = cleaned[: max(1, limit - 1)].rstrip() + "…"
    return cleaned.replace("|", "\\|")


def _preview_items(items: Sequence[str], max_items: int = 2) -> str:
    values = [str(item).strip() for item in items if str(item).strip()]
    if not values:
        return ""
    preview = values[:max_items]
    if len(values) > max_items:
        preview.append("...")
    return _inline_preview("<br>".join(preview))


def _write_index_markdown(index_path: Path, entries: list[dict[str, Any]]) -> None:
    lines = [
        "# Prism Model Runs",
        "",
        "| Target | Status | Started | Duration | Lesson | Run |",
        "|---|---:|---:|---:|---|---|",
    ]
    for entry in entries:
        run_path = entry.get("run_md", "")
        started = entry.get("started_at", "")
        duration = float(entry.get("duration_seconds", 0.0))
        lesson = _preview_items(entry.get("lessons", []), max_items=2) or _inline_preview(
            str(entry.get("summary", "")),
            limit=120,
        )
        lines.append(
            f"| {entry.get('target_name', '')} | {entry.get('status', '')} | {started} | {duration:.2f}s | {lesson} | {run_path} |"
        )
    _write_text(index_path, "\n".join(lines) + "\n")


def _append_index_entry(index_json: Path, entry: dict[str, Any]) -> list[dict[str, Any]]:
    if index_json.exists():
        try:
            payload = json.loads(index_json.read_text(encoding="utf-8"))
            runs = list(payload.get("runs", []))
        except Exception:
            runs = []
    else:
        runs = []
    runs.append(entry)
    return runs[-200:]


def write_run_bundle(
    record: ModelRun,
    *,
    prism_root: Path,
    mirror_root: Path | None = None,
    run_id: str | None = None,
    max_inline_text_bytes: int = 262_144,
) -> RunBundlePaths:
    run_id = run_id or f"{_timestamp_label()}_{_slugify(record.target_name)}"
    canonical_dir = prism_root / "logs" / "model-runs" / record.target_slug / run_id
    canonical_dir.mkdir(parents=True, exist_ok=True)

    mirror_dir = None
    if mirror_root is not None:
        mirror_dir = mirror_root / "analysis" / "prism" / "model-runs" / run_id
        mirror_dir.mkdir(parents=True, exist_ok=True)

    if not record.finished_at:
        record.finished_at = time.time()

    mirror_record = copy.deepcopy(record) if mirror_dir is not None else None

    for idx, phase in enumerate(record.phases):
        _write_phase_files(canonical_dir / "phases" / _slugify(phase.name), phase, max_inline_text_bytes=max_inline_text_bytes)
        if mirror_record is not None:
            mirror_phase = mirror_record.phases[idx]
            _write_phase_files(mirror_dir / "phases" / _slugify(mirror_phase.name), mirror_phase, max_inline_text_bytes=max_inline_text_bytes)

    run_json = canonical_dir / "run.json"
    run_md = canonical_dir / "run.md"
    _write_json(run_json, record.to_dict())
    _write_text(run_md, record.to_markdown())

    if mirror_dir is not None:
        _write_json(mirror_dir / "run.json", mirror_record.to_dict())
        _write_text(mirror_dir / "run.md", mirror_record.to_markdown())

    index_json = prism_root / "logs" / "model-runs" / "index.json"
    index_md = prism_root / "logs" / "model-runs" / "index.md"
    index_entries = _append_index_entry(
        index_json,
        {
            "run_id": run_id,
            "target_name": record.target_name,
            "target_slug": record.target_slug,
            "status": record.status,
            "started_at": _iso_timestamp(datetime.fromtimestamp(record.started_at, tz=timezone.utc)),
            "finished_at": _iso_timestamp(datetime.fromtimestamp(record.finished_at, tz=timezone.utc)),
            "duration_seconds": round(record.duration_seconds, 2),
            "run_json": str(run_json),
            "run_md": str(run_md),
            "summary": record.summary,
            "lessons": list(record.lessons[:5]),
            "findings": record.findings[:3],
        },
    )
    _write_index(index_json, index_entries)
    _write_index_markdown(index_md, index_entries[::-1])

    if mirror_dir is not None:
        _write_json(mirror_dir / "index.json", {"generated_at": _iso_timestamp(), "runs": [mirror_record.to_dict()]})
        _write_text(mirror_dir / "index.md", mirror_record.to_markdown())

    return RunBundlePaths(
        run_id=run_id,
        canonical_dir=canonical_dir,
        mirror_dir=mirror_dir,
        run_json=run_json,
        run_md=run_md,
        index_json=index_json,
        index_md=index_md,
    )


def _capture_stdout_stderr(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, str, str]:
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
        result = func(*args, **kwargs)
    return result, stdout_buf.getvalue(), stderr_buf.getvalue()


def _build_phase(
    *,
    name: str,
    description: str,
    command: Sequence[str],
    cwd: Path,
    started_at: float,
    finished_at: float,
    exit_code: int,
    stdout: str,
    stderr: str,
    notes: list[str] | None = None,
    metrics: dict[str, Any] | None = None,
) -> RunPhase:
    return RunPhase(
        name=name,
        description=description,
        command=list(command),
        cwd=str(cwd),
        started_at=started_at,
        finished_at=finished_at,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        notes=notes or [],
        metrics=metrics or {},
    )


def _count_model_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _collect_genesis_reference_excerpt(genesis_docs: Path) -> dict[str, Any]:
    primary = genesis_docs / "Genesis-152M Structural Characterization v2.docx"
    secondary = genesis_docs / "Genesis-152M-Structural-Characterization-3-9-26.docx"
    excerpts: dict[str, Any] = {}
    for doc in [primary, secondary]:
        excerpts[str(doc)] = _docx_paragraphs(doc)[:8]
    return excerpts


def _genesis_module(genesis_root: Path):
    return _load_module_from_path("prism_external_genesis_validate", genesis_root / "validate_genesis.py")


def _genesis_audit_module(genesis_root: Path):
    return _load_module_from_path("prism_external_genesis_audit", genesis_root / "genesis" / "structure_audit.py")


def analyze_genesis_v3(
    genesis_root: Path,
    *,
    prism_root: Path = DEFAULT_LOG_ROOT.parents[1],
    batch_size: int = 8,
    block_size: int = 64,
    max_iters: int = 50,
    bench_steps: int = 2,
    reference_docs: Path | None = None,
) -> ModelRun:
    """Run the bounded Genesis v3 analysis and return a structured run record."""

    started = time.time()
    genesis_root = genesis_root.resolve()
    prism_root = prism_root.resolve()
    docs_root = reference_docs or genesis_root.parent.parent / "docs"
    audit_mod = _genesis_audit_module(genesis_root)
    validate_mod = _genesis_module(genesis_root)

    run = ModelRun(
        target_name="Genesis v3",
        target_path=str(genesis_root),
        objective="Bounded validation and structural comparison against the v2 reference notes.",
        status="running",
        model_family="Genesis",
        run_kind="validation + structure audit",
        started_at=started,
        command_line=(
            f"structure_audit -> validate_genesis.py --backend torch --max-iters {max_iters} "
            f"--batch-size {batch_size} --block-size {block_size} --bench-steps {bench_steps}"
        ),
        metadata={
            **_runtime_metadata(),
            "genesis_root": str(genesis_root),
            "docs_root": str(docs_root),
            "batch_size": batch_size,
            "block_size": block_size,
            "max_iters": max_iters,
            "bench_steps": bench_steps,
            "reference_excerpt": _collect_genesis_reference_excerpt(docs_root),
        },
    )

    phase_start = time.time()
    try:
        files, violations = audit_mod.audit_genesis(genesis_root / "genesis")
        stdout = audit_mod._format_report(files, violations)
        audit_phase = _build_phase(
            name="structure_audit",
            description="Counts structurally risky files and reports the largest Genesis Python modules.",
            command=[str(genesis_root / "genesis" / "structure_audit.py")],
            cwd=genesis_root / "genesis",
            started_at=phase_start,
            finished_at=time.time(),
            exit_code=0,
            stdout=stdout,
            stderr="",
            notes=[
                f"Audited {len(files)} Python files in the Genesis package.",
                f"Violations: {len(violations)}",
            ],
            metrics={
                "file_count": len(files),
                "violation_count": len(violations),
                "top_hotspots": [
                    {
                        "path": str(item.relpath),
                        "countable_lines": item.countable_lines,
                        "dir_depth": item.dir_depth,
                    }
                    for item in sorted(files, key=lambda x: (-x.countable_lines, str(x.relpath)))[:8]
                ],
                "violations": violations[:20],
            },
        )
    except Exception as exc:
        audit_phase = _build_phase(
            name="structure_audit",
            description="Counts structurally risky files and reports the largest Genesis Python modules.",
            command=[str(genesis_root / "genesis" / "structure_audit.py")],
            cwd=genesis_root / "genesis",
            started_at=phase_start,
            finished_at=time.time(),
            exit_code=1,
            stdout="",
            stderr="".join(traceback.format_exception(exc)),
            notes=[f"Structure audit failed: {exc}"],
            metrics={},
        )
        run.status = "partial"
    run.phases.append(audit_phase)

    phase_start = time.time()
    try:
        config = validate_mod.Config()
        config.batch_size = batch_size
        config.block_size = block_size
        config.max_iters = max_iters
        config.eval_interval = max(10, min(config.eval_interval, max_iters))
        train_loader, val_loader, vocab = validate_mod.load_data(config)
        config.vocab_size = vocab

        flags = {
            "hopmoe": True,
            "mod": True,
            "mom": True,
            "ms": True,
            "ortho": True,
            "variance": True,
            "entropy": False,
            "expert_proj": True,
            "baldwin": True,
            "adaptive": True,
            "n_experts": 8,
            "prune": True,
        }

        tempdir = tempfile.TemporaryDirectory(prefix="prism-genesis-")
        original_cwd = Path.cwd()
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        result: dict[str, Any] | None = None
        try:
            os.chdir(tempdir.name)
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
                result = validate_mod.run_exp(
                    "Baseline_MoC_v2.9a",
                    config,
                    train_loader,
                    val_loader,
                    **flags,
                )
        finally:
            os.chdir(original_cwd)
            tempdir.cleanup()

        result = result or {}
        phase_notes = [
            f"Validation finished with val_loss={float(result.get('val_loss', 0.0)):.4f} and ppl={float(result.get('ppl', 0.0)):.2f}.",
            f"Train loss summary: {float(result.get('train_loss', 0.0)):.4f}.",
            f"Feature delta: {float(result.get('delta', 0.0)):+.1f}%.",
        ]
        if result.get("techs"):
            techs = result["techs"]
            phase_notes.append(f"Tech stack: {', '.join(techs) if isinstance(techs, list) else techs}.")

        validate_phase = _build_phase(
            name="bounded_validation",
            description="Runs the current v3 Genesis validation loop with the existing HopMoE/prune path.",
            command=[
                "validate_genesis.run_exp",
                f"max_iters={max_iters}",
                f"batch_size={batch_size}",
                f"block_size={block_size}",
            ],
            cwd=genesis_root,
            started_at=phase_start,
            finished_at=time.time(),
            exit_code=0,
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            notes=phase_notes,
            metrics={
                "val_loss": result.get("val_loss"),
                "ppl": result.get("ppl"),
                "train_loss": result.get("train_loss"),
                "delta_percent": result.get("delta"),
                "techs": result.get("techs", []),
                "checkpoint": result.get("checkpoint", ""),
            },
        )
    except Exception as exc:
        validate_phase = _build_phase(
            name="bounded_validation",
            description="Runs the current v3 Genesis validation loop with the existing HopMoE/prune path.",
            command=[
                "validate_genesis.run_exp",
                f"max_iters={max_iters}",
                f"batch_size={batch_size}",
                f"block_size={block_size}",
            ],
            cwd=genesis_root,
            started_at=phase_start,
            finished_at=time.time(),
            exit_code=1,
            stdout="",
            stderr="".join(traceback.format_exception(exc)),
            notes=[f"Validation run failed: {exc}"],
            metrics={},
        )
        run.status = "partial"

        fallback_phase = None
        try:
            fallback_phase_start = time.time()
            fallback_config = validate_mod.Config()
            fallback_config.batch_size = batch_size
            fallback_config.block_size = block_size
            fallback_config.max_iters = max_iters
            fallback_config.eval_interval = max(10, min(fallback_config.eval_interval, max_iters))
            fallback_train_loader, fallback_val_loader, fallback_vocab = validate_mod.load_data(fallback_config)
            fallback_config.vocab_size = fallback_vocab

            fallback_flags = dict(flags)
            fallback_flags["use_triton"] = False

            fallback_tempdir = tempfile.TemporaryDirectory(prefix="prism-genesis-fallback-")
            original_cwd = Path.cwd()
            fallback_stdout = io.StringIO()
            fallback_stderr = io.StringIO()
            fallback_result: dict[str, Any] | None = None
            try:
                os.chdir(fallback_tempdir.name)
                with contextlib.redirect_stdout(fallback_stdout), contextlib.redirect_stderr(fallback_stderr):
                    fallback_result = validate_mod.run_exp(
                        "Baseline_MoC_v2.9a",
                        fallback_config,
                        fallback_train_loader,
                        fallback_val_loader,
                        **fallback_flags,
                    )
            finally:
                os.chdir(original_cwd)
                fallback_tempdir.cleanup()

            fallback_result = fallback_result or {}
            fallback_phase = _build_phase(
                name="bounded_validation_no_triton",
                description="Fallback validation path with Triton disabled so the Genesis run can still complete on Windows.",
                command=[
                    "validate_genesis.run_exp",
                    f"max_iters={max_iters}",
                    f"batch_size={batch_size}",
                    f"block_size={block_size}",
                    "use_triton=False",
                ],
                cwd=genesis_root,
                started_at=fallback_phase_start,
                finished_at=time.time(),
                exit_code=0,
                stdout=fallback_stdout.getvalue(),
                stderr=fallback_stderr.getvalue(),
                notes=[
                    f"Fallback completed with val_loss={float(fallback_result.get('val_loss', 0.0)):.4f} and ppl={float(fallback_result.get('ppl', 0.0)):.2f}.",
                    f"Fallback train loss summary: {float(fallback_result.get('train_loss', 0.0)):.4f}.",
                ],
                metrics={
                    "val_loss": fallback_result.get("val_loss"),
                    "ppl": fallback_result.get("ppl"),
                    "train_loss": fallback_result.get("train_loss"),
                    "delta_percent": fallback_result.get("delta"),
                    "techs": fallback_result.get("techs", []),
                    "checkpoint": fallback_result.get("checkpoint", ""),
                },
            )
        except Exception as fallback_exc:
            fallback_phase = _build_phase(
                name="bounded_validation_no_triton",
                description="Fallback validation path with Triton disabled so the Genesis run can still complete on Windows.",
                command=[
                    "validate_genesis.run_exp",
                    f"max_iters={max_iters}",
                    f"batch_size={batch_size}",
                    f"block_size={block_size}",
                    "use_triton=False",
                ],
                cwd=genesis_root,
                started_at=fallback_phase_start,
                finished_at=time.time(),
                exit_code=1,
                stdout="",
                stderr="".join(traceback.format_exception(fallback_exc)),
                notes=[f"Fallback validation failed: {fallback_exc}"],
                metrics={},
            )
            run.status = "partial"

        run.phases.append(validate_phase)
        if fallback_phase is not None:
            run.phases.append(fallback_phase)
        run.finished_at = time.time()
        run.status = "success" if all(phase.exit_code == 0 for phase in run.phases) else "partial"
        run.summary = (
            "Genesis v3 completed a bounded structural audit and a validation attempt, plus a Windows-safe Triton-disabled fallback when the primary path hit a compiler dependency."
        )
        run.lessons = [
            "Keep the structural audit ahead of validation so oversized Genesis modules are visible before we spend time on a failing run.",
            "On Windows, Triton-backed Genesis validation should call out the C compiler dependency up front so the log clearly explains why the primary path is partial.",
            "Preserve the Triton-disabled fallback because it provides a safer comparison signal for later website display and Console review.",
        ]
        run.findings = [
            "The structural audit remains useful for locating oversized or deeply nested Genesis files before another validation pass.",
            "The primary validation path exposed an environment dependency: the Triton-backed path still wants a C compiler on this machine.",
            "The fallback no-Triton path gives us a second, safer signal for later comparison and website display.",
        ]
        run.comparison = {
            "reference_documents": list((_collect_genesis_reference_excerpt(docs_root)).keys()),
            "comparison_notes": [
                "The v2 material still frames the structural story: stable/reproducible findings, preliminary intervention results, and residual-manifold compression.",
                "The current v3 sweep now documents both the failure surface and a fallback surface, which is more useful for later display than a single exception trace.",
                "The same large hotspots remain the main code-shaping signal to watch in follow-up passes.",
            ],
        }
        run.artifacts = [
            ArtifactRef(
                label="Genesis v2 reference excerpt",
                path=str(docs_root / "Genesis-152M Structural Characterization v2.docx"),
                kind="docx",
                notes="Primary comparison material for the v3 pass.",
            ),
            ArtifactRef(
                label="Genesis March 9 note",
                path=str(docs_root / "Genesis-152M-Structural-Characterization-3-9-26.docx"),
                kind="docx",
                notes="Secondary contextual note that was discussed alongside the v2 material.",
            ),
        ]
        return run

    run.phases.append(validate_phase)

    run.finished_at = time.time()
    run.status = "success" if all(phase.exit_code == 0 for phase in run.phases) else "partial"
    run.summary = (
        "Genesis v3 completed a bounded structural audit and validation sweep. "
        "The log captures the current training path, the major package hotspots, and the comparison basis from the v2 materials."
    )
    run.lessons = [
        "Keep the v2 reference docs attached to Genesis runs because they provide the stable comparison frame for judging whether v3 is actually moving the structure forward.",
        "Use the structural audit output as the first triage pass; the same large hotspots are still the most actionable follow-up targets.",
        "Treat a Windows Triton compiler failure as an environment lesson, not a model failure, so Prism can surface the deployment constraint cleanly later.",
    ]
    run.findings = [
        "The structural audit remains useful for locating oversized or deeply nested Genesis files before another validation pass.",
        "The bounded validation path now produces a stable, reproducible log surface for later website display.",
        "The v2 reference notes remain the right comparison basis for judging whether the v3 behavior still matches the earlier external characterization.",
    ]
    run.comparison = {
        "reference_documents": list((_collect_genesis_reference_excerpt(docs_root)).keys()),
        "comparison_notes": [
            "The v2 material still frames the structural story: stable/reproducible findings, preliminary intervention results, and residual-manifold compression.",
            "The current v3 sweep adds a bounded runtime trace and package audit so the same comparison can be replayed later in Prism or on the website.",
            "The same large hotspots remain the main code-shaping signal to watch in follow-up passes.",
        ],
    }
    run.artifacts = [
        ArtifactRef(
            label="Genesis v2 reference excerpt",
            path=str(docs_root / "Genesis-152M Structural Characterization v2.docx"),
            kind="docx",
            notes="Primary comparison material for the v3 pass.",
        ),
        ArtifactRef(
            label="Genesis March 9 note",
            path=str(docs_root / "Genesis-152M-Structural-Characterization-3-9-26.docx"),
            kind="docx",
            notes="Secondary contextual note that was discussed alongside the v2 material.",
        ),
    ]
    return run


def _physics_demo_module(trm_root: Path):
    if str(trm_root) not in sys.path:
        sys.path.insert(0, str(trm_root))
    return _load_module_from_path("prism_external_trm_demo", trm_root / "physics_validation_demo.py")


def _trm_model_module(trm_root: Path):
    if str(trm_root) not in sys.path:
        sys.path.insert(0, str(trm_root))
    return _load_module_from_path("prism_external_trm_model", trm_root / "models" / "recursive_reasoning" / "trm.py")


def analyze_trm_physics_validation(
    trm_root: Path,
    *,
    prism_root: Path = DEFAULT_LOG_ROOT.parents[1],
    physics_epochs: int = 5,
    physics_examples: int = 18,
    hidden_size: int = 64,
    arch_seq_len: int = 8,
) -> ModelRun:
    """Run a lightweight TRM analysis pass and return a structured run record."""

    started = time.time()
    trm_root = trm_root.resolve()
    prism_root = prism_root.resolve()
    demo_mod = _physics_demo_module(trm_root)
    model_mod = _trm_model_module(trm_root)

    run = ModelRun(
        target_name="TRM",
        target_path=str(trm_root),
        objective="Run a bounded TRM architecture smoke test and a small physics-validation sweep.",
        status="running",
        model_family="Tiny Recursive Model",
        run_kind="architecture smoke + physics validation smoke",
        started_at=started,
        command_line=(
            f"arch_smoke -> PhysicsValidationTRM(hidden_size={hidden_size}) -> "
            f"train_physics_validator(epochs={physics_epochs}, examples={physics_examples})"
        ),
        metadata={
            **_runtime_metadata(),
            "trm_root": str(trm_root),
            "physics_epochs": physics_epochs,
            "physics_examples": physics_examples,
            "hidden_size": hidden_size,
            "arch_seq_len": arch_seq_len,
            "reference_documents": [
                str(trm_root / "README.md"),
                str(trm_root / "DEMO_RESULTS.md"),
                str(trm_root / "SESSION_STATE.md"),
            ],
        },
    )

    phase_start = time.time()
    arch_result: dict[str, Any] = {}
    arch_stdout = io.StringIO()
    arch_stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(arch_stdout), contextlib.redirect_stderr(arch_stderr):
            core_model = model_mod.TinyRecursiveReasoningModel_ACTV1(
                {
                    "batch_size": 2,
                    "seq_len": arch_seq_len,
                    "puzzle_emb_ndim": 0,
                    "num_puzzle_identifiers": 1,
                    "vocab_size": 32,
                    "H_cycles": 2,
                    "L_cycles": 2,
                    "H_layers": 1,
                    "L_layers": 1,
                    "hidden_size": hidden_size,
                    "expansion": 2.0,
                    "num_heads": 4,
                    "pos_encodings": "rope",
                    "halt_max_steps": 2,
                    "halt_exploration_prob": 0.0,
                    "forward_dtype": "float32",
                    "mlp_t": False,
                    "puzzle_emb_len": 0,
                    "no_ACT_continue": True,
                }
            )
            batch = {
                "inputs": torch.randint(0, 32, (2, arch_seq_len), dtype=torch.long),
                "puzzle_identifiers": torch.zeros(2, dtype=torch.long),
            }
            carry = core_model.initial_carry(batch)
            next_carry, outputs = core_model(carry, batch)
            arch_result = {
                "parameter_count": _count_model_params(core_model),
                "logits_shape": list(outputs["logits"].shape),
                "q_halt_shape": list(outputs["q_halt_logits"].shape),
                "q_continue_shape": list(outputs["q_continue_logits"].shape),
                "halted_after_step": int(next_carry.halted.sum().item()),
            }
            print("TRM architecture smoke test passed.")
            print(f"Parameters: {arch_result['parameter_count']:,}")
            print(f"Logits shape: {arch_result['logits_shape']}")
            print(f"Q-halt shape: {arch_result['q_halt_shape']}")
            print(f"Q-continue shape: {arch_result['q_continue_shape']}")
    except Exception as exc:
        arch_phase = _build_phase(
            name="architecture_smoke",
            description="Instantiate the raw TRM model and run a single forward pass.",
            command=["TinyRecursiveReasoningModel_ACTV1 smoke", f"hidden_size={hidden_size}", f"seq_len={arch_seq_len}"],
            cwd=trm_root,
            started_at=phase_start,
            finished_at=time.time(),
            exit_code=1,
            stdout=arch_stdout.getvalue(),
            stderr=arch_stderr.getvalue() + "".join(traceback.format_exception(exc)),
            notes=[f"Architecture smoke failed: {exc}"],
            metrics={},
        )
        run.phases.append(arch_phase)
        run.status = "partial"
        run.finished_at = time.time()
        run.summary = "TRM architecture smoke test failed before the physics validation sweep could run."
        run.lessons = [
            "Architecture smoke should stay as its own phase so Prism can report the model-shape failure separately from any downstream physics work.",
            "When the raw model cannot step cleanly, the bundle should still keep the failure trace so the Console can explain the break before future retries.",
        ]
        run.findings = [f"Architecture smoke failed: {exc}"]
        return run

    arch_phase = _build_phase(
        name="architecture_smoke",
        description="Instantiate the raw TRM model and run a single forward pass.",
        command=["TinyRecursiveReasoningModel_ACTV1 smoke", f"hidden_size={hidden_size}", f"seq_len={arch_seq_len}"],
        cwd=trm_root,
        started_at=phase_start,
        finished_at=time.time(),
        exit_code=0,
        stdout=arch_stdout.getvalue(),
        stderr=arch_stderr.getvalue(),
        notes=[
            f"Parameters: {arch_result['parameter_count']:,}",
            f"Output logits shape: {arch_result['logits_shape']}",
            f"Halted sequences after step: {arch_result['halted_after_step']}",
        ],
        metrics=arch_result,
    )
    run.phases.append(arch_phase)

    phase_start = time.time()
    physics_stdout = io.StringIO()
    physics_stderr = io.StringIO()
    physics_metrics: dict[str, Any] = {}
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = demo_mod.PhysicsTokenizer()
        dimensional = demo_mod.generate_dimensional_analysis_puzzles(max(1, physics_examples // 2))
        conservation = demo_mod.generate_conservation_law_puzzles(max(1, physics_examples // 3))
        puzzles = dimensional + conservation
        train_data = demo_mod.create_dataset(puzzles, tokenizer)
        test_puzzles = demo_mod.generate_geometric_dark_energy_tests()
        model = demo_mod.PhysicsValidationTRM(
            vocab_size=tokenizer.vocab_size,
            hidden_size=hidden_size,
            n_layers=2,
            n_cycles=4,
            max_seq_len=128,
        ).to(device)
        with contextlib.redirect_stdout(physics_stdout), contextlib.redirect_stderr(physics_stderr):
            model = demo_mod.train_physics_validator(
                model=model,
                train_data=train_data,
                n_epochs=physics_epochs,
                batch_size=min(8, max(1, len(train_data))),
                lr=1e-3,
                device=device,
            )

        results: list[dict[str, Any]] = []
        model.eval()
        validity_correct = 0
        error_type_correct = 0
        confidence_values: list[float] = []
        error_type_names = ["None", "Dimensional", "Conservation", "Symmetry", "Tensor"]
        with torch.no_grad():
            for puzzle in test_puzzles:
                tokens = tokenizer.encode(puzzle.equation).unsqueeze(0).to(device)
                validity_logits, error_type_logits, confidence = model(tokens)
                pred_validity = int(validity_logits.argmax(dim=1).item())
                pred_error_type = int(error_type_logits.argmax(dim=1).item())
                conf = float(confidence.item())
                confidence_values.append(conf)
                if pred_validity == (1 if puzzle.is_valid else 0):
                    validity_correct += 1
                true_error_type = {
                    None: 0,
                    "dimensional": 1,
                    "conservation": 2,
                    "symmetry": 3,
                    "tensor": 4,
                }.get(puzzle.error_type, 0)
                if pred_error_type == true_error_type:
                    error_type_correct += 1
                results.append(
                    {
                        "equation": puzzle.equation,
                        "true_valid": bool(puzzle.is_valid),
                        "pred_valid": bool(pred_validity == 1),
                        "pred_error_type": error_type_names[pred_error_type],
                        "true_error_type": error_type_names[true_error_type],
                        "confidence": conf,
                        "status": "correct" if (pred_validity == (1 if puzzle.is_valid else 0)) else "wrong",
                    }
                )

        physics_metrics = {
            "device": device,
            "training_examples": len(train_data),
            "test_examples": len(test_puzzles),
            "validity_accuracy": validity_correct / max(1, len(test_puzzles)),
            "error_type_accuracy": error_type_correct / max(1, len(test_puzzles)),
            "mean_confidence": sum(confidence_values) / max(1, len(confidence_values)),
            "results": results,
        }
        print("Physics validation smoke completed.")
        print(f"Device: {device}")
        print(f"Training examples: {len(train_data)}")
        print(f"Test examples: {len(test_puzzles)}")
        print(f"Validity accuracy: {physics_metrics['validity_accuracy']:.3f}")
        print(f"Error-type accuracy: {physics_metrics['error_type_accuracy']:.3f}")
        print(f"Mean confidence: {physics_metrics['mean_confidence']:.3f}")
    except Exception as exc:
        physics_phase = _build_phase(
            name="physics_validation_smoke",
            description="Train the TRM physics validator on a small dataset and evaluate it on geometric DE tests.",
            command=[
                "PhysicsValidationTRM",
                f"epochs={physics_epochs}",
                f"examples={physics_examples}",
                f"hidden_size={hidden_size}",
            ],
            cwd=trm_root,
            started_at=phase_start,
            finished_at=time.time(),
            exit_code=1,
            stdout=physics_stdout.getvalue(),
            stderr=physics_stderr.getvalue() + "".join(traceback.format_exception(exc)),
            notes=[f"Physics validation smoke failed: {exc}"],
            metrics={},
        )
        run.phases.append(physics_phase)
        run.status = "partial"
        run.finished_at = time.time()
        run.summary = "TRM architecture smoke passed, but the physics validation sweep failed."
        run.lessons = [
            "Keep the architecture smoke result even when physics validation fails so Prism can still show the model instantiated cleanly.",
            "A failed physics sweep should be logged with the same structured bundle shape as a successful run so later comparison remains possible.",
        ]
        run.findings = [f"Physics validation smoke failed: {exc}"]
        run.comparison = {
            "reference_documents": [
                str(trm_root / "README.md"),
                str(trm_root / "DEMO_RESULTS.md"),
                str(trm_root / "SESSION_STATE.md"),
            ],
        }
        return run

    physics_phase = _build_phase(
        name="physics_validation_smoke",
        description="Train the TRM physics validator on a small dataset and evaluate it on geometric DE tests.",
        command=[
            "PhysicsValidationTRM",
            f"epochs={physics_epochs}",
            f"examples={physics_examples}",
            f"hidden_size={hidden_size}",
        ],
        cwd=trm_root,
        started_at=phase_start,
        finished_at=time.time(),
        exit_code=0,
        stdout=physics_stdout.getvalue(),
        stderr=physics_stderr.getvalue(),
        notes=[
            f"Training examples: {physics_metrics['training_examples']}",
            f"Test examples: {physics_metrics['test_examples']}",
            f"Validity accuracy: {physics_metrics['validity_accuracy']:.3f}",
            f"Error-type accuracy: {physics_metrics['error_type_accuracy']:.3f}",
            f"Mean confidence: {physics_metrics['mean_confidence']:.3f}",
        ],
        metrics=physics_metrics,
    )
    run.phases.append(physics_phase)

    run.finished_at = time.time()
    run.status = "success" if all(phase.exit_code == 0 for phase in run.phases) else "partial"
    run.summary = (
        "TRM now has a compact architecture smoke pass plus a bounded physics-validation sweep. "
        "The log is structured so the same run can later be surfaced in the Console or on the website."
    )
    run.lessons = [
        "Keep architecture smoke separate from physics validation because the two phases answer different questions and should be displayed independently in Prism.",
        "The current physics validator behaves conservatively on the geometric-DE examples, so Prism should present it as a screening signal rather than a proof engine.",
        "Preserve the per-equation outcomes in the bundle because they are the most website-friendly evidence surface for the TRM example.",
    ]
    run.findings = [
        "The raw TRM model can be instantiated and stepped through a forward pass with a small config.",
        "The physics-validation wrapper produces a compact, human-readable example of how Prism can log and compare model behavior.",
        "The geometric dark energy test set remains a useful website demo surface because each equation outcome can be summarized alongside confidence.",
    ]
    run.comparison = {
        "reference_documents": [
            str(trm_root / "README.md"),
            str(trm_root / "DEMO_RESULTS.md"),
            str(trm_root / "SESSION_STATE.md"),
        ],
        "comparison_notes": [
            "The TRM demo now has a lighter, repeatable smoke lane rather than only the original long-form proof-of-concept.",
            "The resulting output is structured enough to drive a future Prism display console without changing the experiment itself.",
        ],
    }
    run.artifacts = [
        ArtifactRef(
            label="TRM README",
            path=str(trm_root / "README.md"),
            kind="markdown",
            notes="Project overview and canonical setup notes.",
        ),
        ArtifactRef(
            label="TRM demo results",
            path=str(trm_root / "DEMO_RESULTS.md"),
            kind="markdown",
            notes="Prior physics-validation narrative used as context for the smoke run.",
        ),
    ]
    return run
