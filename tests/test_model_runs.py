import shutil
from pathlib import Path

from prism.runs.model_runs import ArtifactRef, ModelRun, RunPhase, write_run_bundle


def test_write_run_bundle_creates_markdown_and_index():
    scratch_root = Path(__file__).resolve().parents[1] / "_tmp_model_runs_test"
    if scratch_root.exists():
        shutil.rmtree(scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=True)
    try:
        prism_root = scratch_root / "prism"
        model_root = scratch_root / "model"
        phase = RunPhase(
            name="smoke",
            description="short smoke test",
            command=["python", "-c", "print('ok')"],
            cwd=str(model_root),
            started_at=10.0,
            finished_at=12.5,
            exit_code=0,
            stdout="hello\nworld\n",
            stderr="",
            notes=["smoke passed"],
            metrics={"value": 1},
        )
        run = ModelRun(
            target_name="Test Target",
            target_path=str(model_root),
            objective="Exercise the run bundle writer.",
            status="success",
            model_family="Test Family",
            run_kind="smoke",
            started_at=10.0,
            finished_at=12.5,
            lessons=["Keep the smoke phase separate from follow-on checks."],
            summary="Synthetic summary for the writer test.",
            phases=[phase],
            artifacts=[ArtifactRef(label="README", path=str(model_root / "README.md"), kind="markdown")],
        )

        bundle = write_run_bundle(run, prism_root=prism_root, mirror_root=model_root, run_id="2026-03-24_000000Z_test-target")

        assert bundle.run_json.exists()
        assert bundle.run_md.exists()
        assert (prism_root / "logs" / "model-runs" / "index.json").exists()
        assert (prism_root / "logs" / "model-runs" / "index.md").exists()

        canonical_phase = prism_root / "logs" / "model-runs" / "test-target" / "2026-03-24_000000Z_test-target" / "phases" / "smoke"
        mirror_phase = model_root / "analysis" / "prism" / "model-runs" / "2026-03-24_000000Z_test-target" / "phases" / "smoke"
        assert (canonical_phase / "phase.json").exists()
        assert (canonical_phase / "phase.md").exists()
        assert (canonical_phase / "stdout.txt").exists()
        assert (mirror_phase / "phase.json").exists()
        assert (mirror_phase / "phase.md").exists()
        assert "Synthetic summary" in bundle.run_md.read_text(encoding="utf-8")
        assert "## Lessons" in bundle.run_md.read_text(encoding="utf-8")

        index_payload = (prism_root / "logs" / "model-runs" / "index.json").read_text(encoding="utf-8")
        assert "Keep the smoke phase separate" in index_payload
        index_markdown = (prism_root / "logs" / "model-runs" / "index.md").read_text(encoding="utf-8")
        assert "| Target | Status | Started | Duration | Lesson | Run |" in index_markdown
    finally:
        if scratch_root.exists():
            shutil.rmtree(scratch_root)
