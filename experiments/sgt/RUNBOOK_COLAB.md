# SGT Colab Runbook

Operational notes for running `colab_runner.ipynb` on Colab Pro. Written from a real session (2026-05-03) so the rough edges are recorded, not just the happy path.

---

## 0. One-time prep

- **Colab Pro** subscription active on the Google account whose Drive will hold the run JSONs.
- `humanaiconvention/prism` (the public OS repo) is up-to-date on `main` — the notebook clones from there. If you've changed `experiments/sgt/sgt_runner.py` in the monorepo, sync to `D:\prism` and push before the Colab session.
- Decide whether you need Drive output. If yes, mount it (cell 1). If you only need the JSON for analyzer round-trip, **skip Drive** — `/content/drive/MyDrive/...` still works as a local directory on the runtime VM, and you can `cat` it back at the end.

---

## 1. Open the notebook

Bookmark this URL pattern — it skips the "save a copy to Drive" friction:

```
https://colab.research.google.com/github/humanaiconvention/prism/blob/main/experiments/sgt/notebooks/colab_runner.ipynb
```

Loads the notebook in playground mode. Cells run; output is ephemeral; no Drive copy is created. Good for one-shot smokes.

---

## 2. Pick the runtime

`Connect ▼` (top-right) → `Change runtime type`. Options that matter for SGT:

| Hardware | Use for | Notes |
|---|---|---|
| **L4 GPU** | 0.5B smoke + 0.5B main sweep + 0.5B correction sweep | High-RAM is auto-bundled. ~15 min/run, ~4 GB VRAM peak. |
| **A100 GPU** | 1.5B if Kaggle is unavailable | 40 GB VRAM, ~3× faster than L4. Burns Colab compute units quickly. |
| T4 GPU | Avoid for SGT | Slower than L4, smaller VRAM. Kaggle's free T4×2 dual-GPU is preferable for 1.5B. |
| H100 / A100 | Save for 1.5B fallback | Compute-unit cost is high; not justified for 0.5B. |

Click `Save` then click `Connect` (top-right). Wait ~10-15s. Status flips from `Connecting…` to `✓ Connected` and bottom-right shows `L4 (Python 3)`.

---

## 3. Run cells in order

### Cell 1 — Drive mount (`drive.mount('/content/drive')`)

**Skip for one-shot smokes.** Mounting Drive triggers an OAuth popup ("Permit this notebook to access your Google Drive files?") that has to be clicked through; not worth the ceremony for a 15-min smoke. The smoke output goes to `/content/drive/MyDrive/sgt_runs/0p5b_smoke/` whether Drive is mounted or not — without the mount, that's just a local dir on the runtime VM.

For the multi-session main sweep, **do mount** so you don't lose runs when the runtime expires.

### Cell 2 — Clone + install (~2-3 min)

When you hit Run, Colab shows "Warning: This notebook was not authored by Google". This is normal for any GitHub-sourced notebook. Click **Run anyway**.

Two benign warnings appear after install:

- `gcsfs requires fsspec==... but you have ...` — gcsfs isn't used by SGT; ignore.
- `torchaudio 2.10.0+cu128 requires torch==2.10.0, but you have torch 2.5.1+cu124 which is incompatible.` AND `torchvision 0.25.0+cu128 requires torch==2.10.0...` — **this one bites.** Read on.

### **CRITICAL FIX — torchvision / torchaudio ABI mismatch**

Colab's base image preinstalls `torch 2.10.0+cu128` plus matching `torchvision 0.25.0+cu128` and `torchaudio 2.10.0+cu128`. Cell 2 forces `torch==2.5.1+cu124` (per the locked stack), but pip's resolver does *not* downgrade torchvision/torchaudio. Result: the cu128 C++ extensions in those packages reference symbols (`torchvision::nms`) that don't exist in cu124-built torch, and the *next* import of `transformers.modeling_utils` raises:

```
RuntimeError: Failed to import transformers.modeling_utils because of the following error
operator torchvision::nms does not exist
```

**Fix**: open the Terminal (bottom-left `⊟ Terminal` button) and run:

```bash
pip uninstall -q -y torchvision torchaudio
```

SGT's runner doesn't import torchvision or torchaudio, and `transformers` falls back gracefully when they're absent. This unblocks `from transformers import AutoModelForCausalLM`. *Alternatively*, install matching versions:

```bash
pip install -q torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```

The notebook's cell 2 should be patched to do this automatically; see "Notebook patches" at the bottom.

### Cell 3 — Set paths (instant)

Just defines `RUNS_DIR`, `MODEL`, `TEACHER`. Runs in <0.1s. The `os.makedirs(RUNS_DIR, exist_ok=True)` will create the directory on the local VM if Drive isn't mounted — that's fine.

### Cell 5 — Smoke (~10-15 min on L4)

`R1, seed=11, generations=2, samples_per_gen=200`. After the model and dataset downloads (~30s), expect:

1. `[gen 0] ppl=… acc=…` — baseline eval. **For Qwen2.5-0.5B, ACC ≈ 0.025** (free-form generation eval against an un-fine-tuned base; below the pre-reg's 0.20 floor → seed will be floor-censored for H1/H4 accuracy).
2. Synthetic generation (200 samples × 128 tokens, ~2 min on L4)
3. Fine-tune (75 steps, ~1 min)
4. `[gen 1] ppl=… acc=…`
5. Repeat for gen 2
6. `wrote /content/drive/MyDrive/sgt_runs/0p5b_smoke/R1_11.json (≈Ns)`

If the smoke finishes with that final `wrote` line, the pipeline is green.

### Cells 6–7 — Main sweep + correction sweep (multi-hour)

Don't run these until smoke is green. Each `subprocess.run(...)` call is one full run (~1 hr on L4). Cell 6 = 15 runs total = ~18 GPU-hr. Cell 7 = 15 sweep runs = ~7 GPU-hr.

**Colab idle disconnect**: if you close the tab and the runtime sits idle without browser, Colab will eventually disconnect (typically 30-90 min after last interaction). Strategies:

- Keep the tab visible in a window (don't minimize).
- Or, run the for-loop inside a single cell — Colab considers a running cell as activity.
- If a session disconnects mid-sweep, completed runs persist in Drive (if mounted). Restart, comment out completed `(regime, seed)` pairs, resume.

---

## 4. Get the JSON back

If Drive was mounted, files are already in `Drive/sgt_runs/0p5b_smoke/`. Easy.

If Drive was skipped, use the Terminal:

```bash
cat /content/drive/MyDrive/sgt_runs/0p5b_smoke/R1_11.json
```

Copy the printed JSON. Or use the Files panel (left sidebar, folder icon) → navigate to the path → right-click → Download.

For multi-file retrieval:

```bash
cd /content/drive/MyDrive/sgt_runs && zip -r runs.zip 0p5b_smoke/ && ls -la runs.zip
```

Then download `runs.zip` from the Files panel.

---

## 5. Common failure modes

| Symptom | Cause | Fix |
|---|---|---|
| `operator torchvision::nms does not exist` | torchvision/torchaudio cu128 vs torch cu124 ABI | Uninstall torchvision/torchaudio (see above) |
| `ValueError: Attempting to unscale FP16 gradients` | Model loaded in fp16 + Trainer fp16=True | Already fixed in `sgt_runner.py` (commit 6af6264b) — re-clone if you see this |
| `TypeError: must be called with a dataclass type or instance` (during ARC load) | Legacy `ai2_arc` HF Hub alias broken with `datasets==3.1.0` | Already fixed in `sgt_runner.py` (uses `allenai/ai2_arc` now) |
| Cell hangs at "Generating test split" | First-time WikiText-103 download (~600 MB) | Wait. ~30s on Colab's egress. |
| "You are not subscribed to Colab Pro" | Account isn't on Pro tier or Pro session pool exhausted | Wait an hour; Pro re-pools fairly quickly |
| OOM on L4 (24 GB) for 0.5B | Should not happen at batch=8/seq=512 | Verify runtime is actually L4 (check `!nvidia-smi`) |
| Cell 5 runs but no `[gen 0]` line ever appears | Output buffering | Ignore — Colab flushes at line boundaries; just wait |

---

## 6. Notebook patches (pending)

The current `colab_runner.ipynb` cell 2 should be amended to:

```bash
%cd /content
!rm -rf prism
!git clone --depth 1 https://github.com/humanaiconvention/prism
%cd /content/prism
!pip install -q -e . transformers==4.46.3 datasets==3.1.0 accelerate==1.1.1 bitsandbytes==0.44.1
!pip install -q torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
!pip uninstall -q -y torchvision torchaudio    # <-- ADD THIS LINE
```

Without that line, every fresh runtime hits the `torchvision::nms` error. Patch is in commit history once applied.

The Kaggle notebook (`kaggle_runner.ipynb`) cell 1 has the same gap *and* doesn't pin torch at all — Kaggle ships its own torch in the base image, version drifts. Before the 1.5B sweep, add an explicit `!pip install torch==2.5.1 ...` line and verify cu version against `!nvidia-smi`.

---

## 7. Compute budget tracking

Colab Pro: ~45 GPU-hr/month effective (variable; depends on Pro session pool availability). Track spend in `sgt_progress.md` as runs land.

| Run | Hardware | Wall time | GPU-hr | Outcome |
|---|---|---|---|---|
| Smoke R1/11/g2/n200 | L4 | ~15 min | 0.25 | (fill in) |
| 0.5B main R1/11 | L4 | ~60 min | 1.0 | |
| 0.5B main R1/23 | L4 | ~60 min | 1.0 | |
| ... | | | | |

Total budget for 0.5B (5 regimes × 3 seeds + 5 fracs × 3 seeds) = 30 runs × ~1 hr ≈ 30 GPU-hr. Fits within Colab Pro.

For 1.5B: budget on Kaggle (T4×2, ~120 hr/mo) instead — 15 runs × ~3.5 hr ≈ 50 GPU-hr.

---

*Author: Claude (Opus 4.7) under operator B. Haslam, 2026-05-03. Living document — append findings as you go.*
