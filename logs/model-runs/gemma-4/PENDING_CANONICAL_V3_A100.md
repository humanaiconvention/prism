# PENDING: Canonical A100 PRISM Run for HAIC Gemma 4 E2B v3

**Status:** Awaiting adapter download + A100 re-run

**Kaggle Session Info:**
- Session: `kaggle-gemma4-v3`
- Adapter location: `gemma4-v3/adapter/best/`
- **URGENT:** Session will expire — download before then
- Training used: T4 GPU (NaN collapse occurred; see `20260413_v3_t4_naN_artifact/run.md`)

**Why canonical A100 is needed:**
- The T4 result (`qh=0.0474`) is a NaN artifact (34% step collapse)
- Canonical requires: A100 (clean activations) + 0 NaN + bf16 merged checkpoint
- Once obtained, run with: `scan_model_geometry(merged_ckpt, ...)` on Colab A100

**Next Steps:**
1. ✅ Download adapter from Kaggle session (`gemma4-v3/adapter/best/`)
2. ⏳ Merge with base Gemma 4 E2B (bf16) on A100 or local
3. ⏳ Run PRISM scan: `prism.geometry.scan_model_geometry()`
4. ⏳ Create `20260413_v3_a100_canonical/run.md` with result
5. ⏳ Update `index.md` to replace T4 artifact row with canonical row

**Do NOT compare v3 to v2 until canonical A100 PRISM is available.**
