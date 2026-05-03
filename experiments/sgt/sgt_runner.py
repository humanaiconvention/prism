"""SGT recursive grounding runner — pre-registered v1.

One run = one (regime, seed, model_scale) configuration.
Outputs runs/<model>/<regime>_<seed>.json consumable by prism.eval.early_warning.

Regimes (must match preregistration.md §2):
  R1        : synthetic-only, replace        (closure)
  R1_accum  : synthetic-only, accumulate     (closure, slow)
  R2        : 50/50 synthetic + frozen real anchor
  R3        : 50/50 synthetic + fresh real slice per generation
  R4        : 50/50 synthetic + teacher-corrected (teacher relabels synthetic)
  Rn_<frac> : R4-style with explicit correction_frac in [0,1]   (sweep)

Usage:
  python sgt_runner.py --regime R1 --seed 11 --model Qwen/Qwen2.5-0.5B --out runs/0p5b
"""
from __future__ import annotations
import argparse, json, os, random, time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed,
)

# ---------- determinism ----------

def lock_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ---------- config ----------

@dataclass
class RunCfg:
    regime: str
    seed: int
    model: str
    teacher: str | None = None
    generations: int = 5
    samples_per_gen: int = 2000
    max_new_tokens: int = 128
    train_epochs: int = 3
    batch_size: int = 8
    eval_arc_n: int = 200
    eval_wikitext_lines: int = 500
    correction_frac: float | None = None  # for sweep variants
    out_dir: str = "runs"

REGIME_TABLE = {
    # regime -> (replace, synthetic_frac, real_source, oracle)
    "R1":       (True,  1.0, "none",   "none"),
    "R1_accum": (False, 1.0, "none",   "none"),
    "R2":       (True,  0.5, "anchor", "none"),
    "R3":       (True,  0.5, "fresh",  "none"),
    "R4":       (True,  0.5, "none",   "teacher_filter"),
}

def regime_spec(cfg: RunCfg):
    if cfg.regime in REGIME_TABLE:
        return REGIME_TABLE[cfg.regime]
    if cfg.regime.startswith("Rn_") and cfg.correction_frac is not None:
        f = cfg.correction_frac
        return (True, 1.0 - f, "none", "teacher_filter" if f > 0 else "none")
    raise ValueError(f"unknown regime: {cfg.regime}")

# ---------- data ----------

def real_train_slice(start: int, n: int) -> Dataset:
    """Disjoint slice of WikiText-103 train. Used for R2 anchor + R3 fresh."""
    return load_dataset("wikitext", "wikitext-103-v1",
                        split=f"train[{start}:{start+n}]")

def real_val() -> Dataset:
    return load_dataset("wikitext", "wikitext-2-v1", split="validation[:500]")

def arc_easy(n: int) -> Dataset:
    # Use the canonical allenai/ai2_arc path. The legacy "ai2_arc" alias on HF Hub
    # now emits a features schema unsupported by datasets==3.1.0; the canonical
    # path returns identical content (same id/question/choices/answerKey schema).
    return load_dataset("allenai/ai2_arc", "ARC-Easy", split=f"test[:{n}]")

# ---------- evaluation ----------

@torch.no_grad()
def eval_perplexity(model, tok, val: Dataset) -> float:
    model.eval()
    txt = "\n\n".join(val["text"][: 500])
    enc = tok(txt, return_tensors="pt", truncation=True, max_length=1024)
    ids = enc.input_ids.to(model.device)
    out = model(ids, labels=ids)
    return float(torch.exp(out.loss).item())

@torch.no_grad()
def eval_arc(model, tok, ds: Dataset) -> float:
    model.eval(); correct = 0
    for ex in ds:
        prompt = f"Question: {ex['question']}\nAnswer:"
        x = tok(prompt, return_tensors="pt").to(model.device)
        gen = model.generate(**x, max_new_tokens=8, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        ans = tok.decode(gen[0, x.input_ids.shape[1]:], skip_special_tokens=True).strip()
        gold_letter = ex["answerKey"].strip()
        if ans[:1].upper() == gold_letter.upper():
            correct += 1
    return correct / len(ds)

# ---------- generation of synthetic corpus ----------

@torch.no_grad()
def make_synthetic(model, tok, prompts_source, n: int, max_new_tokens: int):
    model.eval(); out = []
    seeds = [" ".join(p.split()[:5]) for p in prompts_source["text"][:n] if p.strip()]
    for s in seeds:
        x = tok(s, return_tensors="pt").to(model.device)
        gen = model.generate(**x, max_new_tokens=max_new_tokens, do_sample=True,
                             top_k=50, pad_token_id=tok.eos_token_id)
        out.append({"text": tok.decode(gen[0], skip_special_tokens=True)})
    return Dataset.from_list(out)

# ---------- teacher correction (R4) ----------

def teacher_relabel(synthetic: Dataset, teacher_model, teacher_tok, frac: float) -> Dataset:
    """Replace `frac` of the synthetic items with teacher continuations of the same prefix.
    Cheap proxy for verifier-in-the-loop; deterministic decoding for the teacher."""
    n_keep = int(len(synthetic) * (1.0 - frac)); n_relabel = len(synthetic) - n_keep
    out = list(synthetic.select(range(n_keep)))
    for ex in synthetic.select(range(n_keep, n_keep + n_relabel)):
        seed_txt = " ".join(ex["text"].split()[:5])
        x = teacher_tok(seed_txt, return_tensors="pt").to(teacher_model.device)
        with torch.no_grad():
            g = teacher_model.generate(**x, max_new_tokens=128, do_sample=False,
                                       pad_token_id=teacher_tok.eos_token_id)
        out.append({"text": teacher_tok.decode(g[0], skip_special_tokens=True)})
    return Dataset.from_list(out)

# ---------- training ----------

def fine_tune(model, tok, ds: Dataset, epochs: int, batch_size: int, work_dir: str):
    def tokenize(ex):
        return tok(ex["text"], truncation=True, max_length=512, padding="max_length")
    tok_ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    tok_ds = tok_ds.map(lambda e: {**e, "labels": e["input_ids"]})
    args = TrainingArguments(
        output_dir=work_dir, num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, logging_steps=100,
        save_strategy="no", report_to=[], seed=42, fp16=torch.cuda.is_available(),
    )
    Trainer(model=model, args=args, train_dataset=tok_ds).train()
    return model

# ---------- main loop ----------

def run(cfg: RunCfg) -> dict:
    lock_seeds(cfg.seed)
    replace, synth_frac, real_source, oracle = regime_spec(cfg)

    tok = AutoTokenizer.from_pretrained(cfg.model)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load in fp32 (default). Trainer's fp16=True needs fp32 master weights —
    # loading the model as fp16 here triggers "Attempting to unscale FP16 gradients"
    # at the first optimizer step. Generation/eval still run in mixed precision via Trainer.
    model = AutoModelForCausalLM.from_pretrained(cfg.model).to(device)

    teacher = teacher_tok = None
    if oracle == "teacher_filter" and cfg.teacher:
        teacher_tok = AutoTokenizer.from_pretrained(cfg.teacher)
        if teacher_tok.pad_token is None: teacher_tok.pad_token = teacher_tok.eos_token
        teacher = AutoModelForCausalLM.from_pretrained(
            cfg.teacher, torch_dtype=torch.float16, device_map="auto", load_in_8bit=True
        )

    val = real_val(); arc = arc_easy(cfg.eval_arc_n)
    anchor = real_train_slice(0, cfg.samples_per_gen) if real_source == "anchor" else None

    # gen 0 = baseline
    history = []
    ppl0 = eval_perplexity(model, tok, val); acc0 = eval_arc(model, tok, arc)
    history.append({"generation": 0, "grounded_arc_perplexity": ppl0,
                    "grounded_arc_accuracy": acc0})
    print(f"[gen 0] ppl={ppl0:.3f} acc={acc0:.3f}")

    accum_synth: Dataset | None = None
    fresh_offset = cfg.samples_per_gen  # R3 starts after the seed slice
    seed_real = real_train_slice(0, cfg.samples_per_gen)  # used as initial prompt source

    prompt_source = seed_real
    for gen in range(1, cfg.generations + 1):
        # 1. produce this generation's synthetic corpus
        synth = make_synthetic(model, tok, prompt_source,
                               cfg.samples_per_gen, cfg.max_new_tokens)

        # 2. apply oracle correction if R4-style
        if oracle == "teacher_filter":
            corr_frac = cfg.correction_frac if cfg.correction_frac is not None else 0.5
            synth = teacher_relabel(synth, teacher, teacher_tok, corr_frac)

        # 3. assemble training set per regime
        n_synth = int(cfg.samples_per_gen * synth_frac)
        parts = [synth.select(range(min(n_synth, len(synth))))]
        n_real = cfg.samples_per_gen - n_synth
        if n_real > 0 and real_source == "anchor":
            parts.append(anchor.select(range(min(n_real, len(anchor)))))
        if n_real > 0 and real_source == "fresh":
            parts.append(real_train_slice(fresh_offset, n_real))
            fresh_offset += n_real

        train_ds = concatenate_datasets(parts).shuffle(seed=cfg.seed)

        if not replace and accum_synth is not None:
            train_ds = concatenate_datasets([accum_synth, train_ds]).shuffle(seed=cfg.seed)
        accum_synth = train_ds  # for next gen if accumulate

        # 4. fine-tune on this gen's data
        fine_tune(model, tok, train_ds, cfg.train_epochs, cfg.batch_size,
                  work_dir=f"{cfg.out_dir}/_tmp_g{gen}")

        # 5. eval against immutable anchors
        ppl = eval_perplexity(model, tok, val); acc = eval_arc(model, tok, arc)
        history.append({"generation": gen,
                        "grounded_arc_perplexity": ppl,
                        "grounded_arc_accuracy": acc})
        print(f"[gen {gen}] ppl={ppl:.3f} acc={acc:.3f}")

        prompt_source = train_ds  # next gen seeds from this gen's training data

    return {"cfg": asdict(cfg), "history": history,
            "baseline": {"ppl": ppl0, "acc": acc0},
            "wallclock_s": None, "torch_version": torch.__version__}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--generations", type=int, default=5)
    ap.add_argument("--samples_per_gen", type=int, default=2000)
    ap.add_argument("--correction_frac", type=float, default=None)
    ap.add_argument("--out", default="runs/0p5b")
    a = ap.parse_args()

    cfg = RunCfg(regime=a.regime, seed=a.seed, model=a.model, teacher=a.teacher,
                 generations=a.generations, samples_per_gen=a.samples_per_gen,
                 correction_frac=a.correction_frac, out_dir=a.out)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    result = run(cfg)
    result["wallclock_s"] = round(time.time() - t0, 1)
    name = f"{a.regime}_{a.seed}"
    if a.correction_frac is not None:
        name = f"{a.regime}_f{int(a.correction_frac*100)}_{a.seed}"
    out_path = Path(a.out) / f"{name}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"wrote {out_path}  ({result['wallclock_s']}s)")

if __name__ == "__main__":
    main()
