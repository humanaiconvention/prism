"""Integration-ready LL-ACC scorer for sgt_runner.py.

Self-contained, matches probe_ll_acc.py exactly so it reproduces the C2 baseline
(raw 0.590 / norm 0.545 on Qwen2.5-0.5B, 200-item ARC-Easy). Intended to be
dropped into the runner as an ADDITIVE metric alongside the preregistered
substring grader — see INTEGRATION_NOTE_ll_acc.md.

Locked to the format `"Question: {q}\nAnswer: {a}"` per PREREG_ADDENDUM_ll_acc.md.

Unit test (run when GPU is free):  python ll_acc_eval.py --selftest
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


@torch.no_grad()
def eval_arc_llacc(model, tok, ds, normalize: bool = True) -> float:
    """Log-likelihood ARC accuracy. Locked format: 'Question: {q}\\nAnswer: {a}'.

    normalize=True  -> length-normalized acc_norm (PRIMARY per prereg addendum)
    normalize=False -> raw-sum-LL acc (secondary cross-check)

    Returns accuracy in [0, 1]. Robust to letter ('A'..) and numeric ('1'..)
    answerKeys because it argmaxes over choices and compares the chosen choice's
    label to answerKey.
    """
    model.eval()
    correct = 0
    n = 0
    for item in ds:
        labels = item["choices"]["label"]
        texts = item["choices"]["text"]
        if not texts:
            continue
        n += 1
        prompt = f"Question: {item['question']}\nAnswer:"
        prompt_ids = tok(prompt, return_tensors="pt").input_ids.to(model.device)
        plen = prompt_ids.shape[1]
        best_score = None
        best_label = None
        for lab, t in zip(labels, texts):
            full = tok(prompt + " " + t, return_tensors="pt").input_ids.to(model.device)
            logits = model(full).logits
            target = full[0, plen:]
            used = logits[0, plen - 1: full.shape[1] - 1]
            lp = F.log_softmax(used.float(), dim=-1)
            ll = lp.gather(1, target.unsqueeze(1)).squeeze(1).sum().item()
            score = ll / max(target.shape[0], 1) if normalize else ll
            if best_score is None or score > best_score:
                best_score = score
                best_label = lab
        if best_label is not None and best_label.strip().upper() == item["answerKey"].strip().upper():
            correct += 1
    return correct / n if n else float("nan")


def _selftest():
    """Verify against the C2 probe baseline: acc_norm ~ 0.545, acc_raw ~ 0.590."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B",
                                                 torch_dtype=torch.float16).to("cuda").eval()
    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test[:200]")
    acc_norm = eval_arc_llacc(model, tok, arc, normalize=True)
    acc_raw = eval_arc_llacc(model, tok, arc, normalize=False)
    print(f"acc_norm = {acc_norm:.3f}  (C2 expected ~0.545)")
    print(f"acc_raw  = {acc_raw:.3f}  (C2 expected ~0.590)")
    ok = abs(acc_norm - 0.545) < 0.02 and abs(acc_raw - 0.590) < 0.02
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(0 if _selftest() else 1)
    print("import this module; run with --selftest to verify against C2 baseline")
