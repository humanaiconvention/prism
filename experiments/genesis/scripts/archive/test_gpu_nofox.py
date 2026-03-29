"""Test genesis model speed with fla FoX disabled (GLA-only acceleration)."""
import sys
import time
import torch

# Monkey-patch: block fla FoX import to force naive fallback for FoX only
import importlib
_orig_import = __builtins__.__import__
def _patched_import(name, *args, **kwargs):
    if "forgetting_attn" in name:
        raise ImportError(f"Blocked: {name}")
    return _orig_import(name, *args, **kwargs)
__builtins__.__import__ = _patched_import

sys.path.insert(0, ".")
from scripts.genesis_loader import load_genesis_model, format_chatml_prompt

# Restore import
__builtins__.__import__ = _orig_import

model, tok, cfg = load_genesis_model()
device = next(model.parameters()).device
print(f"Device: {device}")

# Warmup
ids = torch.tensor([tok.encode(format_chatml_prompt("Hello"))], device=device)
print("Warming up...")
t0 = time.time()
with torch.no_grad():
    out = model(ids, use_cache=True)
torch.cuda.synchronize()
print(f"Warmup: {time.time()-t0:.1f}s")

# Speed test
ids2 = torch.tensor([tok.encode(format_chatml_prompt("Explain quantum"))], device=device)
past = None
t1 = time.time()
with torch.no_grad():
    for i in range(32):
        if past:
            out = model(ids2[:, -1:], past_key_values=past, use_cache=True)
        else:
            out = model(ids2, use_cache=True)
        past = out[3]
        nxt = out[0][:, -1, :].argmax(-1)
        ids2 = torch.cat([ids2, nxt.unsqueeze(0)], dim=1)
torch.cuda.synchronize()
elapsed = time.time() - t1
print(f"32 tokens in {elapsed:.1f}s = {32/elapsed:.1f} tok/s")
