"""Quick GPU speed test for WSL + CUDA + fla Triton kernels."""
import sys
import time
import torch

sys.path.insert(0, ".")
from scripts.genesis_loader import load_genesis_model, format_chatml_prompt

model, tok, cfg = load_genesis_model()
device = next(model.parameters()).device
print(f"Device: {device}")

# Warmup (JIT compile Triton kernels)
ids = torch.tensor([tok.encode(format_chatml_prompt("Hello"))], device=device)
print("Warming up (JIT compiling Triton kernels)...")
t0 = time.time()
with torch.no_grad():
    out = model(ids, use_cache=True)
if device.type == "cuda":
    torch.cuda.synchronize()
print(f"Warmup: {time.time()-t0:.1f}s")

# Speed test
ids2 = torch.tensor([tok.encode(format_chatml_prompt("Explain quantum entanglement"))], device=device)
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
if device.type == "cuda":
    torch.cuda.synchronize()
elapsed = time.time() - t1
print(f"32 tokens in {elapsed:.1f}s = {32/elapsed:.1f} tok/s")
print(f"Speedup vs CPU baseline (0.3 tok/s): {(32/elapsed)/0.3:.1f}x")
