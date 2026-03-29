"""Get full traceback from genesis model crash."""
import sys
import traceback
import torch

sys.path.insert(0, ".")
from scripts.genesis_loader import load_genesis_model, format_chatml_prompt

model, tok, cfg = load_genesis_model()
device = next(model.parameters()).device
print(f"Device: {device}")

ids = torch.tensor([tok.encode(format_chatml_prompt("Hello"))], device=device)
print("Running forward pass...")
try:
    with torch.no_grad():
        out = model(ids, use_cache=True)
    print("SUCCESS!")
except Exception as e:
    print(f"\nCRASH: {type(e).__name__}: {e}")
    traceback.print_exc()
