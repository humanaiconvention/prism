import os
os.environ.setdefault("TRITON_INTERPRET", "1")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.genesis_loader import load_genesis_model
model, _, _ = load_genesis_model(device="cpu")
print("Model Attributes:")
for k, v in model.named_modules():
    if "norm" in k or "head" in k or "proj" in k:
        print(f" - {k}: {type(v).__name__}")
print("\nTop-level modules:", [name for name, _ in model.named_children()])
