"""Inspect Genesis model structure for experiment design."""
import os
os.environ["TRITON_INTERPRET"] = "1"

import json
from genesis import Genesis, GenesisConfig
from safetensors import safe_open

# Load config from checkpoint
weights_path = "weights/genesis_152m_instruct.safetensors"
with safe_open(weights_path, framework="pt", device="cpu") as f:
    metadata = f.metadata() or {}

config_dict = json.loads(metadata.get("genesis_config_json", "{}"))
config = GenesisConfig(**config_dict) if config_dict else GenesisConfig.genesis_147m()

print("=" * 60)
print("GENESIS CONFIG")
print("=" * 60)
for k, v in sorted(vars(config).items()):
    print(f"  {k}: {v}")

print("\n" + "=" * 60)
print("MODEL STRUCTURE (blocks)")
print("=" * 60)

import torch
from safetensors.torch import load_file
state_dict = load_file(weights_path, device="cpu")
model = Genesis(config).to(torch.float32)
model.load_state_dict(state_dict, strict=False)

# Inspect block structure
for i, block in enumerate(model.blocks):
    block_type = type(block).__name__
    submodules = [name for name, _ in block.named_children()]
    print(f"Block {i:2d} ({block_type}): {submodules}")
    if i < 2 or i in [3, 7]:  # Sample a few
        for name, mod in block.named_children():
            mod_type = type(mod).__name__
            params = sum(p.numel() for p in mod.parameters())
            print(f"  {name}: {mod_type} ({params:,} params)")
            if hasattr(mod, 'named_children'):
                for sname, smod in mod.named_children():
                    sparams = sum(p.numel() for p in smod.parameters())
                    if sparams > 0:
                        print(f"    {sname}: {type(smod).__name__} ({sparams:,} params)")

# Check TTT layer
print("\n" + "=" * 60)
print("TTT LAYER INSPECTION")
print("=" * 60)
if hasattr(model, 'ttt'):
    ttt = model.ttt
    print(f"TTT module: {type(ttt).__name__}")
    for name, param in ttt.named_parameters():
        print(f"  {name}: {param.shape}")
elif hasattr(model, 'ttt_layer'):
    print(f"TTT via ttt_layer: {type(model.ttt_layer).__name__}")
else:
    # Search for TTT in all modules
    for name, mod in model.named_modules():
        if 'ttt' in name.lower():
            print(f"  Found: {name} -> {type(mod).__name__}")
            for pname, param in mod.named_parameters(recurse=False):
                print(f"    {pname}: {param.shape}")

# Check selective activation
print("\n" + "=" * 60)
print("SELECTIVE ACTIVATION (sel_k)")
print("=" * 60)
for name, mod in model.named_modules():
    if 'sel' in name.lower() or 'gate' in name.lower() or 'topk' in name.lower():
        print(f"  {name}: {type(mod).__name__}")
