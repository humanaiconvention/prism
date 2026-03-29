"""Genesis-152M model loader and utilities.

Wraps the genesis-llm package to provide a consistent interface
for spectral analysis, head ablation, and hidden state extraction.

IMPORTANT: On Windows, set TRITON_INTERPRET=1 before importing genesis
to avoid the C compiler JIT requirement for GLA kernels.
"""

import os
os.environ.setdefault("TRITON_INTERPRET", "1")  # Must be before genesis import

import json
import argparse
import torch
from pathlib import Path


# ============================================================================
# Confirmed FoX/GLA layer layout (from source: (layer_idx+1) % 4 == 0)
# ============================================================================
FOX_LAYER_INDICES = [3, 7, 11, 15, 19, 23, 27]  # 7 layers: O(n^2) softmax + forget gate
GLA_LAYER_INDICES = [i for i in range(30) if i not in FOX_LAYER_INDICES]  # 23 layers: O(n) linear


def load_genesis_model(
    weights_path: str = None,
    device: str = None,
    dtype: torch.dtype = torch.float32,
):
    """Load Genesis-152M-Instruct model and tokenizer.
    
    Args:
        weights_path: Path to .safetensors weights file. If None, tries default locations.
        device: Device string ('cuda', 'mps', 'cpu'). Auto-detected if None.
        dtype: Torch dtype for the model.
    
    Returns:
        Tuple of (model, tokenizer, config)
    """
    from genesis import Genesis, GenesisConfig, get_tokenizer
    from safetensors import safe_open
    from safetensors.torch import load_file
    
    # Find weights
    if weights_path is None:
        candidates = [
            Path("weights/genesis_152m_instruct.safetensors"),
            Path("genesis_152m_instruct.safetensors"),
            Path(__file__).parent.parent / "weights" / "genesis_152m_instruct.safetensors",
        ]
        for c in candidates:
            if c.exists():
                weights_path = str(c)
                break
        if weights_path is None:
            raise FileNotFoundError(
                "Could not find genesis_152m_instruct.safetensors. "
                "Download with: huggingface-cli download guiferrarib/genesis-152m-instruct "
                "genesis_152m_instruct.safetensors --local-dir weights/"
            )
    
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    # Load config from checkpoint metadata
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        metadata = f.metadata() or {}
    
    config_dict = json.loads(metadata.get("genesis_config_json", "{}"))
    if config_dict:
        config = GenesisConfig(**config_dict)
    else:
        config = GenesisConfig.genesis_147m()
    
    # Load model weights
    state_dict = load_file(weights_path, device=device)
    model = Genesis(config).to(dtype).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Setup tokenizer (GPT-NeoX + ChatML tokens)
    tokenizer = get_tokenizer("neox")
    tokenizer.add_chat_tokens()
    
    print(f"[Genesis Loader] Model loaded on {device} ({dtype})")
    print(f"[Genesis Loader] Config: {config.n_layer} layers, n_embd={config.n_embd}, "
          f"heads={config.n_head}Q/{config.n_kv_head}KV, head_dim={config.head_dim}")
    print(f"[Genesis Loader] FoX layers: {FOX_LAYER_INDICES}")
    print(f"[Genesis Loader] GLA layers: {GLA_LAYER_INDICES}")
    
    return model, tokenizer, config


def format_chatml_prompt(prompt: str, system_prompt: str = "You are a helpful assistant."):
    """Format a prompt in ChatML format for Genesis."""
    return (
        f"<|im_start|>system\n{system_prompt}\n<|im_end|>\n"
        f"<|im_start|>user\n{prompt}\n<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def get_layer_info():
    """Return confirmed FoX/GLA layer mapping.
    
    FoX layers are at (layer_idx + 1) % 4 == 0, confirmed from model source.
    """
    return {
        "fox_layers": FOX_LAYER_INDICES,
        "gla_layers": GLA_LAYER_INDICES,
        "n_fox": len(FOX_LAYER_INDICES),
        "n_gla": len(GLA_LAYER_INDICES),
    }


class HiddenStateHook:
    """Forward hook that captures hidden states from each block's output.
    
    Genesis forward doesn't support output_hidden_states, so we hook into
    each GenesisBlock to capture the residual stream at each layer.
    
    Usage:
        hooks = HiddenStateHook.attach_to_model(model)
        # ... run forward pass ...
        hidden_states = hooks.get_hidden_states()
        hooks.remove_all()
    """
    
    def __init__(self):
        self.hidden_states = []  # List of (layer_idx, tensor)
        self.handles = []
    
    def _make_hook(self, layer_idx):
        """Create a hook function for a specific layer."""
        def hook_fn(module, input_tensor, output):
            # GenesisBlock.forward returns (x, aux_loss) or (x, aux_loss, state, kv)
            if isinstance(output, tuple):
                x = output[0]
            else:
                x = output
            # Capture last position hidden state: shape (batch, seq_len, n_embd)
            self.hidden_states.append((layer_idx, x[:, -1, :].detach().float().cpu()))
        return hook_fn
    
    @classmethod
    def attach_to_model(cls, model):
        """Attach hooks to all blocks in a Genesis model."""
        instance = cls()
        for i, block in enumerate(model.blocks):
            handle = block.register_forward_hook(instance._make_hook(i))
            instance.handles.append(handle)
        return instance
    
    def get_hidden_states(self):
        """Get captured hidden states as a list of tensors."""
        return [h for _, h in sorted(self.hidden_states, key=lambda x: x[0])]
    
    def clear(self):
        """Clear captured states (call between forward passes)."""
        self.hidden_states = []
    
    def remove_all(self):
        """Remove all hooks."""
        for h in self.handles:
            h.remove()
        self.handles = []


def inspect_model_architecture(model, config):
    """Print detailed architecture breakdown."""
    print(f"\n{'='*60}")
    print(f"Genesis Architecture Inspection")
    print(f"{'='*60}")
    print(f"Total layers: {config.n_layer}")
    print(f"Hidden dim (n_embd): {config.n_embd}")
    print(f"Intermediate size: {config.intermediate_size}")
    print(f"Head dim: {config.head_dim}")
    print(f"Total heads (query): {config.n_head}")
    print(f"KV heads: {config.n_kv_head}")
    print(f"GQA ratio: {config.n_head // config.n_kv_head}:1")
    print(f"Context length: {config.block_size}")
    print(f"GLA: {config.use_gla}, FoX: {getattr(config, 'use_fox', False)}, TTT: {config.use_ttt}")
    print(f"Hybrid ratio: {config.hybrid_full_attn_ratio} (FoX fraction)")
    
    print(f"\nLayer-by-layer breakdown:")
    print(f"{'-'*60}")
    
    fox_layers = []
    gla_layers = []
    
    for i, block in enumerate(model.blocks):
        is_fox = getattr(block, 'use_full_attention', False)
        attn_type = type(block.attn).__name__
        
        if is_fox:
            fox_layers.append(i)
            marker = "FoX (softmax + forget gate)"
        else:
            gla_layers.append(i)
            marker = "GLA (DeltaNet linear)"
        
        # Get o_proj info for hook point identification
        o_proj_info = ""
        for name, mod in block.attn.named_modules():
            if 'o_proj' in name or 'out_proj' in name:
                if hasattr(mod, 'weight'):
                    o_proj_info = f" | o_proj: {mod.weight.shape}"
                break
        
        print(f"  Layer {i:2d}: {attn_type:30s} [{marker}]{o_proj_info}")
    
    print(f"\n{'='*60}")
    print(f"FoX layers ({len(fox_layers)}): {fox_layers}")
    print(f"GLA layers ({len(gla_layers)}): {gla_layers}")
    print(f"{'='*60}")
    
    return fox_layers, gla_layers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genesis-152M Model Loader")
    parser.add_argument("--weights", type=str, default=None, help="Path to safetensors weights")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--smoke-test", action="store_true", help="Run smoke test generation")
    parser.add_argument("--inspect-layers", action="store_true", help="Inspect model architecture layers")
    args = parser.parse_args()
    
    model, tokenizer, config = load_genesis_model(
        weights_path=args.weights,
        device=args.device,
    )
    
    if args.inspect_layers:
        inspect_model_architecture(model, config)
    
    if args.smoke_test:
        print("\n--- Smoke Test ---")
        prompt = format_chatml_prompt("What is 2 + 2?")
        input_ids = torch.tensor([tokenizer.encode(prompt)], device=next(model.parameters()).device)
        
        print(f"Input shape: {input_ids.shape}")
        
        # Test hidden state hooks
        hooks = HiddenStateHook.attach_to_model(model)
        
        with torch.no_grad():
            # Genesis forward: (logits, loss, metrics) — takes idx directly
            logits, loss, metrics = model(input_ids)
        
        hidden_states = hooks.get_hidden_states()
        hooks.remove_all()
        
        print(f"Captured {len(hidden_states)} layer hidden states")
        if hidden_states:
            print(f"Hidden state shape: {hidden_states[0].shape}")
        
        # Test generation
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=32, temperature=0.7)
        
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:].tolist())
        print(f"Response: {response}")
        print(f"Output tokens: {output_ids.shape[1] - input_ids.shape[1]}")
        print("--- Smoke Test PASSED ---")
