"""Activation patching, attribution patching, and ablation methods."""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple, Callable, Optional
from contextlib import contextmanager

class ActivationPatcher:
    """Tool for exact activation patching between clean and corrupted prompts."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self._cache = {}
        self._hooks = []

    def _clear_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []
        
    def _get_module_by_name(self, name: str) -> Optional[torch.nn.Module]:
        """Simple helper to find module by name string."""
        for n, m in self.model.named_modules():
            if n == name:
                return m
        return None

    @contextmanager
    def trace(self):
        """Context manager to ensure hooks are cleaned up after the forward pass."""
        try:
            yield self
        finally:
            self._clear_hooks()

    def cache_activation(self, module: nn.Module, cache_key: str):
        """Registers a forward hook to save the output of a module."""
        def hook(mod, inp, out):
            state = out[0] if isinstance(out, tuple) else out
            self._cache[cache_key] = state.detach().clone()
        
        handle = module.register_forward_hook(hook)
        self._hooks.append(handle)
        return handle

    def inject_activation(self, module: nn.Module, cache_key: str, position_idx: int = -1):
        """Registers a forward hook to overwrite the output with a cached activation."""
        def hook(mod, inp, out):
            if cache_key not in self._cache:
                return out
            is_tuple = isinstance(out, tuple)
            state = out[0] if is_tuple else out
            cached_state = self._cache[cache_key]
            new_state = state.clone()
            new_state[:, position_idx, :] = cached_state[:, position_idx, :]
            if is_tuple:
                return (new_state,) + out[1:]
            return new_state
            
        handle = module.register_forward_hook(hook)
        self._hooks.append(handle)
        return handle

    def get_activations(self, input_ids: torch.Tensor, layer_names: List[str]) -> Dict[str, torch.Tensor]:
        """Runs the model and returns activations for specified layers."""
        self._cache = {}
        
        for name in layer_names:
            module = self._get_module_by_name(name)
            if module:
                self.cache_activation(module, name)
        
        with self.trace():
            with torch.no_grad():
                self.model(input_ids)
                
        # Return a copy to avoid tying to the class state directly
        return {k: v.clone() for k, v in self._cache.items()}

    def run_patched(self, 
                    input_ids: torch.Tensor, 
                    patches: Dict[str, Tuple[torch.Tensor, Optional[int]]]) -> torch.Tensor:
        """
        Runs model with specified patches applied.
        patches: { layer_name: (patch_tensor, head_idx) }
        """
        # Note: Preload cache so inject_activation finds them
        self._cache = {k: v[0] for k, v in patches.items()}
        
        for name, (patch_data, head_idx) in patches.items():
            module = self._get_module_by_name(name)
            if module:
                if head_idx is not None:
                    # Provide an ad-hoc hook for head injection
                    def hook(mod, inp, out, pd=patch_data, h_idx=head_idx):
                        n_heads = mod.num_heads if hasattr(mod, 'num_heads') else 1
                        d_head = pd.shape[-1] // n_heads
                        start = h_idx * d_head
                        end = start + d_head
                        
                        is_tuple = isinstance(out, tuple)
                        state = out[0] if is_tuple else out
                        new_state = state.clone()
                        new_state[..., start:end] = pd[..., start:end]
                        if is_tuple:
                            return (new_state,) + out[1:]
                        return new_state
                    handle = module.register_forward_hook(hook)
                    self._hooks.append(handle)
                else:
                    self.inject_activation(module, name)
        
        with self.trace():
            outputs = self.model(input_ids)
            return outputs.logits

class AttributionPatcher:
    """Gradient-based approximation of activation patching (AtP)."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self._grad_cache = {}

    def compute_attribution(self, 
                           clean_inputs: Dict[str, torch.Tensor], 
                           corrupt_inputs: Dict[str, torch.Tensor], 
                           target_modules: Dict[str, nn.Module],
                           metric_fn: Callable[[torch.Tensor], torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Estimates the causal effect of activations using the linear approximation:
        Effect = Grad * (Clean_Act - Corrupt_Act)
        """
        # 1. Run corrupt pass to get corrupt activations
        corrupt_acts = {}
        def corrupt_hook_fn(name):
            def hook(mod, inp, out):
                state = out[0] if isinstance(out, tuple) else out
                corrupt_acts[name] = state.detach().clone()
            return hook

        handles = []
        for name, mod in target_modules.items():
            handles.append(mod.register_forward_hook(corrupt_hook_fn(name)))
        
        self.model.eval()
        with torch.no_grad():
            self.model(**corrupt_inputs)
        for h in handles: h.remove()

        # 2. Run clean pass with gradients
        clean_acts = {}
        def clean_hook_fn(name):
            def hook(mod, inp, out):
                state = out[0] if isinstance(out, tuple) else out
                # Store clean act and require grad
                clean_acts[name] = state
                state.retain_grad()
            return hook

        handles = []
        for name, mod in target_modules.items():
            handles.append(mod.register_forward_hook(clean_hook_fn(name)))
        
        outputs = self.model(**clean_inputs)
        logits = outputs.logits
        
        # 3. Compute metric and backprop
        metric_val = metric_fn(logits)
        self.model.zero_grad()
        metric_val.backward()

        # 4. Calculate Attribution
        attributions = {}
        for name in target_modules:
            grad = clean_acts[name].grad
            if grad is not None:
                # Linear approximation of the patch effect
                diff = clean_acts[name] - corrupt_acts[name].to(clean_acts[name].device)
                # Sum over features, keep tokens/batch
                attr = (grad * diff).sum(dim=-1)
                attributions[name] = attr.detach()

        for h in handles: h.remove()
        return attributions
