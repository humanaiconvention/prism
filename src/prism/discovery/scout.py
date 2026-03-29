import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from prism.causal.patching import ActivationPatcher
from prism.attention.circuits import AttentionAnalyzer
from prism.telemetry.schemas import CircuitReport

class CircuitScout:
    """
    Automated circuit discovery tool.
    Bridges low-level primitives into high-level causal analysis.
    """
    
    def __init__(self, model: Any, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self.patcher = ActivationPatcher(model)
        self.analyzer = AttentionAnalyzer(model)

    def _discover_attention_module_names(self) -> List[str]:
        module_to_name = {module: name for name, module in self.model.named_modules()}
        discovered: List[str] = []

        for layer_idx in range(len(self.analyzer.adapter.layers)):
            try:
                layer = self.analyzer.adapter.resolve_layer(layer_idx)
                attn = self.analyzer.adapter.resolve_attention_module(layer)
            except ValueError:
                continue

            name = module_to_name.get(attn)
            if name and name not in discovered:
                discovered.append(name)

        if discovered:
            return discovered

        # Fallback for mocked or unusual models that do not resolve through the adapter yet.
        fallback_tokens = (
            "attn",
            "attention",
            "self_attn",
            "self_attention",
            "selfattention",
            "encdecattention",
        )
        return [
            name
            for name, module in self.model.named_modules()
            if any(token in name.lower() for token in fallback_tokens)
            and (
                hasattr(module, "num_heads")
                or hasattr(module, "q_proj")
                or hasattr(module, "query_key_value")
                or hasattr(module, "self_attn")
                or hasattr(module, "self_attention")
            )
        ]

    def discover_circuit(self, 
                         clean_prompt: str, 
                         corrupt_prompt: str, 
                         target_token: str,
                         foil_token: str,
                         top_k: int = 10) -> Dict[str, Any]:
        """
        Discovers the circuit mediating the logit difference between target and foil.
        Uses a simplified Attribution Patching (AtP) guided approach.
        """
        clean_ids = self.tokenizer.encode(clean_prompt, return_tensors="pt").to(self.model.device)
        corrupt_ids = self.tokenizer.encode(corrupt_prompt, return_tensors="pt").to(self.model.device)
        
        target_id = self.tokenizer.encode(target_token, add_special_tokens=False)[0]
        foil_id = self.tokenizer.encode(foil_token, add_special_tokens=False)[0]
        
        # 1. Get baseline logit differences
        with torch.no_grad():
            clean_logits = self.model(clean_ids).logits[0, -1]
            corrupt_logits = self.model(corrupt_ids).logits[0, -1]
            
        clean_diff = (clean_logits[target_id] - clean_logits[foil_id]).item()
        corrupt_diff = (corrupt_logits[target_id] - corrupt_logits[foil_id]).item()
        
        # 2. Get activations for all attention heads in clean run
        attn_modules = self._discover_attention_module_names()
        
        clean_activations = self.patcher.get_activations(clean_ids, attn_modules)
        
        # 3. Perform head-level patching (Recursive/Iterative)
        head_effects = []
        
        for module_name in attn_modules:
            module = self.patcher._get_module_by_name(module_name)
            if not module: continue
            num_heads = module.num_heads
            
            for head_idx in range(num_heads):
                # Patch this specific head from clean to corrupt run
                patches = {module_name: (clean_activations[module_name], head_idx)}
                patched_logits = self.patcher.run_patched(corrupt_ids, patches)[0, -1]
                patched_diff = (patched_logits[target_id] - patched_logits[foil_id]).item()
                
                # Indirect effect: how much did patching this head restore the clean logit diff?
                effect = (patched_diff - corrupt_diff) / (clean_diff - corrupt_diff + 1e-9)
                head_effects.append({
                    "head": f"{module_name}.h{head_idx}",
                    "effect": effect,
                    "layer": module_name,
                    "idx": head_idx
                })
        
        # 4. Sort and return top-k
        head_effects.sort(key=lambda x: x["effect"], reverse=True)
        top_heads = head_effects[:top_k]

        report = CircuitReport(
            kind="circuit_discovery",
            model_name=getattr(getattr(self.model, "config", None), "_name_or_path", "") or "",
            prompt=clean_prompt,
            sections={
                "clean_diff": clean_diff,
                "corrupt_diff": corrupt_diff,
                "top_heads": top_heads,
                "total_heads_scanned": len(head_effects),
            },
            metadata={
                "target_token": target_token,
                "foil_token": foil_token,
                "corrupt_prompt": corrupt_prompt,
            },
        )

        return report.to_dict()

    def format_report(self, results: Dict[str, Any]) -> str:
        """Formats the discovery results into a readable string report."""
        if hasattr(results, "to_dict"):
            results = results.to_dict()
        report = []
        report.append("=== Circuit Discovery Report ===")
        report.append(f"Clean Logit Diff:   {results['clean_diff']:.4f}")
        report.append(f"Corrupt Logit Diff: {results['corrupt_diff']:.4f}")
        report.append(f"Heads Scanned:      {results['total_heads_scanned']}")
        report.append("-" * 30)
        report.append(f"{'Head':<20} | {'Effect':<10}")
        report.append("-" * 30)
        
        for head in results["top_heads"]:
            report.append(f"{head['head']:<20} | {head['effect']:>10.2%}")
            
        return "\n".join(report)
