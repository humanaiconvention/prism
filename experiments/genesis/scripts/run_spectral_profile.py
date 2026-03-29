"""Phase 0: Spectral profiling for Genesis-152M-Instruct.

Runs spectral telemetry on Genesis, producing per-token
spectral entropy, effective dimension, streaming effective dimension,
and cross-layer projection angle.

Uses forward hooks on GenesisBlock to capture hidden states since
Genesis's forward() doesn't support output_hidden_states.

Usage:
    python scripts/run_spectral_profile.py \
        --weights weights/genesis_152m_instruct.safetensors \
        --prompts prompts/prompts_60.json \
        --max-prompts 60 \
        --max-tokens 256 \
        --output logs/phase0_spectral_profile.csv
"""

import os
os.environ.setdefault("TRITON_INTERPRET", "1")  # Must be before genesis import

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import (
    load_genesis_model, format_chatml_prompt, 
    HiddenStateHook, FOX_LAYER_INDICES,
)
from spectral_microscope.analysis import compute_shannon_effective_rank


def run_spectral_profile(
    model,
    tokenizer,
    config,
    prompts: list,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    do_sample: bool = True,
    output_csv: str = "logs/phase0_spectral_profile.csv",
    streaming_cov_alpha: float = 0.95,
):
    """Run spectral profiling on all prompts.
    
    Inline (autoregressive) capture: generates tokens one at a time,
    captures hidden states via hooks at each step, computes spectral metrics.
    """
    from spectral_microscope.analysis import compute_spectral_metrics
    
    device = next(model.parameters()).device
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    
    gla_layer_indices = set(range(30)) - set(FOX_LAYER_INDICES)
    
    fieldnames = [
        "prompt_idx", "prompt_text", "category", "step", "token",
        "nll", "spectral_entropy", "effective_dim", "shannon_eff_rank",
        "streaming_eff_dim", "projection_angle", "hidden_norm",
        "gla_mean_norm", "fox_mean_norm",
    ]
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for pidx, prompt_data in enumerate(prompts):
            prompt_text = prompt_data if isinstance(prompt_data, str) else prompt_data.get("text", prompt_data.get("prompt", ""))
            category = prompt_data.get("category", "unknown") if isinstance(prompt_data, dict) else "unknown"
            
            print(f"\n[{pidx+1}/{len(prompts)}] Category: {category}")
            print(f"  Prompt: {prompt_text[:80]}...")
            
            t0 = time.time()
            
            # Format and tokenize
            chat_input = format_chatml_prompt(prompt_text)
            input_ids = torch.tensor(
                [tokenizer.encode(chat_input)], 
                device=device
            )
            
            generated_ids = []
            per_step_hidden = []  # Last-layer hidden at each step
            per_step_nll = []
            per_step_angle = []
            
            # Step-by-step autoregressive generation with hidden state capture
            # Genesis uses KV cache via use_cache=True
            current_ids = input_ids
            past_key_values = None
            
            for step_idx in range(max_new_tokens):
                # Attach hooks for this forward pass
                hooks = HiddenStateHook.attach_to_model(model)
                
                with torch.no_grad():
                    if past_key_values is not None:
                        # Feed only last token with cache
                        step_input = current_ids[:, -1:]
                        out = model(step_input, past_key_values=past_key_values, use_cache=True)
                    else:
                        # First step: feed full context
                        out = model(current_ids, use_cache=True)
                    
                    # Unpack: (logits, loss, metrics, past_key_values)
                    logits = out[0]
                    past_key_values = out[3]
                
                # Get hidden states from hooks
                hidden_states = hooks.get_hidden_states()
                hooks.remove_all()
                
                # Logits come from the last position
                step_logits = logits[:, -1, :]
                
                # Sample next token
                if do_sample and temperature > 0:
                    scaled_logits = step_logits / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1).item()
                else:
                    next_id = int(torch.argmax(step_logits, dim=-1).item())
                
                # Compute NLL
                log_probs = torch.log_softmax(step_logits, dim=-1)
                nll = -log_probs[0, next_id].item()
                per_step_nll.append(nll)
                
                generated_ids.append(next_id)
                
                # Update input sequence
                next_token = torch.tensor([[next_id]], device=device)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # Check EOS
                eos_id = getattr(tokenizer, 'eos_token_id', None)
                if eos_id is not None and next_id == eos_id:
                    break
                
                # Capture hidden states per layer type
                if hidden_states:
                    h_last = hidden_states[-1]  # Shape: (1, n_embd)
                    per_step_hidden.append(h_last.squeeze(0))
                    
                    # Per-layer-type norms (GLA vs FoX geometry audit)
                    gla_norms = [hidden_states[i].squeeze(0).norm().item()
                                 for i in sorted(gla_layer_indices) if i < len(hidden_states)]
                    fox_norms = [hidden_states[i].squeeze(0).norm().item()
                                 for i in FOX_LAYER_INDICES if i < len(hidden_states)]
                    per_step_gla_norm = sum(gla_norms) / max(len(gla_norms), 1)
                    per_step_fox_norm = sum(fox_norms) / max(len(fox_norms), 1)
                    
                    # Cross-layer projection angle (early vs late layer)
                    if len(hidden_states) > 2:
                        early = hidden_states[1].squeeze(0)   # Layer 1
                        late = hidden_states[-2].squeeze(0)    # Second-to-last layer
                        angle = torch.nn.functional.cosine_similarity(
                            early.unsqueeze(0), late.unsqueeze(0)
                        ).item()
                        per_step_angle.append(float(angle))
                    else:
                        per_step_angle.append(0.0)
            
            elapsed = time.time() - t0
            tokens_generated = len(generated_ids)
            print(f"  Generated {tokens_generated} tokens in {elapsed:.1f}s "
                  f"({tokens_generated/max(elapsed,0.01):.1f} tok/s)")
            
            if per_step_nll:
                mean_nll = sum(per_step_nll) / len(per_step_nll)
                print(f"  Mean NLL: {mean_nll:.4f}")
            
            # Compute spectral metrics from hidden states
            if per_step_hidden:
                hidden_stack = torch.stack(per_step_hidden, dim=0)
                
                # Streaming covariance for streaming effective dimension
                streaming_cov = None
                alpha = streaming_cov_alpha
                
                for idx in range(len(per_step_hidden)):
                    h_t = per_step_hidden[idx]
                    
                    # Streaming covariance update
                    if streaming_cov is None:
                        streaming_cov = torch.outer(h_t, h_t)
                    else:
                        streaming_cov = alpha * streaming_cov + (1 - alpha) * torch.outer(h_t, h_t)
                    
                    # Streaming effective dimension
                    try:
                        s_evals = torch.linalg.eigvalsh(streaming_cov)
                        s_evals = torch.clamp(s_evals, min=0.0)
                        tot = s_evals.sum()
                        if tot > 0:
                            streaming_eff_dim = float(((tot ** 2) / ((s_evals ** 2).sum() + 1e-12)).item())
                        else:
                            streaming_eff_dim = 0.0
                    except Exception:
                        streaming_eff_dim = 0.0
                    
                    # Windowed spectral metrics
                    window_start = max(0, idx + 1 - 64)
                    hidden_window = hidden_stack[window_start:idx + 1]
                    spectral_entropy, effective_dim = compute_spectral_metrics(hidden_window)
                    
                    # Shannon effective rank (literature-standard metric)
                    shannon_er = compute_shannon_effective_rank(hidden_window)
                    
                    token_text = tokenizer.decode([generated_ids[idx]]) if idx < len(generated_ids) else ""
                    nll_val = per_step_nll[idx] if idx < len(per_step_nll) else 0.0
                    angle_val = per_step_angle[idx] if idx < len(per_step_angle) else 0.0
                    
                    writer.writerow({
                        "prompt_idx": pidx,
                        "prompt_text": prompt_text[:100],
                        "category": category,
                        "step": idx + 1,
                        "token": token_text,
                        "nll": f"{nll_val:.6f}",
                        "spectral_entropy": f"{spectral_entropy:.6f}",
                        "effective_dim": f"{effective_dim:.4f}",
                        "shannon_eff_rank": f"{shannon_er:.4f}",
                        "streaming_eff_dim": f"{streaming_eff_dim:.4f}",
                        "projection_angle": f"{angle_val:.6f}",
                        "hidden_norm": f"{h_t.norm().item():.4f}",
                        "gla_mean_norm": f"{per_step_gla_norm:.4f}",
                        "fox_mean_norm": f"{per_step_fox_norm:.4f}",
                    })
            
            # End-of-prompt cumulative Shannon effective rank
            if per_step_hidden:
                all_hidden = torch.stack(per_step_hidden, dim=0)
                cumulative_er = compute_shannon_effective_rank(all_hidden)
                print(f"  Cumulative Shannon Eff Rank: {cumulative_er:.1f} / 576 ({cumulative_er/576*100:.1f}%)")
    
    print(f"\n{'='*60}")
    print(f"Spectral profile saved to: {output_csv}")
    print(f"Metrics: spectral_entropy, effective_dim (PR), shannon_eff_rank, ")
    print(f"         streaming_eff_dim, projection_angle, hidden_norm,")
    print(f"         gla_mean_norm, fox_mean_norm")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Genesis-152M Spectral Profiling (Phase 0)")
    parser.add_argument("--weights", type=str, default=None, help="Path to safetensors weights")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/mps/cpu)")
    parser.add_argument("--prompts", type=str, default="prompts/prompts_60.json", help="Prompt file")
    parser.add_argument("--max-prompts", type=int, default=None, help="Limit number of prompts")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per generation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding")
    parser.add_argument("--output", type=str, default="logs/phase0_spectral_profile.csv", help="Output CSV")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Load model
    model, tokenizer, config = load_genesis_model(
        weights_path=args.weights,
        device=args.device,
    )
    
    # Load prompts
    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    
    if args.max_prompts:
        prompts = prompts[:args.max_prompts]
    
    print(f"\nLoaded {len(prompts)} prompts from {args.prompts}")
    
    # Run spectral profiling
    run_spectral_profile(
        model=model,
        tokenizer=tokenizer,
        config=config,
        prompts=prompts,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=not args.greedy,
        output_csv=args.output,
    )


if __name__ == "__main__":
    main()
