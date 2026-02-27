"""Core Spectral Microscope telemetry system."""

from typing import Any, Dict, List, Optional
import json
import logging
import torch
from transformers import PreTrainedTokenizerBase

from .analysis import compute_spectral_metrics, compute_top_eigenvalues

logger = logging.getLogger(__name__)

class SpectralMicroscope:
    """Telemetry system for tracking spectral properties during generation."""

    def __init__(
        self,
        max_tokens: int = 512,
        window_size: int = 64,
        top_k_eigenvalues: int = 5,
    ) -> None:
        """Initialize the Spectral Microscope.

        Args:
            max_tokens: Maximum tokens to analyze.
            window_size: Sliding window size for spectral metrics.
            top_k_eigenvalues: Number of eigenvalues to track.
        """
        self.max_tokens = max_tokens
        self.window_size = window_size
        self.top_k_eigenvalues = top_k_eigenvalues

    @torch.no_grad()
    def generate_and_analyze(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        system_prompt: str = "You are a helpful assistant.",
    ) -> Dict[str, Any]:
        """Generate response and capture spectral telemetry inline.

        Args:
            model: HuggingFace model instance.
            tokenizer: Tokenizer instance.
            prompt: User prompt.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            do_sample: Whether to sample or use greedy decoding.
            system_prompt: Optional system prompt context.

        Returns:
            Dict containing the 'response' text and 'telemetry' timeline.
        """
        # Prepare inputs
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            chat_input = f"{system_prompt}\n\nUser: {prompt}\n\nAssistant:"

        inputs = tokenizer(chat_input, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

        generated_ids: List[int] = []
        per_step_hidden: List[torch.Tensor] = []
        per_step_angle: List[float] = []
        
        past_key_values = None

        for step_idx in range(max_new_tokens):
            if step_idx == 0:
                step_input = input_ids
                step_mask = attention_mask
            else:
                step_input = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=model.device)
                if attention_mask is not None:
                    step_mask = torch.cat(
                        [attention_mask, torch.ones((1, 1), dtype=torch.long, device=model.device)],
                        dim=1,
                    )
                    attention_mask = step_mask
                else:
                    step_mask = None

            try:
                outputs = model(
                    input_ids=step_input,
                    attention_mask=step_mask,
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                )
            except Exception as e:
                logger.error(f"Generation failed at step {step_idx}: {e}")
                break

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            
            if do_sample and temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
            else:
                next_id = int(torch.argmax(logits, dim=-1).item())

            generated_ids.append(next_id)

            if next_id == tokenizer.eos_token_id:
                break

            # Capture hidden states
            if step_idx < self.max_tokens:
                hidden_states = getattr(outputs, "hidden_states", None)
                if hidden_states:
                    h_end = hidden_states[-1][0, -1].detach().cpu()
                    per_step_hidden.append(h_end)
                    
                    if len(hidden_states) > 2:
                        early = hidden_states[1][0, -1]
                        late = hidden_states[-2][0, -1]
                        angle = torch.nn.functional.cosine_similarity(early, late, dim=0).detach().cpu().item()
                        per_step_angle.append(float(angle))

        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        telemetry = []
        if per_step_hidden:
            hidden_stack = torch.stack(per_step_hidden, dim=0)
            
            streaming_cov = None
            alpha_ema = 0.95
            
            for idx in range(len(per_step_hidden)):
                h_t = hidden_stack[idx].float().cpu()
                if streaming_cov is None:
                    streaming_cov = torch.outer(h_t, h_t)
                else:
                    streaming_cov = alpha_ema * streaming_cov + (1 - alpha_ema) * torch.outer(h_t, h_t)
                    
                try:
                    streaming_evals = torch.linalg.eigvalsh(streaming_cov)
                    streaming_evals = torch.clamp(streaming_evals, min=0.0)
                    tot = streaming_evals.sum()
                    if tot > 0:
                        denom = (streaming_evals ** 2).sum() + 1e-12
                        streaming_eff_dim = float(((tot ** 2) / denom).item())
                    else:
                        streaming_eff_dim = 0.0
                except Exception:
                    streaming_eff_dim = 0.0

                if self.window_size <= 0:
                    hidden_window = hidden_stack[: idx + 1]
                else:
                    window_start = max(0, idx + 1 - self.window_size)
                    hidden_window = hidden_stack[window_start : idx + 1]
                    
                spectral_entropy, effective_dim = compute_spectral_metrics(hidden_window)
                token_text = tokenizer.decode([generated_ids[idx]], skip_special_tokens=False)
                
                angle_val = per_step_angle[idx] if idx < len(per_step_angle) else 0.0
                
                telemetry.append({
                    "step": idx + 1,
                    "token": token_text,
                    "spectral_entropy": spectral_entropy,
                    "effective_dim": effective_dim,
                    "streaming_eff_dim": streaming_eff_dim,
                    "projection_angle": angle_val,
                })

        return {
            "response": response_text,
            "telemetry": telemetry
        }
