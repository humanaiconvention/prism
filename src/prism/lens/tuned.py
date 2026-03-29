"""Tuned Lens for learned affine translations to vocabulary."""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any, Optional
from ..architecture import TransformerArchitectureAdapter

class TunedLens(nn.Module):
    """Trainable affine translators for layer representations."""
    
    def __init__(self, hidden_size: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        # List of translators (one per layer)
        self.translators = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) 
            for _ in range(num_layers)
        ])
        
    def forward(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Translates a hidden state from layer_idx to the final layer basis."""
        # Ensure dtype match (e.g., if input is float16 and lens is float32)
        translator = self.translators[layer_idx]
        if translator.weight.dtype != x.dtype:
            translator = translator.to(x.dtype)
        return translator(x)

class TunedLensTrainer:
    """Trainer for the TunedLens using distillation loss."""
    
    def __init__(
        self,
        model: nn.Module,
        tuned_lens: TunedLens,
        lr: float = 1e-3,
        adapter: Optional[TransformerArchitectureAdapter] = None,
    ):
        self.model = model
        self.lens = tuned_lens
        self.adapter = adapter or TransformerArchitectureAdapter(model)
        self.optimizer = optim.Adam(self.lens.parameters(), lr=lr)
        self.criterion = nn.KLDivLoss(reduction="batchmean")

    def train_step(self, inputs: Dict[str, torch.Tensor]) -> float:
        """Trains the lens for one step using the model's own hidden states."""
        self.model.eval()
        self.lens.train()
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
            layer_states = self._extract_layer_states(outputs)
            if len(layer_states) < 2:
                raise ValueError("Tuned lens training requires at least two hidden-state layers.")
            # Final hidden state is our target
            target_state = layer_states[-1]
            # Get target distribution
            final_norm = self.adapter.resolve_final_norm() or nn.Identity()
            lm_head = self.adapter.resolve_lm_head()
            if lm_head is None:
                raise ValueError("Could not automatically locate the unembedding head for tuned lens training.")
            target_logits = lm_head(final_norm(target_state))
            target_log_probs = torch.log_softmax(target_logits, dim=-1)

        total_loss = 0
        for i in range(len(layer_states) - 1):
            self.optimizer.zero_grad()
            
            # 1. Translate layer state
            h_l = layer_states[i]
            h_translated = self.lens(i, h_l)
            
            # 2. Unembed
            translated_logits = lm_head(final_norm(h_translated))
            translated_log_probs = torch.log_softmax(translated_logits, dim=-1)
            
            # 3. KL Divergence loss against the final layer's prediction
            # target_log_probs is the 'truth' we want to approximate
            loss = self.criterion(translated_log_probs, target_log_probs.exp())
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            
        return total_loss / (len(layer_states) - 1)

    @staticmethod
    def _extract_layer_states(outputs: Any) -> List[torch.Tensor]:
        """Extracts the decoder or main hidden-state stack across model families."""
        for attr in ("decoder_hidden_states", "hidden_states"):
            states = getattr(outputs, attr, None)
            if states is not None:
                return list(states)
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if isinstance(last_hidden_state, torch.Tensor):
            return [last_hidden_state]
        raise ValueError(
            "Model outputs did not expose decoder_hidden_states, hidden_states, or last_hidden_state."
        )
