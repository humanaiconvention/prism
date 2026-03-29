import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
from prism.lens.logit import LogitLens
from prism.lens.tuned import TunedLens, TunedLensTrainer

class DummyModel(nn.Module):
    def __init__(self, vocab_size=100, hidden_size=16):
        super().__init__()
        self.config = type('Config', (), {'vocab_size': vocab_size, 'hidden_size': hidden_size})()
        
        self.model = nn.Module()
        self.model.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)


class AccessorOnlyModel(nn.Module):
    def __init__(self, vocab_size=100, hidden_size=16):
        super().__init__()
        self.config = type('Config', (), {'vocab_size': vocab_size, 'hidden_size': hidden_size})()
        self.decoder = nn.Module()
        self.decoder.final_layer_norm = nn.LayerNorm(hidden_size)
        self._output_projection = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_output_embeddings(self):
        return self._output_projection


class Seq2SeqOutputModel(nn.Module):
    def __init__(self, vocab_size=100, hidden_size=16):
        super().__init__()
        self.config = type(
            'Config',
            (),
            {
                'vocab_size': vocab_size,
                'hidden_size': hidden_size,
                'num_hidden_layers': 2,
            },
        )()
        self.decoder = nn.Module()
        self.decoder.final_layer_norm = nn.LayerNorm(hidden_size)
        self._output_projection = nn.Linear(hidden_size, vocab_size, bias=False)

    def get_output_embeddings(self):
        return self._output_projection

    def forward(self, input_ids=None, output_hidden_states=False, return_dict=True, **kwargs):
        batch_size, seq_len = input_ids.shape if input_ids is not None else (1, 1)
        hidden_size = self.config.hidden_size
        token_basis = (
            input_ids.float().unsqueeze(-1).expand(batch_size, seq_len, hidden_size)
            if input_ids is not None
            else torch.zeros(batch_size, seq_len, hidden_size)
        )
        hidden_states = tuple(token_basis + (idx + 1) * 0.1 for idx in range(3))
        return SimpleNamespace(
            hidden_states=None,
            decoder_hidden_states=hidden_states,
            last_hidden_state=hidden_states[-1],
        )

def test_logit_lens():
    vocab_size = 100
    model = DummyModel(vocab_size=vocab_size, hidden_size=16)
    lens = LogitLens(model)
    
    batch_size = 2
    seq_len = 5
    hidden_size = 16
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    
    # 1. Project layer states
    probs = lens.project_layer_states(hidden_states)
    assert probs.shape == (batch_size, seq_len, vocab_size)
    assert torch.all(probs >= 0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, seq_len))
    
    # 2. Decode top k
    class DummyTokenizer:
        def decode(self, token_ids):
            return f"token_{token_ids[0]}"
            
    top_k_res = lens.decode_top_k(hidden_states, tokenizer=DummyTokenizer(), k=3, position_idx=-1)
    assert len(top_k_res) == 3
    assert isinstance(top_k_res[0][0], str)
    assert isinstance(top_k_res[0][1], float)
    
    # 3. Prediction entropy trajectory
    hidden_states_list = [torch.randn(batch_size, seq_len, hidden_size) for _ in range(4)]
    entropies = lens.get_prediction_entropy_trajectory(hidden_states_list, position_idx=-1)
    
    assert len(entropies) == 4
    for e in entropies:
        assert isinstance(e, float)
        assert e > 0.0


def test_logit_lens_uses_output_embedding_accessor_and_decoder_norm():
    model = AccessorOnlyModel(vocab_size=64, hidden_size=16)
    lens = LogitLens(model)

    hidden_states = torch.randn(2, 5, 16)
    probs = lens.project_layer_states(hidden_states)

    assert probs.shape == (2, 5, 64)
    assert torch.all(probs >= 0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(2, 5), atol=1e-5)


def test_tuned_lens_trainer_uses_decoder_hidden_states():
    model = Seq2SeqOutputModel(vocab_size=48, hidden_size=16)
    tuned_lens = TunedLens(hidden_size=16, num_layers=2)
    trainer = TunedLensTrainer(model, tuned_lens)

    loss = trainer.train_step({"input_ids": torch.tensor([[1, 2, 3]])})

    assert isinstance(loss, float)
    assert loss >= 0.0
