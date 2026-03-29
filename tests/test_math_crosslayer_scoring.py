import pytest
import torch
import torch.nn.functional as F

from prism.entropy import (
    unpack_logits_and_cache,
    compute_sequence_nll_tokenwise,
    compute_sequence_nll_tokenwise_with_traces,
)

# --- Pytest Tests ---

class MockLMOutputs:
    def __init__(self, logits, past_key_values="mock_cache"):
        self.logits = logits
        self.past_key_values = past_key_values

class MockModel(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, input_ids, use_cache=False, past_key_values=None):
        batch, seq_len = input_ids.shape
        # Return deterministic logits: for each token, logit[t] is high for t+1
        logits = torch.zeros(batch, seq_len, self.vocab_size)
        
        for b in range(batch):
            for i in range(seq_len):
                # The "correct" next token is input_ids[b, i] + 1 (just to have a deterministic pattern)
                expected_next = (input_ids[b, i] + 1).item() % self.vocab_size
                logits[b, i, expected_next] = 20.0 # High confidence
                
        return MockLMOutputs(logits)

def test_compute_sequence_nll_tokenwise():
    vocab_size = 10
    model = MockModel(vocab_size)
    
    # Prompt: [0, 1, 2, 3] -> Mock predicts: [1, 2, 3, 4]
    # NLL targets are [1, 2, 3], which perfectly match predictions
    prompt_ids = torch.tensor([[0, 1, 2, 3]])
    
    mean_nll = compute_sequence_nll_tokenwise(model, prompt_ids, prompt_len=2)
    
    # With logit=20 for correct and 0 for others, softmax ~ 1.0, log_softmax ~ 0.0, so NLL ~ 0.0
    assert torch.isclose(torch.tensor(mean_nll), torch.tensor(0.0), atol=1e-4)

    # Now provide an unexpected sequence: [0, 1, 9, 9] (mock still predicts [1, 2, 3, 4] based on preceding inputs)
    prompt_ids_bad = torch.tensor([[0, 1, 9, 8]])
    
    mean_nll_bad = compute_sequence_nll_tokenwise(model, prompt_ids_bad, prompt_len=2)
    # The actual tokens (9, 8) do not match the expected (2, 3), so NLL will be ~20.0
    assert mean_nll_bad > 10.0 

def test_compute_sequence_nll_tokenwise_with_traces():
    vocab_size = 10
    model = MockModel(vocab_size)
    prompt_ids = torch.tensor([[0, 1, 2, 3]])
    
    mean_nll, nlls, traces = compute_sequence_nll_tokenwise_with_traces(
        model, prompt_ids, prompt_len=2, top_k=3
    )
    
    assert len(nlls) == 3
    assert len(traces) == 3
    
    # Trace 0 is at pos 1 (target=1)
    assert traces[0]["token_position"] == 1
    assert traces[0]["gold_token_id"] == 1
    assert torch.isclose(torch.tensor(traces[0]["token_nll"]), torch.tensor(0.0), atol=1e-4)
    assert traces[0]["topk_token_ids"][0] == 1 # 1 has highest prob

def test_traces_with_baseline():
    vocab_size = 10
    model = MockModel(vocab_size)
    prompt_ids = torch.tensor([[0, 1, 2, 3]])
    
    _, _, baseline_traces = compute_sequence_nll_tokenwise_with_traces(
        model, prompt_ids, prompt_len=3, top_k=3
    )
    
    # Fake a perturbed run by passing the baseline traces back in
    # Here the model behaviour is deterministic so it exactly matches the baseline
    mean_nll, nlls, traces = compute_sequence_nll_tokenwise_with_traces(
        model, prompt_ids, prompt_len=3, baseline_traces=baseline_traces, top_k=3
    )
    
    assert len(traces) == 3
    assert "baseline_token_nll" in traces[0]
    assert "delta_token_nll" in traces[0]
    assert "delta_topk_logprobs" in traces[0]
    
    # Delta should be ~0 since model is same
    assert torch.isclose(torch.tensor(traces[0]["delta_token_nll"]), torch.tensor(0.0), atol=1e-4)
    # Delta logprobs should be a list of ~0s
    for diff in traces[0]["delta_topk_logprobs"]:
        assert torch.isclose(torch.tensor(diff), torch.tensor(0.0), atol=1e-4)
