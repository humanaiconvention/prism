import pytest
import torch
from unittest.mock import MagicMock, patch
from prism.discovery.scout import CircuitScout

def test_circuit_scout_ioi_mock():
    """
    Tests CircuitScout on a mocked IOI task.
    Verifies that it identifies 'Name' heads (heads with simulated high effect).
    """
    # 1. Mock model and tokenizer
    model = MagicMock()
    model.device = "cpu"
    tokenizer = MagicMock()
    
    # Mock tokenization
    def encode_side_effect(text, **kwargs):
        if len(text.split()) > 1: # Prompt
            return torch.tensor([[1, 2, 3, 4, 5]])
        if text == "Bob": return [10]
        if text == "Alice": return [11]
        return [0]
    
    tokenizer.encode.side_effect = encode_side_effect
    
    # Mock attention modules
    # We'll simulate 2 layers with 2 heads each
    layer0 = MagicMock()
    layer0.num_heads = 2
    layer1 = MagicMock()
    layer1.num_heads = 2
    
    model.named_modules.return_value = [
        ("layer.0.attn", layer0),
        ("layer.1.attn", layer1)
    ]
    
    # 2. Setup Baseline Logits
    # target_id = 10, foil_id = 11 (not returned by encode above but we'll manage)
    # We'll just mock the model() call
    clean_output = MagicMock()
    clean_output.logits = torch.zeros((1, 3, 20))
    clean_output.logits[0, -1, 10] = 5.0 # Target high in clean
    clean_output.logits[0, -1, 11] = 1.0 # Foil low in clean
    
    corrupt_output = MagicMock()
    corrupt_output.logits = torch.zeros((1, 3, 20))
    corrupt_output.logits[0, -1, 10] = 1.0 # Target low in corrupt
    corrupt_output.logits[0, -1, 11] = 1.0 # Foil low in corrupt
    
    # Side effect for model calls
    # 1st: clean run, 2nd: corrupt run, then patched runs
    # We'll use a more robust side effect
    call_count = 0
    def model_side_effect(*args, **kwargs):
        nonlocal call_count
        res = MagicMock()
        res.logits = torch.zeros((1, 5, 20)) # seq_len=5
        
        if call_count == 0: # Clean
            res.logits[0, -1, 10] = 5.0
            res.logits[0, -1, 11] = 1.0
        elif call_count == 1: # Corrupt
            res.logits[0, -1, 10] = 1.0
            res.logits[0, -1, 11] = 1.0
        else: # Patched runs
            # We'll simulate Layer 1 Head 0 as the 'Name' head
            # When patched, it should restore some logit diff
            # The current head being patched is determined by ActivationPatcher hooks
            # But here we are mocking at the model level.
            # In CircuitScout, it loops through layers and heads.
            # Total heads = 2 * 2 = 4. 
            # call_count indices for patched runs: 2, 3, 4, 5
            if call_count == 4: # Simulated Layer 1 Head 0
                res.logits[0, -1, 10] = 4.0 # Restored diff
                res.logits[0, -1, 11] = 1.0
            else:
                res.logits[0, -1, 10] = 1.0
                res.logits[0, -1, 11] = 1.0
                
        call_count += 1
        return res

    model.side_effect = model_side_effect

    # 3. Mock ActivationPatcher.get_activations
    # It just needs to return something for the dict keys
    with patch("prism.discovery.scout.ActivationPatcher") as MockPatcher:
        mock_patcher_inst = MockPatcher.return_value
        mock_patcher_inst.get_activations.return_value = {
            "layer.0.attn": torch.randn(1, 3, 10),
            "layer.1.attn": torch.randn(1, 3, 10)
        }
        
        # mock_patcher_inst.run_patched should return logits
        def run_patched_mock(*args, **kwargs):
            return model_side_effect(*args, **kwargs).logits
        
        mock_patcher_inst.run_patched.side_effect = run_patched_mock
        mock_patcher_inst._get_module_by_name.side_effect = lambda name: layer0 if "0" in name else layer1

        scout = CircuitScout(model, tokenizer)
        # Manually set the patcher to our mock instance
        scout.patcher = mock_patcher_inst
        
        results = scout.discover_circuit(
            clean_prompt="Then, Alice and Bob went to the store. Alice gave a gift to",
            corrupt_prompt="Then, Alice and Bob went to the store. Bob gave a gift to",
            target_token="Bob",
            foil_token="Alice",
            top_k=2
        )
        
        assert "top_heads" in results
        assert len(results["top_heads"]) > 0
        # The top head should be layer.1.attn.h0 based on our mock logic
        assert results["top_heads"][0]["head"] == "layer.1.attn.h0"
        assert results["top_heads"][0]["effect"] > 0.5
