import pytest
import math

from prism.phase import (
    select_prompt_window_bounds,
    select_generation_window_bounds,
    resolve_composition_window
)

# --- Pytest Tests ---

def test_select_prompt_window_bounds():
    seq_len = 10
    assert select_prompt_window_bounds(seq_len, "prompt_prefill_only") == (0, 10)
    assert select_prompt_window_bounds(seq_len, "prompt_last_token_only") == (9, 10)
    assert select_prompt_window_bounds(seq_len, "prompt_first_half_only") == (0, 5)
    assert select_prompt_window_bounds(seq_len, "prompt_second_half_only") == (5, 10)
    assert select_prompt_window_bounds(seq_len, "prompt_last_quarter_only") == (7, 10) # ceil(10/4) = 3 -> 10-3 = 7

    seq_len = 3
    assert select_prompt_window_bounds(seq_len, "prompt_first_half_only") == (0, 1) # max(1, 3//2) -> max(1, 1) = 1
    assert select_prompt_window_bounds(seq_len, "prompt_second_half_only") == (1, 3) 
    assert select_prompt_window_bounds(seq_len, "prompt_last_quarter_only") == (2, 3) # ceil(3/4) = 1 -> 3-1 = 2

def test_select_generation_window_bounds():
    total_calls = 10
    assert select_generation_window_bounds(total_calls, "generation_only") == (0, 10)
    assert select_generation_window_bounds(total_calls, "generation_first_step_only") == (0, 1)
    assert select_generation_window_bounds(total_calls, "generation_first_half_only") == (0, 5)
    assert select_generation_window_bounds(total_calls, "generation_second_half_only") == (5, 10)
    assert select_generation_window_bounds(total_calls, "generation_first_quarter_only") == (0, 3) # ceil(10/4) = 3

    total_calls = 3
    assert select_generation_window_bounds(total_calls, "generation_first_half_only") == (0, 1) # max(1, 3//2)
    assert select_generation_window_bounds(total_calls, "generation_second_half_only") == (1, 3)
    assert select_generation_window_bounds(total_calls, "generation_first_quarter_only") == (0, 1) # ceil(3/4) = 1

def test_resolve_composition_window():
    assert resolve_composition_window("full_sequence") == ("prompt_prefill_only", "generation_only")
    assert resolve_composition_window("prompt_second_half_plus_generation_first_quarter_only") == ("prompt_second_half_only", "generation_first_quarter_only")
    assert resolve_composition_window("prompt_second_half_only") == ("prompt_second_half_only", None)

def test_invalid_parameters():
    with pytest.raises(ValueError, match="positive"):
        select_prompt_window_bounds(0, "prompt_prefill_only")
        
    with pytest.raises(ValueError, match="Unsupported"):
        select_prompt_window_bounds(10, "unknown_scope")
        
    with pytest.raises(ValueError, match="positive"):
        select_generation_window_bounds(0, "generation_only")
        
    with pytest.raises(ValueError, match="Unsupported"):
        resolve_composition_window("bad_condition")
