import math
from typing import Tuple, Optional

def select_prompt_window_bounds(seq_len: int, window_scope: str) -> Tuple[int, int]:
    """Calculates start and end token indices for the prompt prefill window."""
    seq_len = int(seq_len)
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")

    if window_scope == "prompt_prefill_only":
        return 0, seq_len
    if window_scope == "prompt_last_token_only":
        return seq_len - 1, seq_len
    if window_scope == "prompt_first_half_only":
        return 0, max(1, seq_len // 2)
    if window_scope == "prompt_second_half_only":
        return seq_len // 2, seq_len
    if window_scope == "prompt_last_quarter_only":
        width = max(1, int(math.ceil(seq_len / 4.0)))
        return seq_len - width, seq_len
    raise ValueError(f"Unsupported window_scope: {window_scope}")

def select_generation_window_bounds(total_generation_calls: int, generation_scope: str) -> Tuple[int, int]:
    """Calculates start and end indices for the generation step window."""
    total_generation_calls = int(total_generation_calls)
    if total_generation_calls <= 0:
        raise ValueError("total_generation_calls must be positive")

    if generation_scope == "generation_only":
        return 0, total_generation_calls
    if generation_scope == "generation_first_step_only":
        return 0, 1
    if generation_scope == "generation_first_half_only":
        return 0, max(1, total_generation_calls // 2)
    if generation_scope == "generation_second_half_only":
        return total_generation_calls // 2, total_generation_calls
    if generation_scope == "generation_first_quarter_only":
        width = max(1, int(math.ceil(total_generation_calls / 4.0)))
        return 0, width
    raise ValueError(f"Unsupported generation_scope: {generation_scope}")

def resolve_composition_window(condition: str) -> Tuple[str, Optional[str]]:
    """Resolves a complex composition window into its prompt and generation parts."""
    if condition == "full_sequence":
        return "prompt_prefill_only", "generation_only"
    if condition == "prompt_second_half_only":
        return "prompt_second_half_only", None
    if condition == "prompt_second_half_plus_generation_only":
        return "prompt_second_half_only", "generation_only"
    if condition == "prompt_second_half_plus_generation_first_half_only":
        return "prompt_second_half_only", "generation_first_half_only"
    if condition == "prompt_second_half_plus_generation_second_half_only":
        return "prompt_second_half_only", "generation_second_half_only"
    if condition == "prompt_second_half_plus_generation_first_quarter_only":
        return "prompt_second_half_only", "generation_first_quarter_only"
    raise ValueError(f"Unsupported composition window condition: {condition}")
