import argparse
import os
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.genesis_loader import load_genesis_model, format_chatml_prompt

def get_covariance(X):
    """X shape (N, D), returns (D, D) covariance"""
    # zero-mean
    X_m = X - X.mean(dim=0, keepdim=True)
    return (X_m.T @ X_m) / (X.shape[0] - 1)

def generate_random_prompt(length):
    vocab_size = 50257
    return torch.randint(100, vocab_size - 100, (1, length))


def extract_recurrent_state_trajectory(model, input_ids, target_layer, gen_steps, shape_log_path=None, prompt_tag=None):
    """Extract recurrent states from model.get_segment_states() over a cached generation loop.

    Returns:
        Tensor of shape (gen_steps + 1, flat_dim) containing the flattened recurrent
        state at the target layer after each forward step.
    """
    past_key_values = None
    curr_ids = input_ids
    trajectory = []
    shape_lines = []

    with torch.no_grad():
        for step in range(gen_steps + 1):
            idx_cond = curr_ids if past_key_values is None else curr_ids[:, -1:]
            logits, loss, metrics, past_key_values = model(
                idx_cond,
                past_key_values=past_key_values,
                use_cache=True,
            )

            states = model.get_segment_states()
            if not isinstance(states, dict):
                raise RuntimeError(f"model.get_segment_states() returned {type(states)}, expected dict.")
            if target_layer not in states:
                available = sorted(states.keys())
                raise RuntimeError(
                    f"Layer {target_layer} not found in segment states. Available layers: {available}"
                )

            state_tensor = states[target_layer].detach().float().cpu()
            flat_state = state_tensor[0].reshape(-1).clone()
            trajectory.append(flat_state)

            shape_lines.append(
                f"prompt={prompt_tag or 'unknown'} step={step} raw_shape={tuple(state_tensor.shape)} "
                f"flat_dim={flat_state.numel()}"
            )

            if step == gen_steps:
                break

            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            curr_ids = torch.cat([curr_ids, next_token], dim=1)

    if shape_log_path is not None:
        os.makedirs(os.path.dirname(shape_log_path), exist_ok=True)
        with open(shape_log_path, "a", encoding="utf-8") as f:
            for line in shape_lines:
                f.write(line + "\n")

    return torch.stack(trajectory, dim=0)


def encode_text_prompt(tokenizer, text, device):
    formatted = format_chatml_prompt(text)
    token_ids = tokenizer.encode(formatted)
    return torch.tensor([token_ids], device=device)

def main():
    parser = argparse.ArgumentParser(description="Phase 8A limit-cycle probe using model.get_segment_states().")
    parser.add_argument("--layer", type=int, default=14, help="Target GLA layer to probe.")
    parser.add_argument("--n-prompts", type=int, default=50, help="Number of random prompts for basis extraction.")
    parser.add_argument("--context-length", type=int, default=512, help="Random prompt length for basis extraction.")
    parser.add_argument("--gen-steps", type=int, default=16, help="Additional greedy generation steps after the initial prompt.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for prompt generation.")
    parser.add_argument(
        "--skip-projection-tests",
        action="store_true",
        help="Only verify state extraction / basis construction, skip syntax/semantic projection prompts.",
    )
    args = parser.parse_args()

    print("Loading Genesis-152M for Phase 8A: Limit-Cycle Variable Extraction...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_genesis_model(device=device)

    target_layer = args.layer
    n_prompts = args.n_prompts
    context_length = args.context_length
    gen_steps = args.gen_steps
    shape_log_path = "logs/phase8a_state_shapes.txt"

    os.makedirs("logs", exist_ok=True)
    with open(shape_log_path, "w", encoding="utf-8") as f:
        f.write(f"Phase 8A state-shape log for layer {target_layer}\n")
        f.write(f"n_prompts={n_prompts} context_length={context_length} gen_steps={gen_steps}\n")

    # 1. EXTRACT THE 13D LIMIT CYCLE BASIS
    print(f"\n[Step 1] Driving model to T={context_length + gen_steps} to reach limit cycle saturation...")

    final_states = []

    torch.manual_seed(args.seed)
    for pidx in tqdm(range(n_prompts), desc="Extracting asymptotic states"):
        input_ids = generate_random_prompt(context_length).to(device)
        trajectory = extract_recurrent_state_trajectory(
            model=model,
            input_ids=input_ids,
            target_layer=target_layer,
            gen_steps=gen_steps,
            shape_log_path=shape_log_path,
            prompt_tag=f"random_{pidx}",
        )
        final_states.append(trajectory[-1].unsqueeze(0))

    if not final_states:
        print(f"ERROR: Could not capture recurrent state from model.get_segment_states() at layer {target_layer}.")
        return

    final_states = torch.cat(final_states, dim=0)  # (N, flat_dim)
    print(f"Captured asymptotic states manifold: {final_states.shape}")

    cov = get_covariance(final_states)
    eigvals, eigvecs = torch.linalg.eigh(cov)

    # Sort descending
    eigvals = eigvals.flip(dims=(0,))
    eigvecs = eigvecs.flip(dims=(1,))

    # Keep the top 13 dimensions (the limit cycle)
    limit_cycle_basis = eigvecs[:, :13]
    variance_explained = eigvals[:13].sum() / eigvals.sum().clamp(min=1e-10)
    print(f"Verified recurrent state access at layer {target_layer} via model.get_segment_states().")
    print(f"Extracted 13D limit-cycle basis. Variance explained: {variance_explained:.1%}")

    # 2. PROJECTION TESTING (Syntax vs Semantics)
    if args.skip_projection_tests:
        print("\nSkipping projection tests (--skip-projection-tests set).")
        print("Phase 8A Step 1 smoke test complete. State access verified.")
        return

    print("\n[Step 2] Projecting domain prompts onto the Limit-Cycle...")

    # Define our two families again
    prompts_syntax = [
        "The quick brown fox jumps over the lazy dog.",
        "While they were walking to the store, it suddenly started raining.",
        "Because of the severe weather warning, all schools in the district were closed.",
        "If you want to understand the algorithm, you must first read the documentation.",
        "Although she tried her best, the final result did not meet the expectations."
    ]

    prompts_semantics = [
        "Quantum superposition logic gates",
        "Melancholy winter sunset orchestra",
        "Recursive neural network optimization",
        "Existential dread existentialism philosopher",
        "Galactic empire spaceship armada"
    ]

    def get_projection_magnitude(text):
        input_ids = encode_text_prompt(tokenizer, text, device)
        trajectory = extract_recurrent_state_trajectory(
            model=model,
            input_ids=input_ids,
            target_layer=target_layer,
            gen_steps=gen_steps,
            shape_log_path=shape_log_path,
            prompt_tag=text[:32].replace("\n", " "),
        )
        final_state = trajectory[-1].unsqueeze(0)

        # Project onto the 13D basis
        # magnitude of the vector projected onto the subspace
        proj = final_state @ limit_cycle_basis  # (1, 13)
        return torch.norm(proj).item(), torch.norm(final_state).item()

    print("\nSyntax-Heavy Prompts:")
    for p in prompts_syntax:
        proj_mag, total_mag = get_projection_magnitude(p)
        print(f"  '{p[:40]}...' | LC-Proj: {proj_mag:8.2f} / {total_mag:8.2f} ({proj_mag/total_mag:.1%})")

    print("\nSemantics-Heavy Prompts (Low Syntax):")
    for p in prompts_semantics:
        proj_mag, total_mag = get_projection_magnitude(p)
        print(f"  '{p[:40]}...' | LC-Proj: {proj_mag:8.2f} / {total_mag:8.2f} ({proj_mag/total_mag:.1%})")

    print("\nPhase 8A Step 1/2 complete. Basis established.")

if __name__ == "__main__":
    main()
