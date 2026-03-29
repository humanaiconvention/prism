"""Phase 9 reliability check: semantic-direction stability under prompt resampling."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import format_chatml_prompt, load_genesis_model
from scripts.run_phase9_extract import MultiLayerCaptureHook, get_category_indices
from scripts.run_phase9_semantic_dirs import isolate_semantic_direction


def parse_int_list(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def infer_pairwise_csv(output_csv):
    root, ext = os.path.splitext(output_csv)
    return f"{root}_pairwise{ext or '.csv'}"


def cosine_similarity(vec_a, vec_b):
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom <= 1e-12:
        return np.nan
    return float(np.dot(vec_a, vec_b) / denom)


def load_prompt_activations(model, tokenizer, prompts, layers):
    hook = MultiLayerCaptureHook(layers)
    hook.attach(model)
    acts = {layer: [] for layer in layers}
    for prompt_entry in tqdm(prompts, desc="Capturing prompts"):
        text = prompt_entry["text"] if isinstance(prompt_entry, dict) else prompt_entry
        chat_input = format_chatml_prompt(text)
        input_ids = torch.tensor([tokenizer.encode(chat_input)], device=next(model.parameters()).device)
        with torch.no_grad():
            model(input_ids)
        for layer in layers:
            acts[layer].append(hook.captured_acts[layer][-1][0].astype(np.float64))
        hook.clear()
    hook.remove()
    return {layer: np.stack(rows, axis=0) for layer, rows in acts.items()}


def bootstrap_indices(rng, population, sample_size):
    chosen = rng.choice(population, size=sample_size, replace=True)
    return np.asarray(chosen, dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description="Phase 9: Direction stability under prompt resampling")
    parser.add_argument("--prompts", type=str, default="prompts/prompts_200.json")
    parser.add_argument("--max-prompts", type=int, default=60)
    parser.add_argument("--layers", type=str, default="15,29")
    parser.add_argument("--n-resamples", type=int, default=24)
    parser.add_argument("--background-sample-size", type=int, default=None)
    parser.add_argument("--math-sample-size", type=int, default=None)
    parser.add_argument("--creative-sample-size", type=int, default=None)
    parser.add_argument("--k-bulk", type=int, default=70)
    parser.add_argument("--min-retained-fraction", type=float, default=0.10)
    parser.add_argument("--rank-tol", type=float, default=1e-8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-csv", type=str, default="logs/phase9/direction_stability_summary.csv")
    parser.add_argument("--pairwise-csv", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(Path(args.output_csv).parent, exist_ok=True)
    pairwise_csv = args.pairwise_csv or infer_pairwise_csv(args.output_csv)
    layers = parse_int_list(args.layers)

    with open(args.prompts, "r", encoding="utf-8") as f:
        prompts = json.load(f)["prompts"][:args.max_prompts]

    math_indices = [i for i in get_category_indices("Mathematical") if i < len(prompts)]
    creative_indices = [i for i in get_category_indices("Creative") if i < len(prompts)]
    if len(math_indices) < 2 or len(creative_indices) < 2 or len(prompts) < 2:
        raise ValueError("Need at least 2 math prompts, 2 creative prompts, and 2 total prompts for stability analysis.")

    background_sample_size = args.background_sample_size or len(prompts)
    math_sample_size = args.math_sample_size or len(math_indices)
    creative_sample_size = args.creative_sample_size or len(creative_indices)

    print("Loading Genesis-152M for direction stability...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)

    all_acts = load_prompt_activations(model, tokenizer, prompts, layers)
    rng = np.random.default_rng(args.seed)

    full_results = {}
    for layer in layers:
        acts = all_acts[layer]
        cov = np.cov(acts, rowvar=False)
        full_results[layer] = isolate_semantic_direction(
            cov=cov,
            math_centroid=acts[math_indices].mean(axis=0),
            creative_centroid=acts[creative_indices].mean(axis=0),
            k_bulk=args.k_bulk,
            min_retained_fraction=args.min_retained_fraction,
            rank_tol=args.rank_tol,
            n_samples=len(prompts),
        )

    detail_rows = []
    pairwise_rows = []
    resample_vectors = {(layer, vector_key): [] for layer in layers for vector_key in ("delta_raw", "delta_perp")}

    print(
        f"\n=== PHASE 9: DIRECTION STABILITY ===\n"
        f"Layers: {layers}\n"
        f"Prompts: {len(prompts)} total | {len(math_indices)} math | {len(creative_indices)} creative\n"
        f"Resamples: {args.n_resamples}"
    )

    for resample_id in range(args.n_resamples):
        bg_idx = bootstrap_indices(rng, len(prompts), background_sample_size)
        math_local = bootstrap_indices(rng, len(math_indices), math_sample_size)
        creative_local = bootstrap_indices(rng, len(creative_indices), creative_sample_size)
        math_idx = np.asarray([math_indices[i] for i in math_local], dtype=np.int64)
        creative_idx = np.asarray([creative_indices[i] for i in creative_local], dtype=np.int64)

        for layer in layers:
            acts = all_acts[layer]
            cov = np.cov(acts[bg_idx], rowvar=False)
            result = isolate_semantic_direction(
                cov=cov,
                math_centroid=acts[math_idx].mean(axis=0),
                creative_centroid=acts[creative_idx].mean(axis=0),
                k_bulk=args.k_bulk,
                min_retained_fraction=args.min_retained_fraction,
                rank_tol=args.rank_tol,
                n_samples=background_sample_size,
            )
            for vector_key, norm_key in (("delta_raw", "raw_norm"), ("delta_perp", "perp_norm")):
                vector = result[vector_key]
                resample_vectors[(layer, vector_key)].append(vector)
                detail_rows.append({
                    "layer": layer,
                    "vector_key": vector_key,
                    "resample_id": resample_id,
                    "norm": float(np.linalg.norm(vector)),
                    "cosine_to_full": cosine_similarity(vector, full_results[layer][vector_key]),
                    "reference_norm": float(full_results[layer][norm_key]),
                    "retained_fraction": float(result["retained_fraction"]),
                    "bulk_variance_explained": float(result["bulk_variance_explained"]),
                    "k_bulk_effective": int(result["k_bulk_effective"]),
                    "numerical_rank": int(result["numerical_rank"]),
                })

    for (layer, vector_key), vectors in resample_vectors.items():
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                pairwise_rows.append({
                    "layer": layer,
                    "vector_key": vector_key,
                    "resample_i": i,
                    "resample_j": j,
                    "cosine_similarity": cosine_similarity(vectors[i], vectors[j]),
                })

    detail_df = pd.DataFrame(detail_rows)
    pairwise_df = pd.DataFrame(pairwise_rows)
    summary_df = (
        detail_df.groupby(["layer", "vector_key"], as_index=False)
        .agg(
            mean_cosine_to_full=("cosine_to_full", "mean"),
            std_cosine_to_full=("cosine_to_full", "std"),
            mean_vector_norm=("norm", "mean"),
            std_vector_norm=("norm", "std"),
            mean_retained_fraction=("retained_fraction", "mean"),
            mean_bulk_variance_explained=("bulk_variance_explained", "mean"),
            mean_k_bulk_effective=("k_bulk_effective", "mean"),
            mean_numerical_rank=("numerical_rank", "mean"),
            n_resamples=("resample_id", "count"),
        )
    )
    pairwise_summary = (
        pairwise_df.groupby(["layer", "vector_key"], as_index=False)
        .agg(
            mean_pairwise_cosine=("cosine_similarity", "mean"),
            std_pairwise_cosine=("cosine_similarity", "std"),
            min_pairwise_cosine=("cosine_similarity", "min"),
            max_pairwise_cosine=("cosine_similarity", "max"),
            n_pairs=("cosine_similarity", "count"),
        )
    )
    summary_df = summary_df.merge(pairwise_summary, on=["layer", "vector_key"], how="left")
    summary_df["background_sample_size"] = background_sample_size
    summary_df["math_sample_size"] = math_sample_size
    summary_df["creative_sample_size"] = creative_sample_size

    summary_df.to_csv(args.output_csv, index=False)
    pairwise_df.to_csv(pairwise_csv, index=False)
    detail_df.to_csv(pairwise_csv.replace("_pairwise", "_detail"), index=False)

    print("\n--- DIRECTION STABILITY SUMMARY ---")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to {args.output_csv}")
    print(f"Pairwise detail saved to {pairwise_csv}")
    print(f"Resample detail saved to {pairwise_csv.replace('_pairwise', '_detail')}")


if __name__ == "__main__":
    main()