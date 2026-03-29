import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import FOX_LAYER_INDICES, HiddenStateHook, load_genesis_model
from scripts.phase10_experiment_utils import (
    compute_representation_metrics,
    configure_matplotlib,
    ensure_parent_dir,
    infer_companion_csv,
    infer_companion_png,
    layer_type_name,
    mean_pairwise_metrics,
)
from scripts.run_phase9_recurrent_state_patching import reset_model_decode_state
from scripts.run_phase9_semantic_steering import load_eval_items
from scripts.run_phase9_token_position_steering import prepare_eval_item


def unpack_logits(model_output):
    return model_output[0] if isinstance(model_output, tuple) else model_output


def collect_hidden_states(model, hidden_hook, prompt_ids):
    reset_model_decode_state(model)
    hidden_hook.clear()
    with torch.inference_mode():
        _ = unpack_logits(model(prompt_ids))
    states = [tensor.squeeze(0).detach().float().cpu().numpy() for tensor in hidden_hook.get_hidden_states()]
    reset_model_decode_state(model)
    return states


def bootstrap_contrast(layer_to_matrix, layer_a, layer_b, metric_key, n_boot=2000, seed=0):
    a = np.asarray(layer_to_matrix[layer_a], dtype=np.float64)
    b = np.asarray(layer_to_matrix[layer_b], dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Bootstrap contrast requires same matrix shapes; got {a.shape} vs {b.shape}")

    def metric_value(matrix, key):
        metrics = compute_representation_metrics(matrix)
        pairwise = mean_pairwise_metrics(matrix)
        merged = {**metrics, **pairwise}
        if key not in merged:
            raise KeyError(f"Unknown geometry metric for bootstrap contrast: {key}")
        return merged[key]

    rng = np.random.default_rng(int(seed))
    observed = metric_value(a, metric_key) - metric_value(b, metric_key)
    samples = []
    n = a.shape[0]
    for _ in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        diff = metric_value(a[idx], metric_key) - metric_value(b[idx], metric_key)
        samples.append(diff)
    samples = np.asarray(samples, dtype=np.float64)
    pvalue = float(min(1.0, 2.0 * min(np.mean(samples <= 0.0), np.mean(samples >= 0.0))))
    return {
        "metric": metric_key,
        "layer_a": int(layer_a),
        "layer_b": int(layer_b),
        "observed_difference": float(observed),
        "ci95_low": float(np.percentile(samples, 2.5)),
        "ci95_high": float(np.percentile(samples, 97.5)),
        "bootstrap_pvalue": pvalue,
    }


def save_plot(summary_df, output_path):
    plt = configure_matplotlib()
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    layers = summary_df["layer"].to_numpy()
    fox_layers = set(FOX_LAYER_INDICES)
    for ax in axes:
        for fox_layer in fox_layers:
            ax.axvline(fox_layer, color="gray", linestyle="--", alpha=0.15)
    axes[0].plot(layers, summary_df["participation_ratio"], marker="o", label="participation_ratio")
    axes[0].plot(layers, summary_df["normalized_spectral_entropy"], marker="s", label="normalized_spectral_entropy")
    axes[0].set_ylabel("Geometry")
    axes[0].set_title("10I geometry profile")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.2)
    axes[1].plot(layers, summary_df["pc1_explained_variance_ratio"], marker="o", label="PC1")
    axes[1].plot(layers, summary_df["pc5_cumulative_explained_variance_ratio"], marker="s", label="PC5 cumulative")
    axes[1].plot(layers, summary_df["pc10_cumulative_explained_variance_ratio"], marker="^", label="PC10 cumulative")
    axes[1].set_ylabel("Explained variance")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.2)
    axes[2].plot(layers, summary_df["mean_pairwise_cosine_distance"], marker="o", label="pairwise cosine distance")
    axes[2].plot(layers, summary_df["mean_feature_variance"], marker="s", label="mean feature variance")
    axes[2].plot(layers, summary_df["mean_activation_norm"], marker="^", label="mean activation norm")
    axes[2].set_ylabel("Distance / variance")
    axes[2].set_xlabel("Layer")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.2)
    fig.tight_layout()
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Phase 10I: descriptive layerwise geometry profile")
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--output-csv", type=str, default="logs/phase10/geometry_profile_summary.csv")
    parser.add_argument("--pairwise-csv", type=str, default=None)
    parser.add_argument("--stats-csv", type=str, default=None)
    parser.add_argument("--plot-path", type=str, default=None)
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, _ = load_genesis_model(device=device)
    hidden_hook = HiddenStateHook.attach_to_model(model)
    items = load_eval_items(args.eval_json)
    if args.max_eval_items is not None:
        items = items[: args.max_eval_items]
    prepared = [prepare_eval_item(tokenizer, item, device) for item in items]
    layer_vectors = {layer: [] for layer in range(30)}
    labels = []

    try:
        for item in prepared:
            states = collect_hidden_states(model, hidden_hook, item["prompt_ids"])
            for layer, vec in enumerate(states):
                layer_vectors[layer].append(vec)
            labels.append(item["item"]["label"].strip().lower())
    finally:
        hidden_hook.remove_all()
        reset_model_decode_state(model)

    labels = np.asarray(labels)
    summary_rows = []
    pairwise_rows = []
    stats_rows = []
    layer_to_matrix = {layer: np.stack(vectors, axis=0) for layer, vectors in layer_vectors.items()}
    for layer in range(30):
        matrix = layer_to_matrix[layer]
        metrics = compute_representation_metrics(matrix)
        pairwise = mean_pairwise_metrics(matrix)
        summary_rows.append(
            {
                "layer": layer,
                "layer_type": layer_type_name(layer, FOX_LAYER_INDICES),
                "n_items": int(matrix.shape[0]),
                **metrics,
                **pairwise,
            }
        )
        for subset_name, subset_mask in (
            ("all", np.ones(labels.shape[0], dtype=bool)),
            ("math", labels == "math"),
            ("creative", labels == "creative"),
        ):
            subset_matrix = matrix[subset_mask]
            subset_pairwise = mean_pairwise_metrics(subset_matrix)
            pairwise_rows.append(
                {
                    "layer": layer,
                    "layer_type": layer_type_name(layer, FOX_LAYER_INDICES),
                    "subset": subset_name,
                    "n_items": int(subset_matrix.shape[0]),
                    **subset_pairwise,
                }
            )

    for metric_key in ["participation_ratio", "normalized_spectral_entropy", "mean_pairwise_cosine_distance", "mean_feature_variance"]:
        for other_layer in [7, 15]:
            stats_rows.append(bootstrap_contrast(layer_to_matrix, 11, other_layer, metric_key, n_boot=args.bootstrap_samples, seed=1000 + other_layer))

    summary_df = pd.DataFrame(summary_rows).sort_values("layer")
    pairwise_df = pd.DataFrame(pairwise_rows).sort_values(["layer", "subset"])
    stats_df = pd.DataFrame(stats_rows)
    output_path = args.output_csv
    pairwise_path = args.pairwise_csv or infer_companion_csv(output_path, "pairwise")
    stats_path = args.stats_csv or infer_companion_csv(output_path, "stats")
    plot_path = args.plot_path or infer_companion_png(output_path, "summary")
    ensure_parent_dir(output_path)
    summary_df.to_csv(output_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)
    stats_df.to_csv(stats_path, index=False)
    save_plot(summary_df, plot_path)
    print(summary_df.to_string(index=False))
    if not stats_df.empty:
        print("\n[stats]")
        print(stats_df.to_string(index=False))
    print(f"[saved] summary -> {output_path}")
    print(f"[saved] pairwise -> {pairwise_path}")
    print(f"[saved] stats -> {stats_path}")
    print(f"[saved] plot -> {plot_path}")


if __name__ == "__main__":
    main()