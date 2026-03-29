"""Phase 14: localize the historical L15 bottleneck across attn_output vs ffn_output."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import load_genesis_model
from scripts.phase10_experiment_utils import ensure_parent_dir, infer_companion_csv, paired_signflip_test, signflip_test
from scripts.phase10_site_hooks import extract_output_tensor, parse_site_list, replace_output_tensor, resolve_site_module
from scripts.run_phase9_recurrent_state_patching import reset_model_decode_state
from scripts.run_phase9_semantic_steering import load_eval_items, unpack_logits
from scripts.run_phase9_token_position_steering import prepare_eval_item


def pair_name_from_item_name(item_name):
    return item_name.rsplit("__", 1)[0]


def phase14_output_paths(output_csv):
    return {
        "summary": output_csv,
        "detail": infer_companion_csv(output_csv, "detail"),
        "pair_effect": infer_companion_csv(output_csv, "pair_effect"),
        "stats": infer_companion_csv(output_csv, "stats"),
        "pca": infer_companion_csv(output_csv, "pca"),
    }


def score_choice_logits(logits, prepared_item):
    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
    math_lp = float(log_probs[0, int(prepared_item["math_token_id"])].item())
    creative_lp = float(log_probs[0, int(prepared_item["creative_token_id"])].item())
    margin = math_lp - creative_lp
    label_sign = float(prepared_item["label_sign"])
    math_prob = float(torch.exp(log_probs[0, int(prepared_item["math_token_id"])]).item())
    creative_prob = float(torch.exp(log_probs[0, int(prepared_item["creative_token_id"])]).item())
    pairwise_denom = max(math_prob + creative_prob, 1e-12)
    return {
        "signed_label_margin": label_sign * margin,
        "label_target_pairwise_prob": float(math_prob / pairwise_denom) if label_sign > 0 else float(creative_prob / pairwise_denom),
        "label_accuracy": float((margin >= 0.0) if label_sign > 0 else (margin <= 0.0)),
    }


def project_orthogonal_noise(noise, basis):
    if basis is None or basis.numel() == 0:
        return noise
    flat_noise = noise.reshape(-1, noise.shape[-1])
    proj_flat = (flat_noise @ basis) @ basis.T
    return (flat_noise - proj_flat).reshape_as(noise)


def fit_principal_basis(chunks, top_k):
    matrix = np.concatenate([np.asarray(chunk, dtype=np.float64) for chunk in chunks], axis=0)
    if matrix.shape[0] < 2:
        raise ValueError("Need at least two activation rows to fit a principal subspace.")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    max_rank = min(int(top_k), vh.shape[0], matrix.shape[1], centered.shape[0] - 1)
    if max_rank <= 0:
        raise ValueError(f"Unable to fit a non-empty basis: top_k={top_k}, matrix_shape={matrix.shape}")
    variance = singular_values ** 2
    total_variance = float(np.sum(variance))
    explained = variance[:max_rank] / max(total_variance, 1e-12)
    return {
        "basis": torch.tensor(vh[:max_rank].T.copy(), dtype=torch.float32),
        "n_rows": int(matrix.shape[0]),
        "n_features": int(matrix.shape[1]),
        "top_k_used": int(max_rank),
        "pc1_explained_variance_ratio": float(explained[0]) if explained.size else 0.0,
        "topk_cumulative_explained_variance_ratio": float(np.sum(explained)),
        "orthogonal_complement_variance_ratio": float(max(0.0, 1.0 - np.sum(explained))),
    }


class FullTensorSiteCaptureHook:
    def __init__(self):
        self.captured = None
        self.handle = None

    def clear(self):
        self.captured = None

    def _pre_hook(self):
        def hook_fn(module, args):
            self.captured = args[0].detach().float().cpu()
            return None

        return hook_fn

    def _forward_hook(self):
        def hook_fn(module, args, output):
            self.captured = extract_output_tensor(output).detach().float().cpu()
            return output

        return hook_fn

    def attach(self, model, layer, site):
        module, kind = resolve_site_module(model, layer, site)
        if kind == "pre":
            self.handle = module.register_forward_pre_hook(self._pre_hook())
        else:
            self.handle = module.register_forward_hook(self._forward_hook())

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class OrthogonalNoiseSiteHook:
    def __init__(self, basis, noise_scale=0.5, position_scope="full_sequence", seed=0):
        self.basis = basis.detach().cpu()
        self.noise_scale = float(noise_scale)
        self.position_scope = str(position_scope)
        self.rng = np.random.default_rng(int(seed))
        self.handle = None
        self.last_activation_std = 0.0
        self.last_raw_noise_rms = 0.0
        self.last_orthogonal_noise_rms = 0.0
        self.last_projected_energy_fraction = 0.0
        self.last_effective_position_count = 0

    def _apply(self, x):
        basis = self.basis.to(device=x.device, dtype=x.dtype)
        raw_noise = torch.from_numpy(self.rng.standard_normal(size=tuple(x.shape)).astype(np.float32)).to(device=x.device, dtype=x.dtype)
        activation_std = torch.std(x)
        raw_noise = raw_noise * (activation_std * self.noise_scale)
        orthogonal_noise = project_orthogonal_noise(raw_noise, basis)
        if self.position_scope == "last_token":
            x_mod = x.clone()
            x_mod[:, -1:, :] = x_mod[:, -1:, :] + orthogonal_noise[:, -1:, :]
            raw_applied = raw_noise[:, -1:, :]
            orth_applied = orthogonal_noise[:, -1:, :]
            self.last_effective_position_count = 1
        elif self.position_scope == "full_sequence":
            x_mod = x + orthogonal_noise
            raw_applied = raw_noise
            orth_applied = orthogonal_noise
            self.last_effective_position_count = int(x.shape[1])
        else:
            raise ValueError(f"Unsupported position_scope: {self.position_scope}")
        raw_energy = float(torch.sum(raw_applied.float() ** 2).item())
        orth_energy = float(torch.sum(orth_applied.float() ** 2).item())
        self.last_activation_std = float(activation_std.item())
        self.last_raw_noise_rms = float(torch.sqrt(torch.mean(raw_applied.float() ** 2)).item())
        self.last_orthogonal_noise_rms = float(torch.sqrt(torch.mean(orth_applied.float() ** 2)).item())
        self.last_projected_energy_fraction = float(max(0.0, 1.0 - (orth_energy / max(raw_energy, 1e-12))))
        return x_mod

    def _forward_hook(self):
        def hook_fn(module, args, output):
            x_mod = self._apply(extract_output_tensor(output))
            return replace_output_tensor(output, x_mod)

        return hook_fn

    def attach(self, model, layer, site):
        module, kind = resolve_site_module(model, layer, site)
        if kind != "forward":
            raise ValueError(f"Phase 14 orthogonal-noise hook currently expects a forward-hook site, got {site} ({kind})")
        self.handle = module.register_forward_hook(self._forward_hook())

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def run_prompt(model, prompt_ids):
    reset_model_decode_state(model)
    with torch.inference_mode():
        logits = unpack_logits(model(prompt_ids))
    reset_model_decode_state(model)
    return logits.detach().clone()


def build_pair_effect_df(detail_df):
    metric_cols = [
        "delta_from_baseline_signed_label_margin",
        "delta_from_baseline_label_target_pairwise_prob",
        "delta_from_baseline_label_accuracy",
        "activation_std",
        "raw_noise_rms",
        "orthogonal_noise_rms",
        "projected_energy_fraction",
    ]
    grouped = (
        detail_df.groupby(["target_layer", "site", "pair_name"], as_index=False)[metric_cols]
        .mean()
        .sort_values(["target_layer", "site", "pair_name"])
        .reset_index(drop=True)
    )
    return grouped


def build_stats_rows(pair_effect_df, n_perm=20000, seed=0):
    rows = []
    site_metric_specs = [
        "delta_from_baseline_signed_label_margin",
        "delta_from_baseline_label_target_pairwise_prob",
        "delta_from_baseline_label_accuracy",
    ]
    for (target_layer, site), group in pair_effect_df.groupby(["target_layer", "site"]):
        for metric_idx, metric_name in enumerate(site_metric_specs):
            effect = signflip_test(group[metric_name].to_numpy(dtype=np.float64), n_perm=n_perm, seed=seed + 101 * metric_idx + 17 * int(target_layer))
            rows.append(
                {
                    "comparison_type": "site_vs_baseline",
                    "target_layer": int(target_layer),
                    "site": site,
                    "metric_name": metric_name,
                    "mean": effect["mean"],
                    "ci95_low": effect["ci95_low"],
                    "ci95_high": effect["ci95_high"],
                    "pvalue": effect["pvalue"],
                    "n_pairs": effect["n"],
                }
            )
    for target_layer in sorted(pair_effect_df["target_layer"].unique()):
        layer_group = pair_effect_df[pair_effect_df["target_layer"] == target_layer]
        pivot = layer_group.pivot(index="pair_name", columns="site")
        for metric_idx, metric_name in enumerate(site_metric_specs + ["orthogonal_noise_rms", "projected_energy_fraction"]):
            if (metric_name, "attn_output") not in pivot.columns or (metric_name, "ffn_output") not in pivot.columns:
                continue
            contrast = paired_signflip_test(
                pivot[(metric_name, "attn_output")].to_numpy(dtype=np.float64),
                pivot[(metric_name, "ffn_output")].to_numpy(dtype=np.float64),
                n_perm=n_perm,
                seed=seed + 1000 + 101 * metric_idx + 19 * int(target_layer),
            )
            rows.append(
                {
                    "comparison_type": "site_contrast",
                    "target_layer": int(target_layer),
                    "site_a": "attn_output",
                    "site_b": "ffn_output",
                    "metric_name": metric_name,
                    "mean_a": contrast["mean_a"],
                    "mean_b": contrast["mean_b"],
                    "mean_a_minus_b": contrast["mean"],
                    "ci95_low": contrast["ci95_low"],
                    "ci95_high": contrast["ci95_high"],
                    "pvalue": contrast["pvalue"],
                    "n_pairs": int(pivot.shape[0]),
                }
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Phase 14: L15 attn_output vs ffn_output orthogonal-noise localization")
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--output-csv", type=str, default="logs/phase14/phase14_l15_sublayer_localization_summary.csv")
    parser.add_argument("--dataset-name", type=str, default="heldout_shared")
    parser.add_argument("--target-layer", type=int, default=15)
    parser.add_argument("--sites", type=str, default="attn_output,ffn_output")
    parser.add_argument("--top-k", type=int, default=185)
    parser.add_argument("--noise-scale", type=float, default=0.5)
    parser.add_argument("--position-scope", type=str, default="full_sequence", choices=["full_sequence", "last_token"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--weights-path", type=str, default=None)
    parser.add_argument("--max-eval-items", type=int, default=None)
    args = parser.parse_args()

    output_paths = phase14_output_paths(args.output_csv)
    ensure_parent_dir(output_paths["summary"])
    sites = parse_site_list(args.sites)
    if set(sites) != {"attn_output", "ffn_output"}:
        raise ValueError(f"Phase 14 is intentionally bounded to attn_output and ffn_output; got {sites}")

    eval_items = load_eval_items(args.eval_json)
    if args.max_eval_items is not None:
        eval_items = eval_items[: int(args.max_eval_items)]

    model, tokenizer, _ = load_genesis_model(weights_path=args.weights_path, device=args.device)
    device = next(model.parameters()).device
    prepared_items = [prepare_eval_item(tokenizer, item, device) for item in eval_items]

    capture_hooks = {site: FullTensorSiteCaptureHook() for site in sites}
    for site, hook in capture_hooks.items():
        hook.attach(model, args.target_layer, site)

    baseline_rows = []
    site_chunks = {site: [] for site in sites}
    try:
        for prepared_item in tqdm(prepared_items, desc="Phase 14 baseline + site capture"):
            for hook in capture_hooks.values():
                hook.clear()
            logits = run_prompt(model, prepared_item["prompt_ids"])
            metrics = score_choice_logits(logits, prepared_item)
            item = prepared_item["item"]
            baseline_rows.append(
                {
                    "dataset_name": args.dataset_name,
                    "item_name": item["name"],
                    "pair_name": pair_name_from_item_name(item["name"]),
                    "label": item["label"],
                    "prompt_token_count": int(prepared_item["prompt_token_count"]),
                    **{f"baseline_{key}": float(value) for key, value in metrics.items()},
                }
            )
            for site, hook in capture_hooks.items():
                if hook.captured is None:
                    raise RuntimeError(f"Missing baseline capture for site={site} item={item['name']}")
                site_chunks[site].append(hook.captured.squeeze(0).numpy())
    finally:
        for hook in capture_hooks.values():
            hook.remove()
        reset_model_decode_state(model)

    baseline_df = pd.DataFrame(baseline_rows)
    baseline_map = baseline_df.set_index("item_name").to_dict(orient="index")

    basis_fits = {site: fit_principal_basis(site_chunks[site], args.top_k) for site in sites}
    pca_rows = [
        {
            "target_layer": int(args.target_layer),
            "site": site,
            "top_k_requested": int(args.top_k),
            "position_scope": args.position_scope,
            **{key: value for key, value in fit.items() if key != "basis"},
        }
        for site, fit in basis_fits.items()
    ]

    detail_rows = []
    for site_idx, site in enumerate(sites):
        basis = basis_fits[site]["basis"]
        for item_idx, prepared_item in enumerate(tqdm(prepared_items, desc=f"Phase 14 orthogonal-noise @ {site}")):
            item = prepared_item["item"]
            item_name = item["name"]
            hook = OrthogonalNoiseSiteHook(
                basis=basis,
                noise_scale=args.noise_scale,
                position_scope=args.position_scope,
                seed=int(args.seed) + 100000 * site_idx + item_idx,
            )
            hook.attach(model, args.target_layer, site)
            try:
                logits = run_prompt(model, prepared_item["prompt_ids"])
            finally:
                hook.remove()
                reset_model_decode_state(model)
            metrics = score_choice_logits(logits, prepared_item)
            baseline = baseline_map[item_name]
            detail_rows.append(
                {
                    "dataset_name": args.dataset_name,
                    "item_name": item_name,
                    "pair_name": pair_name_from_item_name(item_name),
                    "label": item["label"],
                    "target_layer": int(args.target_layer),
                    "site": site,
                    "intervention_kind": "orthogonal_noise",
                    "position_scope": args.position_scope,
                    "top_k_requested": int(args.top_k),
                    "top_k_used": int(basis_fits[site]["top_k_used"]),
                    "noise_scale": float(args.noise_scale),
                    "prompt_token_count": int(prepared_item["prompt_token_count"]),
                    **{key: float(value) for key, value in metrics.items()},
                    **{key: float(baseline[f"baseline_{key}"]) for key in ("signed_label_margin", "label_target_pairwise_prob", "label_accuracy")},
                    "delta_from_baseline_signed_label_margin": float(metrics["signed_label_margin"] - baseline["baseline_signed_label_margin"]),
                    "delta_from_baseline_label_target_pairwise_prob": float(metrics["label_target_pairwise_prob"] - baseline["baseline_label_target_pairwise_prob"]),
                    "delta_from_baseline_label_accuracy": float(metrics["label_accuracy"] - baseline["baseline_label_accuracy"]),
                    "activation_std": float(hook.last_activation_std),
                    "raw_noise_rms": float(hook.last_raw_noise_rms),
                    "orthogonal_noise_rms": float(hook.last_orthogonal_noise_rms),
                    "projected_energy_fraction": float(hook.last_projected_energy_fraction),
                    "effective_position_count": int(hook.last_effective_position_count),
                }
            )

    detail_df = pd.DataFrame(detail_rows)
    pair_effect_df = build_pair_effect_df(detail_df)
    stats_df = pd.DataFrame(build_stats_rows(pair_effect_df, seed=int(args.seed)))

    baseline_summary = {
        "dataset_name": args.dataset_name,
        "target_layer": int(args.target_layer),
        "site": "baseline",
        "intervention_kind": "baseline",
        "position_scope": args.position_scope,
        "top_k_requested": int(args.top_k),
        "top_k_used": 0,
        "noise_scale": 0.0,
        "n_items": int(baseline_df.shape[0]),
        "n_pairs": int(baseline_df["pair_name"].nunique()),
        "mean_signed_label_margin": float(baseline_df["baseline_signed_label_margin"].mean()),
        "mean_label_target_pairwise_prob": float(baseline_df["baseline_label_target_pairwise_prob"].mean()),
        "mean_label_accuracy": float(baseline_df["baseline_label_accuracy"].mean()),
        "delta_from_baseline_mean_signed_label_margin": 0.0,
        "delta_from_baseline_mean_label_target_pairwise_prob": 0.0,
        "delta_from_baseline_mean_label_accuracy": 0.0,
        "mean_activation_std": np.nan,
        "mean_raw_noise_rms": np.nan,
        "mean_orthogonal_noise_rms": np.nan,
        "mean_projected_energy_fraction": np.nan,
    }
    site_summaries = (
        detail_df.groupby(["dataset_name", "target_layer", "site", "intervention_kind", "position_scope", "top_k_requested", "top_k_used", "noise_scale"], as_index=False)
        .agg(
            n_items=("item_name", "count"),
            n_pairs=("pair_name", "nunique"),
            mean_signed_label_margin=("signed_label_margin", "mean"),
            mean_label_target_pairwise_prob=("label_target_pairwise_prob", "mean"),
            mean_label_accuracy=("label_accuracy", "mean"),
            delta_from_baseline_mean_signed_label_margin=("delta_from_baseline_signed_label_margin", "mean"),
            delta_from_baseline_mean_label_target_pairwise_prob=("delta_from_baseline_label_target_pairwise_prob", "mean"),
            delta_from_baseline_mean_label_accuracy=("delta_from_baseline_label_accuracy", "mean"),
            mean_activation_std=("activation_std", "mean"),
            mean_raw_noise_rms=("raw_noise_rms", "mean"),
            mean_orthogonal_noise_rms=("orthogonal_noise_rms", "mean"),
            mean_projected_energy_fraction=("projected_energy_fraction", "mean"),
        )
        .sort_values(["target_layer", "site"])
        .reset_index(drop=True)
    )
    summary_df = pd.concat([pd.DataFrame([baseline_summary]), site_summaries], ignore_index=True)

    summary_df.to_csv(output_paths["summary"], index=False)
    detail_df.to_csv(output_paths["detail"], index=False)
    pair_effect_df.to_csv(output_paths["pair_effect"], index=False)
    stats_df.to_csv(output_paths["stats"], index=False)
    pd.DataFrame(pca_rows).to_csv(output_paths["pca"], index=False)

    print("Saved Phase 14 outputs:")
    for key, path in output_paths.items():
        print(f"  {key}: {Path(path)}")


if __name__ == "__main__":
    main()