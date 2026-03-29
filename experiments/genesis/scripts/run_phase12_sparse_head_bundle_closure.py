import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.genesis_loader import FOX_LAYER_INDICES, load_genesis_model
from scripts.phase10_experiment_utils import ensure_parent_dir, infer_companion_csv, paired_signflip_test, signflip_test
from scripts.phase10_site_hooks import TensorSiteInterventionHook
from scripts.phase9_semantic_utils import load_semantic_direction
from scripts.run_phase10_tail_conditioned_necessity import add_sign_aware_fields, expand_pair_items, load_pair_items, pair_name_from_item_name
from scripts.run_phase9_semantic_steering import load_anchor_direction, make_random_orthogonal_control, validate_upstream_metadata
from scripts.run_phase9_token_position_steering import evaluate_prepared_item, format_position_label, prepare_eval_item, resolve_position_index


DEFAULT_BUNDLE_SPECS = "l7_top3=7:5|6|0,l11_top3=11:5|1|0,joint_l7_l11_top6=7:5|6|0;11:5|1|0"
CONDITION_BASELINE = "baseline"
CONDITION_BUNDLE_ABLATE_ONLY = "bundle_ablate_only"
CONDITION_SEMANTIC_STEER_ONLY = "semantic_steer_only"
CONDITION_SEMANTIC_STEER_PLUS_BUNDLE_ABLATE = "semantic_steer_plus_bundle_ablate"
CONDITION_RANDOM_STEER_ONLY = "random_steer_only"
CONDITION_RANDOM_STEER_PLUS_BUNDLE_ABLATE = "random_steer_plus_bundle_ablate"
CONDITION_ORDER = [
    CONDITION_BASELINE,
    CONDITION_BUNDLE_ABLATE_ONLY,
    CONDITION_SEMANTIC_STEER_ONLY,
    CONDITION_SEMANTIC_STEER_PLUS_BUNDLE_ABLATE,
    CONDITION_RANDOM_STEER_ONLY,
    CONDITION_RANDOM_STEER_PLUS_BUNDLE_ABLATE,
]
ALLOWED_CONTROLS = ("semantic", "random")
METRIC_SPECS = [
    ("signed_label_margin", "signed_label_margin"),
    ("label_target_pairwise_prob", "label_target_pairwise_prob"),
    ("label_accuracy", "label_accuracy"),
    ("anchor_cosine", "anchor_cosine"),
    ("next_token_entropy", "next_token_entropy"),
]
REQUIRED_PAIR_INTERACTION_COLUMNS = [
    "dataset_name", "pair_name", "source_layer", "target_layer", "target_layer_type", "site",
    "bundle_name", "bundle_spec", "bundle_size", "vector_key", "mode", "alpha", "position_label",
    "position_fraction", "n_items_per_pair", "baseline_signed_label_margin",
    "bundle_ablate_only_signed_label_margin", "semantic_steer_only_signed_label_margin",
    "semantic_steer_plus_bundle_ablate_signed_label_margin", "random_steer_only_signed_label_margin",
    "random_steer_plus_bundle_ablate_signed_label_margin", "semantic_interaction_signed_label_margin",
    "random_interaction_signed_label_margin",
]
REQUIRED_STATS_COLUMNS = [
    "dataset_name", "source_layer", "target_layer", "target_layer_type", "site", "bundle_name",
    "bundle_spec", "bundle_size", "vector_key", "mode", "alpha", "position_label", "position_fraction",
    "metric_name", "comparison_type", "n_pairs", "mean_a", "mean_b", "mean_difference",
    "ci95_low", "ci95_high", "pvalue", "n_perm", "seed",
]


def phase12_output_paths(output_csv):
    return {
        "summary": output_csv,
        "detail": infer_companion_csv(output_csv, "detail"),
        "pair_interaction": infer_companion_csv(output_csv, "pair_interaction"),
        "stats": infer_companion_csv(output_csv, "stats"),
    }


def layer_type_name(layer):
    return "fox" if int(layer) in FOX_LAYER_INDICES else "gla"


def parse_controls(raw_controls):
    ordered = []
    for token in raw_controls.split(","):
        control = token.strip().lower()
        if not control:
            continue
        if control not in ALLOWED_CONTROLS:
            raise ValueError(f"Unsupported control: {control}. Expected only {list(ALLOWED_CONTROLS)}")
        if control not in ordered:
            ordered.append(control)
    if set(ordered) != set(ALLOWED_CONTROLS):
        raise ValueError("Phase 12F requires exactly the semantic and random matched controls.")
    return ["semantic", "random"]


def _normalize_bundle_spec(layer_heads):
    return ";".join(f"{layer}:{'|'.join(str(head) for head in heads)}" for layer, heads in layer_heads)


def parse_bundle_specs(raw_bundle_specs):
    bundles = []
    seen_names = set()
    for token in raw_bundle_specs.split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"Invalid bundle spec '{token}'. Expected name=layer:head|head[;layer:head|head].")
        name, raw_spec = token.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid bundle spec '{token}': missing bundle name.")
        if name in seen_names:
            raise ValueError(f"Duplicate bundle name: {name}")
        layer_heads = []
        seen_pairs = set()
        for segment in raw_spec.split(";"):
            segment = segment.strip()
            if not segment:
                continue
            if ":" not in segment:
                raise ValueError(f"Invalid bundle segment '{segment}' in '{token}'.")
            raw_layer, raw_heads = segment.split(":", 1)
            layer = int(raw_layer.strip())
            heads = []
            for raw_head in raw_heads.split("|"):
                raw_head = raw_head.strip()
                if not raw_head:
                    continue
                head = int(raw_head)
                pair = (layer, head)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                heads.append(head)
            if not heads:
                raise ValueError(f"Bundle '{name}' has an empty head list for layer {layer}.")
            layer_heads.append((layer, tuple(heads)))
        if not layer_heads:
            raise ValueError(f"Bundle '{name}' is empty.")
        seen_names.add(name)
        bundles.append(
            {
                "bundle_name": name,
                "bundle_spec": _normalize_bundle_spec(layer_heads),
                "layer_heads": tuple(layer_heads),
                "bundle_size": int(sum(len(heads) for _, heads in layer_heads)),
            }
        )
    if not bundles:
        raise ValueError("No bundle specs parsed.")
    return bundles


def validate_bundle_specs(bundle_specs, config):
    n_layers = int(config.n_layer)
    n_heads = int(config.n_head)
    for bundle in bundle_specs:
        for layer, heads in bundle["layer_heads"]:
            if layer < 0 or layer >= n_layers:
                raise ValueError(f"Bundle {bundle['bundle_name']} uses invalid layer {layer}; model has {n_layers} layers.")
            for head in heads:
                if head < 0 or head >= n_heads:
                    raise ValueError(
                        f"Bundle {bundle['bundle_name']} uses invalid head {head} at layer {layer}; model has {n_heads} heads."
                    )


class OProjHeadBundleAblation:
    def __init__(self, model, layer_heads, head_dim):
        self.model = model
        self.layer_heads = tuple(layer_heads)
        self.head_dim = int(head_dim)
        self.saved_slices = []

    def __enter__(self):
        for layer, heads in self.layer_heads:
            attn = getattr(self.model.blocks[int(layer)], "attn", None)
            if attn is None or not hasattr(attn, "o_proj") or not hasattr(attn.o_proj, "weight"):
                raise ValueError(f"Layer {layer} has no attn.o_proj weight to ablate")
            weight = attn.o_proj.weight
            for head in heads:
                start = int(head) * self.head_dim
                end = start + self.head_dim
                if end > weight.shape[1]:
                    raise ValueError(
                        f"Head slice [{start}:{end}] exceeds o_proj input width {weight.shape[1]} at layer {layer}."
                    )
                with torch.no_grad():
                    saved = weight[:, start:end].detach().clone()
                    weight[:, start:end].zero_()
                self.saved_slices.append((int(layer), int(head), saved))
        return self

    def __exit__(self, exc_type, exc, tb):
        for layer, head, saved in reversed(self.saved_slices):
            attn = self.model.blocks[layer].attn
            start = head * self.head_dim
            end = start + self.head_dim
            with torch.no_grad():
                attn.o_proj.weight[:, start:end].copy_(saved)
        self.saved_slices = []


def add_condition_row(detail_rows, *, baseline, row, prepared_item, args, dataset_name, position_label, condition, control_name, bundle):
    pos_idx = resolve_position_index(prepared_item["prompt_token_count"], args.position_fraction)
    label_target_prob = row["pairwise_math_prob"] if prepared_item["label_sign"] > 0 else row["pairwise_creative_prob"]
    baseline_target_prob = baseline["pairwise_math_prob"] if prepared_item["label_sign"] > 0 else baseline["pairwise_creative_prob"]
    enriched = dict(row)
    enriched.update(
        {
            "dataset_name": dataset_name,
            "pair_name": pair_name_from_item_name(prepared_item["item"]["name"]),
            "source_layer": int(args.source_layer),
            "target_layer": int(args.target_layer),
            "target_layer_type": layer_type_name(args.target_layer),
            "site": args.site,
            "bundle_name": bundle["bundle_name"],
            "bundle_spec": bundle["bundle_spec"],
            "bundle_size": int(bundle["bundle_size"]),
            "vector_key": args.vector_key,
            "condition": condition,
            "control": control_name,
            "steering_enabled": int("steer" in condition),
            "bundle_ablation_enabled": int("ablate" in condition),
            "alpha": float(args.alpha) if "steer" in condition else 0.0,
            "mode": args.mode,
            "position_fraction": float(args.position_fraction),
            "position_label": position_label,
            "selected_position_index": pos_idx,
            "normalized_position": float(pos_idx / max(prepared_item["prompt_token_count"] - 1, 1)),
            "distance_to_answer": int(prepared_item["prompt_token_count"] - 1 - pos_idx),
            "label_target_pairwise_prob": float(label_target_prob),
            "baseline_math_minus_creative_logprob": float(baseline["math_minus_creative_logprob"]),
            "delta_from_baseline_math_minus_creative_logprob": float(
                row["math_minus_creative_logprob"] - baseline["math_minus_creative_logprob"]
            ),
            "baseline_signed_label_margin": float(baseline["signed_label_margin"]),
            "delta_from_baseline_signed_label_margin": float(row["signed_label_margin"] - baseline["signed_label_margin"]),
            "baseline_label_accuracy": float(baseline["label_accuracy"]),
            "delta_from_baseline_label_accuracy": float(row["label_accuracy"] - baseline["label_accuracy"]),
            "baseline_label_target_pairwise_prob": float(baseline_target_prob),
            "delta_from_baseline_label_target_pairwise_prob": float(label_target_prob - baseline_target_prob),
            "baseline_anchor_cosine": float(baseline["anchor_cosine"]),
            "delta_from_baseline_anchor_cosine": float(row["anchor_cosine"] - baseline["anchor_cosine"]),
            "baseline_next_token_entropy": float(baseline["next_token_entropy"]),
            "delta_from_baseline_next_token_entropy": float(row["next_token_entropy"] - baseline["next_token_entropy"]),
        }
    )
    detail_rows.append(enriched)


def build_summary_df(detail_df):
    summary_df = (
        detail_df.groupby(
            [
                "dataset_name", "source_layer", "target_layer", "target_layer_type", "site", "bundle_name",
                "bundle_spec", "bundle_size", "vector_key", "condition", "control", "steering_enabled",
                "bundle_ablation_enabled", "alpha", "mode", "position_label", "position_fraction",
            ],
            as_index=False,
        )
        .agg(
            n_items=("item_name", "count"),
            n_pairs=("pair_name", "nunique"),
            mean_math_bias_logprob=("math_minus_creative_logprob", "mean"),
            mean_label_target_pairwise_prob=("label_target_pairwise_prob", "mean"),
            mean_signed_label_margin=("signed_label_margin", "mean"),
            mean_label_accuracy=("label_accuracy", "mean"),
            mean_anchor_cosine=("anchor_cosine", "mean"),
            mean_next_token_entropy=("next_token_entropy", "mean"),
            baseline_mean_math_bias_logprob=("baseline_math_minus_creative_logprob", "mean"),
            delta_from_baseline_mean_math_bias_logprob=("delta_from_baseline_math_minus_creative_logprob", "mean"),
            baseline_mean_label_target_pairwise_prob=("baseline_label_target_pairwise_prob", "mean"),
            delta_from_baseline_mean_label_target_pairwise_prob=("delta_from_baseline_label_target_pairwise_prob", "mean"),
            baseline_mean_signed_label_margin=("baseline_signed_label_margin", "mean"),
            delta_from_baseline_mean_signed_label_margin=("delta_from_baseline_signed_label_margin", "mean"),
            baseline_mean_label_accuracy=("baseline_label_accuracy", "mean"),
            delta_from_baseline_mean_label_accuracy=("delta_from_baseline_label_accuracy", "mean"),
            baseline_mean_anchor_cosine=("baseline_anchor_cosine", "mean"),
            delta_from_baseline_mean_anchor_cosine=("delta_from_baseline_anchor_cosine", "mean"),
            baseline_mean_next_token_entropy=("baseline_next_token_entropy", "mean"),
            delta_from_baseline_mean_next_token_entropy=("delta_from_baseline_next_token_entropy", "mean"),
        )
    )
    order_map = {name: idx for idx, name in enumerate(CONDITION_ORDER)}
    summary_df["_condition_order"] = summary_df["condition"].map(order_map).fillna(len(order_map)).astype(int)
    summary_df = summary_df.sort_values(["dataset_name", "bundle_name", "target_layer", "site", "_condition_order"])
    return summary_df.drop(columns=["_condition_order"])


def build_pair_interaction_df(detail_df):
    pair_condition_df = (
        detail_df.groupby(
            [
                "dataset_name", "pair_name", "source_layer", "target_layer", "target_layer_type", "site",
                "bundle_name", "bundle_spec", "bundle_size", "vector_key", "mode", "alpha", "position_label",
                "position_fraction", "condition",
            ],
            as_index=False,
        )
        .agg(
            n_items_per_pair=("item_name", "count"),
            signed_label_margin=("signed_label_margin", "mean"),
            label_target_pairwise_prob=("label_target_pairwise_prob", "mean"),
            label_accuracy=("label_accuracy", "mean"),
            anchor_cosine=("anchor_cosine", "mean"),
            next_token_entropy=("next_token_entropy", "mean"),
        )
    )
    required_conditions = {
        CONDITION_BASELINE,
        CONDITION_BUNDLE_ABLATE_ONLY,
        CONDITION_SEMANTIC_STEER_ONLY,
        CONDITION_SEMANTIC_STEER_PLUS_BUNDLE_ABLATE,
        CONDITION_RANDOM_STEER_ONLY,
        CONDITION_RANDOM_STEER_PLUS_BUNDLE_ABLATE,
    }
    group_cols = [
        "dataset_name", "pair_name", "source_layer", "target_layer", "target_layer_type", "site", "bundle_name",
        "bundle_spec", "bundle_size", "vector_key", "mode", "position_label", "position_fraction",
    ]
    rows = []
    for group_key, group in pair_condition_df.groupby(group_cols, dropna=False):
        condition_table = group.set_index("condition")
        if not required_conditions.issubset(set(condition_table.index)):
            continue
        row = dict(zip(group_cols, group_key))
        row["alpha"] = float(group["alpha"].max())
        row["n_items_per_pair"] = int(condition_table.loc[CONDITION_BASELINE, "n_items_per_pair"])
        for metric_col, metric_name in METRIC_SPECS:
            baseline = float(condition_table.loc[CONDITION_BASELINE, metric_col])
            ablate = float(condition_table.loc[CONDITION_BUNDLE_ABLATE_ONLY, metric_col])
            semantic_steer = float(condition_table.loc[CONDITION_SEMANTIC_STEER_ONLY, metric_col])
            semantic_both = float(condition_table.loc[CONDITION_SEMANTIC_STEER_PLUS_BUNDLE_ABLATE, metric_col])
            random_steer = float(condition_table.loc[CONDITION_RANDOM_STEER_ONLY, metric_col])
            random_both = float(condition_table.loc[CONDITION_RANDOM_STEER_PLUS_BUNDLE_ABLATE, metric_col])
            row[f"baseline_{metric_name}"] = baseline
            row[f"bundle_ablate_only_{metric_name}"] = ablate
            row[f"semantic_steer_only_{metric_name}"] = semantic_steer
            row[f"semantic_steer_plus_bundle_ablate_{metric_name}"] = semantic_both
            row[f"random_steer_only_{metric_name}"] = random_steer
            row[f"random_steer_plus_bundle_ablate_{metric_name}"] = random_both
            row[f"semantic_steer_effect_{metric_name}"] = semantic_steer - baseline
            row[f"semantic_ablated_steer_effect_{metric_name}"] = semantic_both - ablate
            row[f"semantic_interaction_{metric_name}"] = semantic_both - semantic_steer - ablate + baseline
            row[f"random_steer_effect_{metric_name}"] = random_steer - baseline
            row[f"random_ablated_steer_effect_{metric_name}"] = random_both - ablate
            row[f"random_interaction_{metric_name}"] = random_both - random_steer - ablate + baseline
        rows.append(row)
    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        return pair_df
    return pair_df.sort_values(["dataset_name", "bundle_name", "target_layer", "site", "pair_name"]).reset_index(drop=True)


def _stats_row(common, metric_name, comparison_type, result, *, n_perm, seed):
    return {
        **common,
        "metric_name": metric_name,
        "comparison_type": comparison_type,
        "n_pairs": int(result["n"]),
        "mean_a": float(result.get("mean_a", result["mean"])),
        "mean_b": float(result.get("mean_b", 0.0)),
        "mean_difference": float(result["mean"]),
        "ci95_low": float(result["ci95_low"]),
        "ci95_high": float(result["ci95_high"]),
        "pvalue": float(result["pvalue"]),
        "n_perm": int(n_perm),
        "seed": int(seed),
    }


def build_stats_rows(pair_interaction_df, *, n_perm=100000, seed=1234):
    rows = []
    if pair_interaction_df.empty:
        return rows
    group_cols = [
        "dataset_name", "source_layer", "target_layer", "target_layer_type", "site", "bundle_name",
        "bundle_spec", "bundle_size", "vector_key", "mode", "alpha", "position_label", "position_fraction",
    ]
    for group_key, group in pair_interaction_df.groupby(group_cols, dropna=False):
        common = dict(zip(group_cols, group_key))
        bundle_seed = sum(ord(ch) for ch in str(common["bundle_name"]))
        for metric_idx, (_, metric_name) in enumerate(METRIC_SPECS):
            semantic = group[f"semantic_interaction_{metric_name}"].to_numpy(dtype=np.float64)
            random = group[f"random_interaction_{metric_name}"].to_numpy(dtype=np.float64)
            seed_base = int(seed) + (metric_idx + 1) * 1000 + 31 * int(common["target_layer"]) + 17 * bundle_seed
            semantic_zero = signflip_test(semantic, n_perm=n_perm, seed=seed_base + 1)
            random_zero = signflip_test(random, n_perm=n_perm, seed=seed_base + 2)
            semantic_vs_random = paired_signflip_test(semantic, random, n_perm=n_perm, seed=seed_base + 3)
            rows.append(_stats_row(common, metric_name, "semantic_interaction_vs_zero", semantic_zero, n_perm=n_perm, seed=seed_base + 1))
            rows.append(_stats_row(common, metric_name, "random_interaction_vs_zero", random_zero, n_perm=n_perm, seed=seed_base + 2))
            rows.append(
                _stats_row(
                    common,
                    metric_name,
                    "semantic_interaction_vs_random_interaction",
                    semantic_vs_random,
                    n_perm=n_perm,
                    seed=seed_base + 3,
                )
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description="Phase 12F: sparse head-bundle closure interaction at L11 attn_output")
    parser.add_argument("--semantic-directions", type=str, default="logs/phase9/vectors")
    parser.add_argument("--data-dir", type=str, default="logs/phase9/data")
    parser.add_argument("--source-layer", type=int, default=15)
    parser.add_argument("--target-layer", type=int, default=11)
    parser.add_argument("--site", type=str, default="attn_output")
    parser.add_argument("--bundle-specs", type=str, default=DEFAULT_BUNDLE_SPECS)
    parser.add_argument("--alpha", type=float, default=12.5)
    parser.add_argument("--mode", type=str, default="add", help="add or ablate")
    parser.add_argument("--controls", type=str, default="semantic,random")
    parser.add_argument("--vector-key", type=str, default="delta_perp")
    parser.add_argument("--anchor-layer", type=int, default=29)
    parser.add_argument("--position-fraction", type=float, default=1.0)
    parser.add_argument("--eval-json", type=str, default="prompts/phase9_shared_eval_heldout.json")
    parser.add_argument("--dataset-name", type=str, default="heldout_shared")
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--output-csv", type=str, default="logs/phase12/phase12f_sparse_head_bundle_closure.csv")
    parser.add_argument("--detail-csv", type=str, default=None)
    parser.add_argument("--pair-interaction-csv", type=str, default=None)
    parser.add_argument("--stats-csv", type=str, default=None)
    parser.add_argument("--n-perm", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--allow-invalid-metadata", action="store_true")
    args = parser.parse_args()

    if args.mode not in {"add", "ablate"}:
        raise ValueError(f"Unsupported mode: {args.mode}")
    controls = parse_controls(args.controls)
    bundle_specs = parse_bundle_specs(args.bundle_specs)
    paths = phase12_output_paths(args.output_csv)
    if args.detail_csv is not None:
        paths["detail"] = args.detail_csv
    if args.pair_interaction_csv is not None:
        paths["pair_interaction"] = args.pair_interaction_csv
    if args.stats_csv is not None:
        paths["stats"] = args.stats_csv
    for path in paths.values():
        ensure_parent_dir(path)

    validate_upstream_metadata(args.data_dir, allow_invalid_metadata=args.allow_invalid_metadata)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer, config = load_genesis_model(device=device)
    validate_bundle_specs(bundle_specs, config)
    anchor_direction = load_anchor_direction(
        args.data_dir,
        args.anchor_layer,
        allow_invalid_metadata=args.allow_invalid_metadata,
    )
    pair_items = load_pair_items(args.eval_json, max_eval_items=args.max_eval_items)
    prepared_items = [prepare_eval_item(tokenizer, item, device) for item in expand_pair_items(pair_items)]
    semantic_vec = torch.tensor(
        load_semantic_direction(args.semantic_directions, args.source_layer, vector_key=args.vector_key),
        device=device,
        dtype=torch.float32,
    )
    direction_bank = {
        "semantic": semantic_vec,
        "random": make_random_orthogonal_control(semantic_vec, args.seed + int(args.source_layer)),
    }

    baseline_by_item = {}
    for prepared_item in tqdm(prepared_items, desc="Baseline", leave=False):
        baseline_by_item[prepared_item["item"]["name"]] = add_sign_aware_fields(
            prepared_item,
            evaluate_prepared_item(model, prepared_item, args.anchor_layer, anchor_direction),
        )

    steered_by_control = {control_name: {} for control_name in controls}
    condition_names = {
        "semantic": (CONDITION_SEMANTIC_STEER_ONLY, CONDITION_SEMANTIC_STEER_PLUS_BUNDLE_ABLATE),
        "random": (CONDITION_RANDOM_STEER_ONLY, CONDITION_RANDOM_STEER_PLUS_BUNDLE_ABLATE),
    }
    for control_name in controls:
        hook = TensorSiteInterventionHook(
            direction_bank[control_name],
            alpha=args.alpha,
            mode=args.mode,
            position_fraction=args.position_fraction,
        )
        hook.attach(model, args.target_layer, args.site)
        try:
            for prepared_item in tqdm(prepared_items, desc=f"{control_name} steer", leave=False):
                steered_by_control[control_name][prepared_item["item"]["name"]] = add_sign_aware_fields(
                    prepared_item,
                    evaluate_prepared_item(model, prepared_item, args.anchor_layer, anchor_direction),
                )
        finally:
            hook.remove()

    position_label = format_position_label(args.position_fraction)
    print("\n=== PHASE 12F: SPARSE HEAD-BUNDLE CLOSURE INTERACTION ===")
    print(f"Dataset: {args.dataset_name}")
    print(f"Pairs: {len(pair_items)} | Eval items: {len(prepared_items)}")
    print(f"Source layer: {args.source_layer} | Target: L{args.target_layer} {args.site}")
    print(f"Bundles: {[bundle['bundle_spec'] for bundle in bundle_specs]}")
    print(f"Controls: {controls} | Alpha: {args.alpha} | Mode: {args.mode}")
    print(f"Position: {position_label} ({args.position_fraction:.2f})")

    detail_rows = []
    for bundle in bundle_specs:
        ablated_by_item = {}
        with OProjHeadBundleAblation(model, bundle["layer_heads"], config.head_dim):
            for prepared_item in tqdm(prepared_items, desc=f"{bundle['bundle_name']} ablate", leave=False):
                ablated_by_item[prepared_item["item"]["name"]] = add_sign_aware_fields(
                    prepared_item,
                    evaluate_prepared_item(model, prepared_item, args.anchor_layer, anchor_direction),
                )
        for prepared_item in prepared_items:
            item_name = prepared_item["item"]["name"]
            add_condition_row(
                detail_rows,
                baseline=baseline_by_item[item_name],
                row=baseline_by_item[item_name],
                prepared_item=prepared_item,
                args=args,
                dataset_name=args.dataset_name,
                position_label=position_label,
                condition=CONDITION_BASELINE,
                control_name="none",
                bundle=bundle,
            )
            add_condition_row(
                detail_rows,
                baseline=baseline_by_item[item_name],
                row=ablated_by_item[item_name],
                prepared_item=prepared_item,
                args=args,
                dataset_name=args.dataset_name,
                position_label=position_label,
                condition=CONDITION_BUNDLE_ABLATE_ONLY,
                control_name="none",
                bundle=bundle,
            )
            for control_name in controls:
                steer_condition, _ = condition_names[control_name]
                add_condition_row(
                    detail_rows,
                    baseline=baseline_by_item[item_name],
                    row=steered_by_control[control_name][item_name],
                    prepared_item=prepared_item,
                    args=args,
                    dataset_name=args.dataset_name,
                    position_label=position_label,
                    condition=steer_condition,
                    control_name=control_name,
                    bundle=bundle,
                )
        for control_name in controls:
            _, both_condition = condition_names[control_name]
            hook = TensorSiteInterventionHook(
                direction_bank[control_name],
                alpha=args.alpha,
                mode=args.mode,
                position_fraction=args.position_fraction,
            )
            hook.attach(model, args.target_layer, args.site)
            try:
                with OProjHeadBundleAblation(model, bundle["layer_heads"], config.head_dim):
                    for prepared_item in tqdm(prepared_items, desc=f"{bundle['bundle_name']} {control_name}+ablate", leave=False):
                        both = add_sign_aware_fields(
                            prepared_item,
                            evaluate_prepared_item(model, prepared_item, args.anchor_layer, anchor_direction),
                        )
                        add_condition_row(
                            detail_rows,
                            baseline=baseline_by_item[prepared_item["item"]["name"]],
                            row=both,
                            prepared_item=prepared_item,
                            args=args,
                            dataset_name=args.dataset_name,
                            position_label=position_label,
                            condition=both_condition,
                            control_name=control_name,
                            bundle=bundle,
                        )
            finally:
                hook.remove()

    detail_df = pd.DataFrame(detail_rows)
    summary_df = build_summary_df(detail_df)
    pair_interaction_df = build_pair_interaction_df(detail_df)
    stats_df = pd.DataFrame(build_stats_rows(pair_interaction_df, n_perm=args.n_perm, seed=args.seed))

    summary_df.to_csv(paths["summary"], index=False)
    detail_df.to_csv(paths["detail"], index=False)
    pair_interaction_df.to_csv(paths["pair_interaction"], index=False)
    stats_df.to_csv(paths["stats"], index=False)

    primary_stats = stats_df[stats_df["metric_name"] == "signed_label_margin"].copy()
    print("\n--- PHASE 12F SUMMARY ---")
    print(summary_df.to_string(index=False))
    print("\n--- PHASE 12F PRIMARY INTERACTION STATS ---")
    print(primary_stats.to_string(index=False))
    print(f"\nSummary saved to {paths['summary']}")
    print(f"Detail saved to {paths['detail']}")
    print(f"Pair interactions saved to {paths['pair_interaction']}")
    print(f"Stats saved to {paths['stats']}")


if __name__ == "__main__":
    main()