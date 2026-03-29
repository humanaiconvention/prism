import json
from pathlib import Path

import numpy as np


def parse_int_list(raw):
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def parse_float_list(raw):
    return [float(x.strip()) for x in str(raw).split(",") if x.strip()]


def infer_detail_csv(output_csv):
    path = Path(output_csv)
    return str(path.with_name(f"{path.stem}_detail{path.suffix or '.csv'}"))


def infer_summary_csv(output_csv):
    path = Path(output_csv)
    stem = path.stem
    if stem.endswith("_results"):
        stem = f"{stem[:-8]}_summary"
    else:
        stem = f"{stem}_summary"
    return str(path.with_name(f"{stem}{path.suffix or '.csv'}"))


def _load_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Semantic-directions manifest must be a JSON object: {path}")
    return payload


def resolve_semantic_spec(semantic_directions):
    path = Path(semantic_directions)
    if path.is_dir():
        return {"mode": "vector_dir", "vector_dir": path}

    if path.exists() and path.suffix.lower() == ".json":
        payload = _load_manifest(path)
        vector_dir = payload.get("vector_dir")
        vector_dir_path = None
        if vector_dir is not None:
            vector_dir_path = Path(vector_dir)
            if not vector_dir_path.is_absolute():
                vector_dir_path = path.parent / vector_dir_path
        return {"mode": "manifest", "path": path, "payload": payload, "vector_dir": vector_dir_path}

    if path.exists():
        raise ValueError(
            f"Unsupported semantic-directions path: {path}. Expected a directory or JSON manifest."
        )

    fallback_dir = path.parent / "vectors"
    if path.suffix.lower() == ".json" and fallback_dir.exists():
        return {"mode": "vector_dir", "vector_dir": fallback_dir}

    raise FileNotFoundError(f"Semantic-directions path not found: {path}")


def load_semantic_direction(semantic_directions, layer, vector_key="delta_perp"):
    spec = resolve_semantic_spec(semantic_directions)
    vector_path = None

    if spec["mode"] == "manifest":
        payload = spec["payload"]
        layer_entry = payload.get("layers", {}).get(str(layer))
        if isinstance(layer_entry, dict) and layer_entry.get("path"):
            vector_path = Path(layer_entry["path"])
            if not vector_path.is_absolute():
                vector_path = spec["path"].parent / vector_path
        elif spec.get("vector_dir") is not None:
            vector_path = spec["vector_dir"] / f"layer_{layer}_vector.npz"
    else:
        vector_path = spec["vector_dir"] / f"layer_{layer}_vector.npz"

    if vector_path is None or not vector_path.exists():
        raise FileNotFoundError(f"Semantic vector for layer {layer} not found under {semantic_directions}")

    payload = np.load(vector_path)
    if vector_key not in payload:
        raise KeyError(f"Vector key '{vector_key}' not found in {vector_path}")
    vec = payload[vector_key].astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        raise ValueError(f"Vector '{vector_key}' in {vector_path} has near-zero norm.")
    return vec / norm