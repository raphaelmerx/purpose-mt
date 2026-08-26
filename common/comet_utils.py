"""Shared COMET utilities: model loading and cached prediction.

Usage:
    from common.comet_utils import load_comet_model, comet_cached_predict

    model = load_comet_model("Unbabel/XCOMET-XL")
    scores = comet_cached_predict(model, data, model_name="Unbabel/XCOMET-XL")
"""
import json
import time
from hashlib import md5
from typing import List, Tuple

import diskcache


cache = diskcache.Cache("cache/comet_scores")


def load_comet_model(model_name: str = "Unbabel/XCOMET-XL"):
    """Download (if needed) and load a COMET model."""
    from comet import download_model, load_from_checkpoint

    start = time.time()
    print(f"Downloading/loading COMET model: {model_name}")
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    print(f"Loaded COMET model in {time.time() - start:.2f}s")
    return model


def _sample_cache_key(model_name: str, sample: dict) -> str:
    """Deterministic cache key for a single (model, src, mt, ref) tuple."""
    payload = model_name + json.dumps(
        {"src": sample["src"], "mt": sample["mt"], "ref": sample["ref"]},
        sort_keys=True,
        ensure_ascii=False,
    )
    return md5(payload.encode("utf-8")).hexdigest()


def comet_cached_predict(
    comet_model,
    data: List[dict],
    batch_size: int = 8,
    gpus: int = 1,
    model_name: str = "unknown",
) -> List[float]:
    """Drop-in replacement for ``comet_model.predict`` with per-sample caching.

    Args:
        comet_model: A loaded COMET model instance.
        data: List of dicts with keys ``src``, ``mt``, ``ref``.
        batch_size: Batch size forwarded to ``comet_model.predict``.
        gpus: Number of GPUs forwarded to ``comet_model.predict``.
        model_name: COMET model identifier used in the cache key.

    Returns:
        List of float scores aligned with *data* (one per sample).
    """
    # --- check cache for each sample ---
    keys = [_sample_cache_key(model_name, s) for s in data]
    cached_scores: List[float | None] = []
    uncached_indices: List[int] = []

    for idx, key in enumerate(keys):
        val = cache.get(key)
        if val is not None:
            cached_scores.append(float(val))
        else:
            cached_scores.append(None)
            uncached_indices.append(idx)

    n_cached = len(data) - len(uncached_indices)
    print(f"COMET cache: {n_cached}/{len(data)} samples cached, {len(uncached_indices)} to compute")

    if not uncached_indices:
        return cached_scores  # type: ignore[return-value]

    # --- run COMET only on uncached samples ---
    uncached_data = [data[i] for i in uncached_indices]
    output = comet_model.predict(uncached_data, batch_size=batch_size, gpus=gpus)
    new_scores = [float(s) for s in output.scores]

    # --- store new scores in cache and merge back ---
    for i, idx in enumerate(uncached_indices):
        score = new_scores[i]
        cache[keys[idx]] = score
        cached_scores[idx] = score

    return cached_scores  # type: ignore[return-value]
