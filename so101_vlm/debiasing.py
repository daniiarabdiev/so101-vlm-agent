"""Calibration primitives for categorical readout scores.

All functions are CPU-only. Permutation aggregation consumes independently
measured pass probabilities; it never derives counterfactual passes from one
recorded score vector.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize


def probabilities(logits: Any) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    if values.ndim not in (1, 2) or values.shape[-1] < 2 or not np.isfinite(values).all():
        raise ValueError("finite logits with at least two labels required")
    values = values - values.max(axis=-1, keepdims=True)
    result = np.exp(values)
    return result / result.sum(axis=-1, keepdims=True)


def apply_contextual_correction(logits: Any, content_free_logits: Any) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    prior = np.asarray(content_free_logits, dtype=float)
    if values.shape != prior.shape or values.ndim != 1 or not np.isfinite(values).all() or not np.isfinite(prior).all():
        raise ValueError("matching finite one-dimensional logits required")
    return values - prior


def apply_permutation_average(
    passes: Iterable[tuple[Mapping[str, str], Mapping[str, float]]],
    semantics: Sequence[str],
) -> dict[str, float]:
    semantics = tuple(semantics)
    collected = []
    for options, label_probabilities in passes:
        if set(options) != set(label_probabilities) or set(options.values()) != set(semantics):
            raise ValueError("each permutation must cover the same labels and semantics")
        values = np.asarray([float(label_probabilities[label]) for label in options], dtype=float)
        if not np.isfinite(values).all() or np.any(values < 0) or not np.isclose(values.sum(), 1.0):
            raise ValueError("permutation pass must contain finite normalized probabilities")
        collected.append({semantic: float(label_probabilities[label]) for label, semantic in options.items()})
    if not collected:
        raise ValueError("at least one independently measured permutation pass required")
    return {semantic: float(np.mean([row[semantic] for row in collected])) for semantic in semantics}


def _validate_sets(logits: Any, acceptable_sets: Sequence[set[int]]):
    values = np.asarray(logits, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("finite two-dimensional logits required")
    if len(acceptable_sets) != len(values):
        raise ValueError("acceptable-set count must match rows")
    for accepted in acceptable_sets:
        if not accepted:
            raise ValueError("empty acceptable label set")
        if min(accepted) < 0 or max(accepted) >= values.shape[1]:
            raise ValueError("acceptable label index outside logits")
    return values


def score_sets(logits: Any, acceptable_sets: Sequence[set[int]]) -> dict[str, float | int]:
    values = _validate_sets(logits, acceptable_sets)
    probs = probabilities(values)
    winners = np.argmax(values, axis=1)
    masses = np.asarray([sum(probs[index, list(accepted)]) for index, accepted in enumerate(acceptable_sets)])
    correct = sum(int(winner in accepted) for winner, accepted in zip(winners, acceptable_sets))
    return {
        "correct": correct,
        "total": len(values),
        "agreement": correct / len(values),
        "set_nll": float(-np.log(np.clip(masses, 1e-300, 1.0)).mean()),
    }


@dataclass(frozen=True)
class DiagonalAffineCalibration:
    scales: tuple[float, ...]
    offsets: tuple[float, ...]
    scale_regularization: float
    bias_regularization: float
    fit_rows: int

    def apply(self, logits: Any) -> np.ndarray:
        values = np.asarray(logits, dtype=float)
        scales = np.asarray(self.scales)
        offsets = np.asarray(self.offsets)
        if values.shape[-1] != len(scales) or not np.isfinite(values).all():
            raise ValueError("finite logits matching calibration width required")
        return values * scales + offsets

    def to_dict(self):
        return asdict(self)


def fit_diagonal_affine(
    logits: Any,
    acceptable_sets: Sequence[set[int]],
    *,
    scale_regularization: float = 1.0,
    bias_regularization: float = 1.0,
) -> DiagonalAffineCalibration:
    values = _validate_sets(logits, acceptable_sets)
    if scale_regularization < 0 or bias_regularization < 0:
        raise ValueError("regularization must be nonnegative")
    width = values.shape[1]

    def objective(parameters):
        log_scales = parameters[:width]
        offsets = parameters[width:] - np.mean(parameters[width:])
        corrected = values * np.exp(log_scales) + offsets
        metrics = score_sets(corrected, acceptable_sets)
        return (
            metrics["set_nll"]
            + scale_regularization * float(np.mean(log_scales ** 2))
            + bias_regularization * float(np.mean(offsets ** 2))
        )

    result = minimize(objective, np.zeros(width * 2), method="L-BFGS-B", bounds=[(-4, 4)] * width + [(None, None)] * width)
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"affine optimization failed: {result.message}")
    log_scales = result.x[:width]
    offsets = result.x[width:] - np.mean(result.x[width:])
    scales = np.exp(log_scales)
    if not np.isfinite(scales).all() or not np.isfinite(offsets).all() or np.any(scales <= 0):
        raise RuntimeError("affine optimization produced invalid parameters")
    return DiagonalAffineCalibration(
        scales=tuple(map(float, scales)),
        offsets=tuple(map(float, offsets)),
        scale_regularization=float(scale_regularization),
        bias_regularization=float(bias_regularization),
        fit_rows=len(values),
    )


def assert_disjoint_fingerprints(*partitions: set[str]) -> None:
    seen: set[str] = set()
    for partition in partitions:
        overlap = seen & set(partition)
        if overlap:
            raise ValueError(f"fingerprint overlap across partitions: {sorted(overlap)[:3]}")
        seen |= set(partition)


def deduplicate_rows(rows: Sequence[Mapping[str, Any]]):
    unique = []
    indices: dict[str, int] = {}
    inverse = []
    for row in rows:
        fingerprint = row.get("fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            raise ValueError("each row requires a nonempty fingerprint")
        if fingerprint not in indices:
            indices[fingerprint] = len(unique)
            unique.append(row)
        inverse.append(indices[fingerprint])
    return unique, inverse


def input_fingerprint(row: Mapping[str, Any]) -> str:
    response = row.get("model_response") or {}
    pixels = row.get("processed_images") or {}
    payload = {
        "model_id": response.get("model_id"),
        "model_revision": response.get("model_revision"),
        "mode": response.get("mode"),
        "compute_dtype": response.get("compute_dtype"),
        "reasoning_effort": response.get("reasoning_effort"),
        "prompt": row.get("prompt"),
        "processed_pixel_hashes": [pixels[key].get("sha256") for key in sorted(pixels)],
        "options": row.get("options"),
        "label_token_ids": response.get("label_token_ids"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def validate_recorded_readout(
    row: Mapping[str, Any],
    labels: Sequence[str],
    *,
    model_id: str,
    model_revision: str,
) -> np.ndarray:
    labels = tuple(labels)
    response = row.get("model_response") or {}
    if response.get("model_id") != model_id:
        raise ValueError("model id mismatch")
    if response.get("model_revision") != model_revision:
        raise ValueError("model revision mismatch")
    if response.get("mode") != "readout":
        raise ValueError("readout mode required")
    logits = response.get("label_logits")
    tokens = response.get("label_token_ids")
    if not isinstance(logits, Mapping) or set(logits) != set(labels):
        raise ValueError("record must contain logits for all labels")
    if not isinstance(tokens, Mapping) or set(tokens) != set(labels) or len(set(tokens.values())) != len(labels):
        raise ValueError("record must contain distinct token ids for all labels")
    values = np.asarray([float(logits[label]) for label in labels])
    if not np.isfinite(values).all():
        raise ValueError("all label logits must be finite")
    options = row.get("options")
    if not isinstance(options, Mapping) or set(options) != set(labels):
        raise ValueError("options must cover all labels")
    return values
