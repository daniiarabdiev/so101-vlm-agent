"""OpenRouter image-plus-logprob readout client for bounded Run 2 ranking."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import httpx

from .budget import ProviderBudgetLedger


API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_KEY_PATH = Path.home() / ".config/openrouter/so101-vlm-api-key"


@dataclass(frozen=True)
class OpenRouterPriceBound:
    """Official endpoint price snapshot used only to reserve an upper bound."""

    prompt_usd_per_token: float
    completion_usd_per_token: float
    context_tokens: int
    maximum_visual_tokens_per_image: int
    source_checked_at: str

    def request_bound_usd(
        self,
        *,
        prompt: str,
        image_count: int,
        max_completion_tokens: int,
    ) -> float:
        if max_completion_tokens < 1 or max_completion_tokens > self.context_tokens:
            raise ValueError("invalid completion token bound")
        if image_count < 1:
            raise ValueError("at least one image is required")
        # UTF-8 bytes upper-bound text BPE tokens. The per-image cap is a
        # reviewed processor limit, with 1,024 tokens retained for templates
        # and multimodal sentinels. Cap at the advertised context because a
        # larger request is rejected before inference.
        input_tokens = min(
            self.context_tokens,
            len(prompt.encode("utf-8"))
            + image_count * self.maximum_visual_tokens_per_image
            + 1_024,
        )
        return (
            input_tokens * self.prompt_usd_per_token
            + max_completion_tokens * self.completion_usd_per_token
        )


# Provider-specific prices from OpenRouter's official endpoints API, checked
# 2026-09-20. The full advertised context window is reserved, which is much
# more conservative than the observed short grid prompts.
PRICE_BOUNDS: dict[tuple[str, str], OpenRouterPriceBound] = {
    ("qwen/qwen3-vl-8b-instruct", "Parasail"): OpenRouterPriceBound(
        0.25 / 1_000_000, 0.75 / 1_000_000, 262_144, 16_384, "openrouter_endpoints_api_2026-09-20"
    ),
    ("qwen/qwen3-vl-8b-instruct", "Alibaba"): OpenRouterPriceBound(
        0.117 / 1_000_000, 0.455 / 1_000_000, 262_144, 16_384, "openrouter_endpoints_api_2026-09-20"
    ),
    ("qwen/qwen3-vl-30b-a3b-instruct", "Novita"): OpenRouterPriceBound(
        0.20 / 1_000_000, 0.70 / 1_000_000, 262_144, 16_384, "openrouter_endpoints_api_2026-09-20"
    ),
    ("qwen/qwen3-vl-32b-instruct", "Alibaba"): OpenRouterPriceBound(
        0.104 / 1_000_000, 0.416 / 1_000_000, 262_144, 16_384, "openrouter_endpoints_api_2026-09-20"
    ),
    ("qwen/qwen3-vl-235b-a22b-instruct", "Parasail"): OpenRouterPriceBound(
        0.21 / 1_000_000, 1.90 / 1_000_000, 262_144, 16_384, "openrouter_endpoints_api_2026-09-20"
    ),
    ("google/gemma-4-26b-a4b-it", "Darkbloom"): OpenRouterPriceBound(
        0.042 / 1_000_000, 0.22 / 1_000_000, 131_072, 16_384, "openrouter_endpoints_api_2026-09-20"
    ),
    ("google/gemma-4-26b-a4b-it", "DekaLLM"): OpenRouterPriceBound(
        0.06 / 1_000_000, 0.33 / 1_000_000, 262_144, 16_384, "openrouter_endpoints_api_2026-09-20"
    ),
    ("qwen/qwen3.5-9b", "Parasail"): OpenRouterPriceBound(
        0.10 / 1_000_000, 0.25 / 1_000_000, 262_144, 16_384, "openrouter_endpoints_api_2026-09-20"
    ),
    ("qwen/qwen3.5-9b", "Venice"): OpenRouterPriceBound(
        0.10 / 1_000_000, 0.15 / 1_000_000, 256_000, 16_384, "openrouter_endpoints_api_2026-09-20"
    ),
    ("qwen/qwen3.8-27b", "Parasail"): OpenRouterPriceBound(
        0.24 / 1_000_000, 2.20 / 1_000_000, 1_000_000, 16_384, "openrouter_endpoints_api_2026-09-20"
    ),
    ("qwen/qwen3.8-27b", "Darkbloom"): OpenRouterPriceBound(
        0.10 / 1_000_000, 1.80 / 1_000_000, 262_144, 16_384, "openrouter_endpoints_api_2026-09-20"
    ),
}


class ProviderCapabilityError(RuntimeError):
    pass


class ProviderHTTPError(RuntimeError):
    """Sanitized provider error with bounded-retry metadata."""

    def __init__(self, status_code: int, details: Mapping[str, object], retry_after_s: float | None):
        self.status_code = status_code
        self.details = dict(details)
        self.retry_after_s = retry_after_s
        super().__init__(
            f"OpenRouter HTTP {status_code}: {json.dumps(self.details, sort_keys=True)}"
        )


def load_api_key() -> str:
    value = os.environ.get("OPENROUTER_API_KEY")
    if value:
        return value.strip()
    if DEFAULT_KEY_PATH.exists():
        return DEFAULT_KEY_PATH.read_text(encoding="utf-8").strip()
    raise RuntimeError("OpenRouter credential is unavailable")


def image_data_url(image: object) -> str:
    if isinstance(image, (str, Path)):
        source = Path(image)
        if not source.is_file():
            raise ValueError(f"image does not exist: {source}")
        mime = mimetypes.guess_type(source.name)[0] or "image/png"
        raw = source.read_bytes()
    elif hasattr(image, "save"):
        stream = io.BytesIO()
        image.save(stream, format="PNG")  # type: ignore[attr-defined]
        mime = "image/png"
        raw = stream.getvalue()
    else:
        raise TypeError("images must be paths or PIL-compatible image objects")
    if not raw:
        raise ValueError("image payload is empty")
    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _validated_labels(labels: Iterable[str]) -> tuple[str, ...]:
    wanted = tuple(labels)
    if not wanted or any(not isinstance(label, str) or not label for label in wanted):
        raise ValueError("labels must be nonempty strings")
    if len(wanted) != len(set(wanted)):
        raise ValueError("labels must be distinct")
    return wanted


def first_token_top_logprobs(response: dict) -> dict[str, float]:
    try:
        content = response["choices"][0]["logprobs"]["content"]
        candidates = content[0]["top_logprobs"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderCapabilityError("response lacks first-token top_logprobs") from exc
    result: dict[str, float] = {}
    for item in candidates:
        token = item.get("token")
        logprob = item.get("logprob")
        if not isinstance(token, str) or not isinstance(logprob, (int, float)):
            raise ProviderCapabilityError("malformed top_logprobs entry")
        if not math.isfinite(float(logprob)):
            raise ProviderCapabilityError("nonfinite top_logprob")
        result[token] = float(logprob)
    return result


def exact_label_logits(response: dict, labels: Iterable[str]) -> dict[str, float]:
    wanted = _validated_labels(labels)
    if len(wanted) > 20:
        raise ProviderCapabilityError("OpenRouter top_logprobs cannot cover more than 20 labels")
    top = first_token_top_logprobs(response)
    missing = [label for label in wanted if label not in top]
    if missing:
        raise ProviderCapabilityError(f"incomplete label coverage: {missing}")
    return {label: top[label] for label in wanted}


def provider_reported_cost(response: dict) -> float | None:
    usage = response.get("usage") or {}
    value = usage.get("cost")
    if isinstance(value, (int, float)) and math.isfinite(float(value)) and value >= 0:
        return float(value)
    return None


class OpenRouterReadoutClient:
    def __init__(
        self,
        *,
        ledger: ProviderBudgetLedger,
        api_key: str | None = None,
        timeout_s: float = 60,
    ):
        self.ledger = ledger
        self.api_key = api_key or load_api_key()
        self.timeout_s = timeout_s

    def readout(
        self,
        *,
        model: str,
        prompt: str,
        image_paths: list[object],
        labels: list[str],
        price_bound: OpenRouterPriceBound,
        provider_only: list[str] | None = None,
        top_logprobs: int = 20,
        reasoning_enabled: bool | None = None,
        label_token_ids: Mapping[str, int] | None = None,
        common_label_logit_bias: float | None = None,
    ) -> dict:
        if not image_paths:
            raise ValueError("at least one image is required")
        labels = list(_validated_labels(labels))
        if top_logprobs < 1 or top_logprobs > 20:
            raise ValueError("top_logprobs must be in [1, 20]")
        if len(labels) > top_logprobs:
            raise ValueError("top_logprobs must be at least the label count")
        if not model or not prompt:
            raise ValueError("model and prompt are required")
        if (label_token_ids is None) != (common_label_logit_bias is None):
            raise ValueError("label token IDs and common bias must be provided together")
        if label_token_ids is not None:
            if set(label_token_ids) != set(labels):
                raise ValueError("label token IDs must cover exactly the requested labels")
            if len(set(label_token_ids.values())) != len(labels) or any(
                not isinstance(value, int) or value < 0 for value in label_token_ids.values()
            ):
                raise ValueError("label token IDs must be distinct nonnegative integers")
            if not isinstance(common_label_logit_bias, (int, float)) or not -100 <= common_label_logit_bias <= 100:
                raise ValueError("common label logit bias must be in [-100, 100]")
        image_urls = [image_data_url(image) for image in image_paths]
        image_sha256 = [
            hashlib.sha256(base64.b64decode(url.split(",", 1)[1])).hexdigest()
            for url in image_urls
        ]
        max_tokens = 1
        max_cost_usd = price_bound.request_bound_usd(
            prompt=prompt,
            image_count=len(image_paths),
            max_completion_tokens=max_tokens,
        )
        reservation = self.ledger.reserve(
            max_cost_usd,
            {
                "provider": "openrouter",
                "model": model,
                "purpose": "image_logprob_capability_probe",
                "label_count": len(labels),
                "top_logprobs": top_logprobs,
                "price_source": price_bound.source_checked_at,
                "max_prompt_tokens": price_bound.context_tokens,
                "max_completion_tokens": max_tokens,
                "equal_label_logit_bias": common_label_logit_bias,
            },
        )
        content = [{"type": "text", "text": prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": url}}
            for url in image_urls
        )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "logprobs": True,
            "top_logprobs": top_logprobs,
            "usage": {"include": True},
            "provider": {
                "allow_fallbacks": False,
                "require_parameters": True,
                **({"only": provider_only} if provider_only else {}),
            },
        }
        if reasoning_enabled is not None:
            payload["reasoning"] = {"enabled": reasoning_enabled}
        if label_token_ids is not None:
            payload["logit_bias"] = {
                str(token_id): common_label_logit_bias
                for token_id in label_token_ids.values()
            }
        started = time.perf_counter()
        try:
            result = httpx.post(
                API_URL,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=self.timeout_s,
            )
        except httpx.HTTPError as exc:
            # The provider may have accepted and billed the request. Keep the
            # reservation open for later billing reconciliation.
            self.ledger.record_pending_note(
                reservation,
                reason="transport_error_charge_unknown",
                details={"error_type": type(exc).__name__},
            )
            raise
        elapsed = time.perf_counter() - started
        if not result.is_success:
            error_summary: dict[str, object] = {}
            try:
                error = result.json().get("error") or {}
                if isinstance(error, dict):
                    for key in ("code", "type", "message"):
                        if key in error:
                            error_summary[key] = str(error[key])[:500]
            except (ValueError, AttributeError):
                pass
            self.ledger.release_confirmed_no_charge(
                reservation,
                status_code=result.status_code,
                reason="openrouter_http_error",
                details=error_summary,
            )
            retry_after = result.headers.get("retry-after")
            try:
                retry_after_s = float(retry_after) if retry_after is not None else None
            except ValueError:
                retry_after_s = None
            raise ProviderHTTPError(result.status_code, error_summary, retry_after_s)
        response = result.json()
        cost = provider_reported_cost(response)
        if cost is not None:
            self.ledger.settle(
                reservation,
                cost,
                {
                    "provider": response.get("provider"),
                    "model": response.get("model"),
                    "request_id": response.get("id"),
                    "usage": response.get("usage"),
                },
            )
        else:
            self.ledger.record_pending_note(
                reservation,
                reason="successful_response_missing_provider_cost",
                details={"request_id": response.get("id")},
            )
        try:
            provider_label_logprobs = exact_label_logits(response, labels)
            logits = (
                {
                    label: value - float(common_label_logit_bias)
                    for label, value in provider_label_logprobs.items()
                }
                if common_label_logit_bias is not None
                else provider_label_logprobs
            )
            capability = {"passed": True, "reason": None}
        except ProviderCapabilityError as exc:
            # Capability failures are evidence, so return and persist the raw
            # provider response while withholding an incomplete distribution.
            logits = None
            provider_label_logprobs = None
            capability = {"passed": False, "reason": str(exc)}
        choice = response["choices"][0]
        raw_generated_text = (choice.get("message") or {}).get("content")
        selected_label = max(logits, key=logits.get) if logits is not None else None
        return {
            "schema_version": 1,
            "request_id": response.get("id"),
            "provider": response.get("provider"),
            "model": response.get("model", model),
            "requested_model": model,
            "labels": labels,
            "label_logits": logits,
            "provider_label_logprobs": provider_label_logprobs,
            "logit_bias_audit": {
                "applied": common_label_logit_bias is not None,
                "common_bias": common_label_logit_bias,
                "token_ids": dict(label_token_ids) if label_token_ids is not None else None,
                "correction": "subtracted common bias; logits remain identifiable only up to an additive constant",
            },
            "capability": capability,
            "selected_label": selected_label,
            "raw_generated_text": raw_generated_text,
            "finish_reason": choice.get("finish_reason"),
            "token_audit": {
                "source": "provider_response",
                "status": "actual",
                "usage": response.get("usage"),
                "all_labels_returned_as_first_token_candidates": capability["passed"],
            },
            "timing": {
                "total_network_s": elapsed,
                "source": "client_wall_clock",
                "prefill_s": None,
                "readout_s": None,
                "unsupported_reason": "OpenRouter response does not separate image encoding, prefill, and readout",
            },
            "cost": {
                "actual_usd": cost,
                "source": "provider_response" if cost is not None else "unknown_pending_reservation",
                "reservation_id": reservation,
            },
            "raw_provider_response": response,
            "request_audit": {
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "ordered_image_sha256": image_sha256,
                "provider_only": list(provider_only) if provider_only else None,
                "reasoning_enabled_requested": reasoning_enabled,
                "top_logprobs": top_logprobs,
                "max_tokens": max_tokens,
            },
        }

    def decide_many(
        self,
        *,
        context: str,
        images: list[object],
        questions: list[Mapping[str, object]],
        model: str,
        provider: str,
        price_bound: OpenRouterPriceBound | None = None,
        top_logprobs: int = 20,
        reasoning_enabled: bool | None = None,
        label_token_ids: Mapping[str, int] | None = None,
        common_label_logit_bias: float | None = None,
    ) -> dict:
        """Adapt factorized grid questions to independent exact-readout calls.

        OpenRouter exposes no shared-prefill primitive, so each question is a
        separately billed request and the response states that explicitly.
        """
        if not questions:
            raise ValueError("at least one question is required")
        bound = price_bound or PRICE_BOUNDS.get((model, provider))
        if bound is None:
            raise ValueError("no reviewed price bound for model/provider")
        outputs = []
        for index, question in enumerate(questions):
            options = question.get("options")
            raw_labels = question.get("labels")
            if isinstance(options, Mapping):
                labels = list(options.keys())
            elif isinstance(raw_labels, list):
                labels = raw_labels
            else:
                raise ValueError(f"question {index} lacks ordered options or labels")
            suffix = question.get("prompt_suffix", question.get("prompt", ""))
            if not isinstance(suffix, str) or not suffix:
                raise ValueError(f"question {index} lacks prompt text")
            option_lines = ""
            if isinstance(options, Mapping):
                option_lines = "\nAllowed options in order:\n" + "\n".join(
                    f"{label} = {options[label]}" for label in labels
                )
            record = self.readout(
                model=model,
                prompt=f"{context}\n\n{suffix}{option_lines}\nReturn exactly one allowed label.",
                image_paths=images,
                labels=labels,
                price_bound=bound,
                provider_only=[provider],
                top_logprobs=top_logprobs,
                reasoning_enabled=reasoning_enabled,
                label_token_ids=(
                    {label: label_token_ids[label] for label in labels}
                    if label_token_ids is not None
                    else None
                ),
                common_label_logit_bias=common_label_logit_bias,
            )
            logits = record["label_logits"]
            probabilities = None
            if logits is not None:
                peak = max(logits.values())
                weights = {label: math.exp(value - peak) for label, value in logits.items()}
                denominator = sum(weights.values())
                probabilities = {label: value / denominator for label, value in weights.items()}
            outputs.append(
                {
                    "name": question.get("name", question.get("kind", f"question_{index}")),
                    "kind": question.get("kind"),
                    "options": dict(options) if isinstance(options, Mapping) else None,
                    "label_logits": logits,
                    "provider_label_logprobs": record["provider_label_logprobs"],
                    "logit_bias_audit": record["logit_bias_audit"],
                    "probabilities": probabilities,
                    "probability_scope": "normalized_over_requested_labels" if probabilities else None,
                    "selected_label": record["selected_label"],
                    "raw_generated_text": record["raw_generated_text"],
                    "capability": record["capability"],
                    "raw_provider_response": record["raw_provider_response"],
                    "request_audit": record.get("request_audit"),
                    "revision": {
                        "provider_model": record["model"],
                        "exact_weight_revision": None,
                        "status": "unavailable",
                        "reason": "OpenRouter does not expose the served weight commit",
                    },
                    "token_audit": record["token_audit"],
                    "timing": record["timing"],
                    "cost": record["cost"],
                }
            )
        return {
            "schema_version": 1,
            "backend": "openrouter",
            "requested_model": model,
            "provider": provider,
            "questions": outputs,
            "shared_prefill": False,
            "shared_prefill_reason": "OpenRouter chat completions exposes no shared-prefill API",
            "model_calls": len(outputs),
        }


def sanitized_record(record: dict) -> dict:
    """Return the persisted probe record; no request headers or key are present."""
    return json.loads(json.dumps(record))
