"""Metrics for LLM/RAG evaluation.

Covers the three tracked dimensions:
  * accuracy / pass-rate (exact-match rate, judge pass rate, combined pass rate)
  * latency (mean, median, p95 in milliseconds)
  * cost (estimated USD from prompt/completion token usage)
Aggregation is intentionally pure (no network I/O) so it is unit-testable.
"""

from __future__ import annotations

import logging
import statistics
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable, Iterator, Optional

logger = logging.getLogger(__name__)

# Reference pricing in USD per 1M tokens; overridable via .env
# (PRICING_INPUT_PER_1M / PRICING_OUTPUT_PER_1M) or by passing prices explicitly.
DEFAULT_PRICING: Dict[str, Dict[str, float]] = {
    "gpt-4o-mini": {"input_per_1m": 0.15, "output_per_1m": 0.60},
    "gpt-4o": {"input_per_1m": 2.50, "output_per_1m": 10.00},
    "gpt-4.1-mini": {"input_per_1m": 0.40, "output_per_1m": 1.60},
}

_CHARS_PER_TOKEN_FALLBACK = 4.0


def estimate_tokens_from_text(text: str) -> int:
    """Rough token estimate used when the API reports no usage info."""
    if not text:
        return 0
    return max(1, round(len(text) / _CHARS_PER_TOKEN_FALLBACK))


@contextmanager
def timer() -> Iterator[dict]:
    """Context manager that records elapsed milliseconds into `{"ms": ...}`.

    Usage::

        with timer() as t:
            ...
        latency_ms = t["ms"]
    """
    holder: Dict[str, float] = {"ms": 0.0}
    started = time.perf_counter()
    try:
        yield holder
    finally:
        holder["ms"] = (time.perf_counter() - started) * 1000.0


def cost_for_model(model: str) -> Dict[str, float]:
    """Return per-1M-token prices for a model, falling back to env overrides."""
    import os

    pricing = DEFAULT_PRICING.get(model, DEFAULT_PRICING["gpt-4o-mini"]).copy()

    def _env_float(key: str, default: float) -> float:
        raw = os.getenv(key, "")
        try:
            return float(raw) if raw.strip() else default
        except ValueError:
            logger.warning("Ignoring invalid numeric env variable %s=%r", key, raw)
            return default

    pricing["input_per_1m"] = _env_float("PRICING_INPUT_PER_1M", pricing["input_per_1m"])
    pricing["output_per_1m"] = _env_float("PRICING_OUTPUT_PER_1M", pricing["output_per_1m"])
    return pricing


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_per_1m: float,
    output_per_1m: float,
) -> float:
    """Estimate USD cost of a call from token usage."""
    return (prompt_tokens / 1_000_000.0) * input_per_1m + (
        completion_tokens / 1_000_000.0
    ) * output_per_1m


def _pct(value: float) -> float:
    return round(value * 100.0, 2)


def aggregate_scores(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize per-sample evaluation results into aggregate metrics.

    Expected per-sample keys (result schema):
      - judge_pass (bool, optional), exact_match (bool, optional)
      - latency_ms (float), cost_usd (float)
      - category, edge_type, id (str)
    """
    rows = list(results)
    if not rows:
        return {"n": 0, "note": "no results"}

    def _rate(rows: Iterable[Dict[str, Any]], key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return None
        return sum(1 for v in vals if v) / len(vals)

    latencies = sorted(r.get("latency_ms", 0.0) for r in rows if r.get("latency_ms") is not None)
    costs = [r.get("cost_usd", 0.0) for r in rows if r.get("cost_usd") is not None]

    exact_rate = _rate(rows, "exact_match")
    semantic_rate = _rate(rows, "semantic_pass")
    semantic_scores = [r["semantic_score"] for r in rows if r.get("semantic_score") is not None]

    judge_rows = [r for r in rows if r.get("judge_verdict") is not None or r.get("judge_pass") is not None]
    judge_rate = None
    if judge_rows:
        judge_verdicts = [r.get("judge_verdict", r.get("judge_pass")) for r in judge_rows]
        judge_rate = sum(1 for v in judge_verdicts if v) / len(judge_verdicts)

    combined_rows = [r for r in rows if r.get("combined_pass") is not None]
    combined_rate = None
    if combined_rows:
        combined_rate = sum(1 for r in combined_rows if r["combined_pass"]) / len(combined_rows)

    def _group_rate(key: str) -> Dict[str, Dict[str, Any]]:
        groups: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            label = r.get(key) or "unspecified"
            bucket = groups.setdefault(label, {"count": 0, "pass": 0, "fail": 0, "ids_failed": []})
            bucket["count"] += 1
            verdict = r.get("combined_pass", r.get("judge_verdict", r.get("judge_pass")))
            if verdict is None:
                continue
            if verdict:
                bucket["pass"] += 1
            else:
                bucket["fail"] += 1
                bucket["ids_failed"].append(r.get("id", "?"))
        for bucket in groups.values():
            bucket["pass_rate"] = (
                _pct(bucket["pass"] / bucket["count"]) if bucket["count"] else None
            )
        return groups

    return {
        "n": len(rows),
        "exact_match_rate": _pct(exact_rate) if exact_rate is not None else None,
        "semantic_pass_rate": _pct(semantic_rate) if semantic_rate is not None else None,
        "mean_semantic_score": (
            round(statistics.fmean(semantic_scores), 4) if semantic_scores else None
        ),
        "judge_pass_rate": _pct(judge_rate) if judge_rate is not None else None,
        "combined_pass_rate": _pct(combined_rate) if combined_rate is not None else None,
        "failed_ids": [r.get("id", "?") for r in rows if r.get("combined_pass") is False],
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 2) if latencies else None,
            "median": round(statistics.median(latencies), 2) if latencies else None,
            "p95": _percentile(latencies, 95) if latencies else None,
        },
        "cost_usd": {
            "total": round(sum(costs), 6) if costs else None,
            "mean": round(statistics.fmean(costs), 6) if costs else None,
        },
        "by_category": _group_rate("category"),
        "by_edge_type": _group_rate("edge_type"),
    }


def _percentile(sorted_samples: list, p: float) -> float:
    if not sorted_samples:
        return 0.0
    k = (len(sorted_samples) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_samples) - 1)
    return round(sorted_samples[f] + (sorted_samples[c] - sorted_samples[f]) * (k - f), 2)