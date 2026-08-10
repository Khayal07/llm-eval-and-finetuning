"""Evaluation package for LLM/RAG outputs.

Exposes metric aggregation, exact-match + LLM-as-Judge scoring, and
judge bias analysis as a single public surface.
"""

from evals.bias_handler import length_bias_report, position_bias_check
from evals.evaluator import exact_match, judge_score, normalize_text
from evals.metrics import DEFAULT_PRICING, aggregate_scores, estimate_cost, timer

__all__ = [
    "DEFAULT_PRICING",
    "aggregate_scores",
    "estimate_cost",
    "exact_match",
    "judge_score",
    "length_bias_report",
    "normalize_text",
    "position_bias_check",
    "timer",
]