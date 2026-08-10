"""Evaluation package for LLM/RAG outputs.

Exposes metric aggregation, exact-match + LLM-as-Judge scoring, and
judge bias analysis as a single public surface.
"""

from evals.bias_handler import length_bias_report, position_bias_check
from evals.evaluator import exact_match, judge_score, normalize_text
from evals.metrics import DEFAULT_PRICING, aggregate_scores, estimate_cost, timer
from evals.run_tracker import (
    RUNS_DIR,
    list_runs,
    load_run,
    make_result_record,
    save_run,
    summarize_for_console,
)

__all__ = [
    "DEFAULT_PRICING",
    "RUNS_DIR",
    "aggregate_scores",
    "estimate_cost",
    "exact_match",
    "judge_score",
    "length_bias_report",
    "list_runs",
    "load_run",
    "make_result_record",
    "normalize_text",
    "position_bias_check",
    "save_run",
    "summarize_for_console",
    "timer",
]