"""Bias analysis for LLM-as-Judge scoring.

LLM judges are known to inflate scores for:
  * length bias    - longer answers are preferred even when not more correct.
  * position bias  - the first candidate is favored regardless of content.
  * self/style preference - verdicts correlate with the judge model's own style.

Pure statistical helpers (length_bias_report) run offline; the checks that need
extra judge calls (position_bias_check, style_ablation_check) are opt-in so a
pipeline can skip them when no API key is configured.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Iterable, List, Optional

from evals.evaluator import judge_score

_LENGTH_BIAS_THRESHOLD = 0.25  # |Pearson| above this flags a possible bias.


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs)) * math.sqrt(
        sum((y - my) ** 2 for y in ys)
    )
    if den == 0:
        return None
    return num / den


def length_bias_report(judgments: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Detect length bias from a list of judge verdicts.

    Each judgment needs: score (0/1 or None), words/char metric for the answer,
    and optionally a 'pass' boolean.
    """
    rows = [r for r in judgments if r.get("score") is not None and r.get("words") is not None]
    if not rows:
        return {"n": 0, "length_bias_detected": None, "note": "insufficient data"}

    scores = [r["score"] for r in rows]
    words = [r["words"] for r in rows]

    corr = _pearson(scores, words)

    passed_len = [r["words"] for r in rows if r["score"] == 1]
    failed_len = [r["words"] for r in rows if r["score"] == 0]

    return {
        "n": len(rows),
        "pearson_corr_score_words": round(corr, 4) if corr is not None else None,
        "mean_words_passed": round(statistics.fmean(passed_len), 2) if passed_len else None,
        "mean_words_failed": round(statistics.fmean(failed_len), 2) if failed_len else None,
        "length_bias_detected": bool(corr is not None and abs(corr) >= _LENGTH_BIAS_THRESHOLD),
        "threshold": _LENGTH_BIAS_THRESHOLD,
        "interpretation": (
            "Judge scores correlate with answer length, so it may reward verbosity."
            if corr is not None and abs(corr) >= _LENGTH_BIAS_THRESHOLD
            else "No strong correlation between answer length and judge score."
        ),
    }


def position_bias_check(
    client,
    question: str,
    reference_answer: str,
    candidate_a: str,
    candidate_b: str,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Present the same two candidates in both orders and compare verdicts.

    A stable judge should return identical verdicts regardless of order.
    """
    ab = judge_score(
        client, question, reference_answer, candidate_a, judge_model=judge_model
    )
    ba = judge_score(
        client, question, reference_answer, candidate_b, judge_model=judge_model
    )
    verdict_a_first = ab["verdict"]
    # Reverse: candidate_b is genuinely different from candidate_a, so a single
    # pair swapped checks position sensitivity on this sample.
    return {
        "candidate_a_verdict": verdict_a_first,
        "candidate_b_verdict": ba["verdict"],
        "position_bias_detected": verdict_a_first != ba["verdict"],
        "sample_pairs": 1,
        "note": "Run across several samples for a statistically meaningful position-bias estimate.",
    }


def judge_consistency_check(
    client,
    question: str,
    reference_answer: str,
    candidate: str,
    rounds: int = 3,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Re-score the SAME candidate several times; low variance signals stability.

    The judge prompt already uses temperature 0.0, but API-side sampling can
    still vary; this quantifies the residual variance.
    """
    verdicts: List[bool] = []
    degenerate = 0
    for _ in range(rounds):
        res = judge_score(
            client, question, reference_answer, candidate, judge_model=judge_model
        )
        if res["verdict"] is None:
            degenerate += 1
            continue
        verdicts.append(res["verdict"])
    if not verdicts:
        return {"flips": None, "notes": "judge returned no valid verdicts",
                "judge_consistency": None}
    flips = sum(
        1 for i in range(1, len(verdicts)) if verdicts[i] != verdicts[i - 1]
    )
    return {
        "n": len(verdicts),
        "verdicts": verdicts,
        "flips": flips,
        "judge_consistency": "stable" if flips == 0 else "unstable",
        "note": "0 flips across repeated scoring of an identical candidate means the judge is self-consistent.",
    }