"""Bias analysis for LLM-as-Judge scoring.

LLM judges are known to inflate scores for:
  * length bias    - longer answers are preferred even when not more correct.
  * position bias  - the first candidate is favored regardless of content.
  * self/style preference - verdicts correlate with the judge model's own style.

Pure statistical helpers (length_bias_report) run offline; the probes that need
extra judge calls (position_bias_check, verbosity_robustness_check,
self_preference_check, judge_consistency_check) are opt-in via `run_bias_audit`
so a pipeline can skip them when no API key is configured.
"""

from __future__ import annotations

import math
import statistics
from typing import Any, Dict, Iterable, List, Optional, Sequence

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


def _flips(verdicts: List[Optional[bool]]) -> int:
    return sum(
        1 for i in range(1, len(verdicts)) if verdicts[i] != verdicts[i - 1]
    )


def position_bias_check(
    client,
    question: str,
    reference_answer: str,
    candidate: str,
    judge_model: Optional[str] = None,
    rounds: int = 1,
) -> Dict[str, Any]:
    """Grade the SAME candidate with reference and candidate order swapped.

    The judge prompt explicitly asks to ignore ordering; this probe verifies the
    verdict is stable regardless of which block appears first. Any flip between
    the two orderings signals position sensitivity.
    """
    verdicts: List[Optional[bool]] = []
    for candidate_first in (False, True) * rounds:
        res = judge_score(
            client,
            question,
            reference_answer,
            candidate,
            judge_model=judge_model,
            candidate_first=candidate_first,
        )
        verdicts.append(res["verdict"])
    return {
        "verdicts": verdicts,
        "flips": _flips(verdicts),
        "position_bias_detected": _flips(verdicts) > 0,
        "samples": len(verdicts),
        "note": "Verdicts must be identical when the answer/reference order is swapped.",
    }


def verbosity_robustness_check(
    client,
    question: str,
    reference_answer: str,
    terse_answer: str,
    verbose_answer: str,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Grade a short and a long phrasing of the SAME correct facts.

    Both convey the same key facts, so a stable judge must score them equally
    (usually both 1). A difference means the judge rewards or punishes
    verbosity/style despite the anti-length instruction.
    """
    terse = judge_score(
        client, question, reference_answer, terse_answer, judge_model=judge_model
    )
    verbose = judge_score(
        client, question, reference_answer, verbose_answer, judge_model=judge_model
    )
    return {
        "terse_verdict": terse["verdict"],
        "verbose_verdict": verbose["verdict"],
        "verbosity_bias_detected": terse["verdict"] != verbose["verdict"],
        "note": "Terse and verbose answers carry the same facts; equal verdicts expected.",
    }


def self_preference_check(
    client,
    question: str,
    reference_answer: str,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Detect whether the judge favors its own writing style.

    The judge model first writes its own candidate answer, then the same judge
    grades (a) its own answer and (b) a plain restatement of the reference.
    A judge that passes its own phrasing while failing an equally-correct plain
    restatement shows self/style preference.
    """
    gen_prompt = (
        "Answer the QUESTION below using ONLY the facts in the REFERENCE. "
        "Reply in one or two sentences, in your natural writing style.\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "REFERENCE:\n"
        f"{reference_answer}"
    )
    model = judge_model or None
    try:
        generated = client.complete(
            messages=[{"role": "user", "content": gen_prompt}],
            model=model,
            temperature=0.0,
        ).text.strip()
    except Exception:  # noqa: BLE001 - probe must degrade gracefully
        return {
            "generated": None,
            "own_verdict": None,
            "plain_verdict": None,
            "self_preference_detected": None,
            "note": "Judge could not generate a self-candidate; probe skipped.",
        }
    own = judge_score(
        client, question, reference_answer, generated, judge_model=judge_model
    )
    plain = judge_score(
        client, question, reference_answer, reference_answer, judge_model=judge_model
    )
    return {
        "generated": generated,
        "own_verdict": own["verdict"],
        "plain_verdict": plain["verdict"],
        "self_preference_detected": bool(own["verdict"] and not plain["verdict"]),
        "note": "Self-preference means the judge passes its own phrasing but not a plain correct restatement.",
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
    flips = _flips(verdicts)
    return {
        "n": len(verdicts),
        "verdicts": verdicts,
        "flips": flips,
        "judge_consistency": "stable" if flips == 0 else "unstable",
        "note": "0 flips across repeated scoring of an identical candidate means the judge is self-consistent.",
    }


def run_bias_audit(
    client,
    samples: Sequence[Dict[str, Any]],
    judge_model: Optional[str] = None,
    max_samples: int = 3,
    consistency_rounds: int = 3,
) -> Dict[str, Any]:
    """Run the opt-in bias probe suite over a small sample subset.

    Returns per-sample probe results plus the list of samples that triggered any
    bias flag. This is the "quality check trick" gate: the script must actually
    detect bias, not merely claim it is guarded against.
    """
    pool = [s for s in samples if s.get("category") == "edge_case"] or list(samples)
    pool = pool[:max_samples]

    probes: List[Dict[str, Any]] = []
    flagged: List[str] = []
    for sample in pool:
        ref = sample["expected_answer"]
        terse = ref
        verbose = f"{ref} This is because the policy states these limits and the rules apply to all employees."
        pos = position_bias_check(client, sample["question"], ref, ref, judge_model=judge_model)
        verb = verbosity_robustness_check(
            client, sample["question"], ref, terse, verbose, judge_model=judge_model
        )
        selfp = self_preference_check(
            client, sample["question"], ref, judge_model=judge_model
        )
        cons = judge_consistency_check(
            client, sample["question"], ref, ref, rounds=consistency_rounds, judge_model=judge_model
        )
        probe = {
            "id": sample["id"],
            "question": sample["question"],
            "position_bias_detected": bool(pos["position_bias_detected"]),
            "verbosity_bias_detected": bool(verb["verbosity_bias_detected"]),
            "self_preference_detected": bool(selfp.get("self_preference_detected")),
            "judge_consistency": cons.get("judge_consistency"),
        }
        if any(
            [
                probe["position_bias_detected"],
                probe["verbosity_bias_detected"],
                probe["self_preference_detected"],
                probe["judge_consistency"] == "unstable",
            ]
        ):
            flagged.append(sample["id"])
        probes.append(probe)

    return {
        "n": len(probes),
        "probes": probes,
        "flagged_ids": flagged,
        "bias_guards": {
            "length": "length_bias_report per run (Pearson |r| >= 0.25)",
            "position": "anti-position judge instruction + order-swap probe",
            "verbosity": "anti-length judge instruction + terse/verbose probe",
            "self_preference": "anti-style judge instruction + self-candidate probe",
            "consistency": "temperature-0.0 judge + repeated-scoring probe",
        },
        "summary": (
            "No bias flags triggered on the audited subset."
            if not flagged
            else f"Bias flags triggered on: {', '.join(flagged)}. "
            "Judge prompt mitigations are active; consider scoring with an alternative judge model."
        ),
    }