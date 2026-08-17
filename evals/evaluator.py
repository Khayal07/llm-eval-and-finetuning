"""Automated scoring for LLM/RAG outputs.

Two complementary scorers:
  1. exact_match  - strict normalized string equality (token-level accuracy).
  2. LLM-as-Judge - semantic grading via a reference answer rubric.

A `combined_pass` is intended to be computed at the pipeline level by
falling back to the judge when exact match fails.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from evals.llm_client import Completion, LLMClient
from evals.text_utils import cosine_similarity, tokenize

_PUNCT_RE = re.compile(r"[\W_]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")

# Cosine-similarity score above which the answer is treated as semantically
# matching the reference. A deterministic objective cross-check on the judge.
SEMANTIC_PASS_THRESHOLD = 0.35

_JUDGE_SYSTEM_PROMPT = """You are a strict but fair evaluation judge for a company-policy question-answering system.

Grade the AI answer against the provided REFERENCE answer for factual correctness.

Rules:
- Score 1 ONLY if the answer is factually correct: it contains the key facts/numbers from the reference (e.g. the right count, threshold, or policy) and does NOT add a wrong, invented, or unsupported claim.
- If the answer is correct but phrased differently, shorter, or in a different style, still score 1.
- IGNORE answer length. A correct short answer beats a long verbose guess.
- IGNORE writing style, politeness, and self-confidence. Never reward or punish tone.
- IGNORE answer position and ordering inside the prompt; grade the content, not where it appears.
- Do not favor any particular writing style, including the style you (the judge) would write yourself. Judge the facts only.
- If the answer misses a key fact, gives a wrong number, or hallucinates a benefit/policy that is not supported, score 0.
- If the answer says it cannot find/confirm information for a question whose reference itself states the policy is not documented, that is CORRECT.

Respond ONLY with a JSON object, no extra text:
{"score": 0 or 1, "reason": "<one sentence explaining the verdict>"}"""


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace for comparison."""
    if text is None:
        return ""
    lowered = text.lower()
    lowered = _PUNCT_RE.sub(" ", lowered)
    return _WS_RE.sub(" ", lowered).strip()


def exact_match(generated: str, expected: str) -> bool:
    """Strict normalized equality. Exact-match only; see judge_score for fuzziness."""
    if generated is None or expected is None:
        return False
    return normalize_text(generated) == normalize_text(expected)


def semantic_similarity(
    generated: str,
    expected: str,
    corpus: Optional[Sequence[str]] = None,
) -> float:
    """Deterministic cosine-similarity score between a generated answer and a
    reference (optionally IDF-weighted over the knowledge base). Returns [0, 1].

    Serves as an objective, judge-independent cross-check on LLM-as-Judge.
    """
    return cosine_similarity(generated or "", expected or "", corpus=corpus)


def semantic_pass(
    score: float,
    threshold: float = SEMANTIC_PASS_THRESHOLD,
) -> bool:
    """Whether a semantic-similarity score clears the pass threshold."""
    return score is not None and score >= threshold


def _parse_judge_response(raw: str) -> dict:
    """Robust extraction of the expected JSON object from a judge response."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        text = match.group(0)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # Last resort: find a bare 0/1 token and treat it as the score.
    score_match = re.search(r"\b([01])\b", raw)
    if score_match:
        return {"score": int(score_match.group(1)), "reason": raw}
    return {"score": None, "reason": raw}


def judge_score(
    client: LLMClient,
    question: str,
    reference_answer: str,
    generated_answer: str,
    judge_model: Optional[str] = None,
    candidate_first: bool = False,
) -> Dict[str, Any]:
    """Grade a generated answer with an LLM judge.

    Returns a dict with keys: score (0/1/None), verdict (bool or None),
    reason (str), completion (Completion), judge_model (str).

    `candidate_first=True` swaps reference/answer order in the prompt; used by
    the position-bias probe to verify verdict stability across orderings.
    """
    model = judge_model or _judge_model_from_env()
    if candidate_first:
        user_prompt = (
            "QUESTION:\n"
            f"{question}\n\n"
            "AI ANSWER TO GRADE:\n"
            f"{generated_answer}\n\n"
            "REFERENCE ANSWER:\n"
            f"{reference_answer}\n\n"
            "Respond with JSON: {\"score\": 0 or 1, \"reason\": \"...\"}."
        )
    else:
        user_prompt = (
            "QUESTION:\n"
            f"{question}\n\n"
            "REFERENCE ANSWER:\n"
            f"{reference_answer}\n\n"
            "AI ANSWER TO GRADE:\n"
            f"{generated_answer}\n\n"
            "Respond with JSON: {\"score\": 0 or 1, \"reason\": \"...\"}."
        )
    completion = client.complete(
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=model,
        temperature=_judge_temperature_from_env(),
        json_mode=True,
    )
    parsed = _parse_judge_response(completion.text)
    score = parsed.get("score")
    verdict = bool(score) if score in (0, 1) else None
    return {
        "score": score,
        "verdict": verdict,
        "reason": parsed.get("reason", ""),
        "completion": completion,
        "judge_model": model,
    }


def _judge_model_from_env() -> str:
    import os

    return os.getenv("JUDGE_MODEL", "gpt-4o")


def _judge_temperature_from_env() -> float:
    import os

    return float(os.getenv("JUDGE_TEMPERATURE", "0.0"))