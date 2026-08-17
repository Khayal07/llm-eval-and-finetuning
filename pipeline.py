"""Evaluation pipeline orchestration.

Runs the agent (agent_or_rag.answer) over the held-out test set, scores each
sample with exact-match and (optionally) LLM-as-Judge, aggregates metrics,
persists the run artifact, and computes judge bias diagnostics.

This module is the single entry point reused by `main.py` for the default run
and by `optimize.py` for before/after comparisons.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent_or_rag import AnswerResult, answer as default_answer
from evals.bias_handler import length_bias_report
from evals.evaluator import exact_match, judge_score, semantic_pass, semantic_similarity
from evals.llm_client import LLMClient, LLMClientError
from evals.metrics import aggregate_scores, cost_for_model, estimate_cost
from evals.run_tracker import (
    make_result_record,
    save_run,
    summarize_for_console,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent


def _load_samples(path: str) -> List[Dict[str, Any]]:
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["samples"]


def _try_judge_client() -> Optional[LLMClient]:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    try:
        return LLMClient()
    except LLMClientError as exc:
        logger.warning("Judge disabled: %s", exc)
        return None


def run_evaluation(
    run_name: str,
    test_path: str = "data/test_heldout.json",
    answer_fn: Callable[..., AnswerResult] = default_answer,
    use_judge: bool = True,
    judge_model: Optional[str] = None,
    judge_on_fail_only: bool = False,
    answer_kwargs: Optional[Dict[str, Any]] = None,
    meta: Optional[Dict[str, Any]] = None,
    skip_final_console: bool = False,
) -> Dict[str, Any]:
    """Evaluate a model config over the held-out set and persist the run.

    Returns: {"records", "summary", "run_path", "bias", "judge_enabled"}
    """
    samples = _load_samples(test_path)
    answer_kwargs = dict(answer_kwargs or {})

    from agent_or_rag import load_knowledge_base

    kb_corpus = [doc["content"] for doc in load_knowledge_base()]

    completion_model = answer_kwargs.get("model") or os.getenv("EVAL_MODEL", "gpt-4o-mini")
    pricing = cost_for_model(completion_model)

    judge_client = _try_judge_client() if use_judge else None
    judge_enabled = judge_client is not None

    records: List[Dict[str, Any]] = []
    for sample in samples:
        result: AnswerResult = answer_fn(sample["question"], **answer_kwargs)

        em = exact_match(result.text, sample["expected_answer"])
        sem_score = semantic_similarity(result.text, sample["expected_answer"], corpus=kb_corpus)
        sem_pass = semantic_pass(sem_score)
        verdict, reason = None, ""
        if judge_enabled:
            if judge_on_fail_only and em:
                verdict, reason = True, "exact match"
            else:
                judge = judge_score(
                    judge_client,
                    sample["question"],
                    sample["expected_answer"],
                    result.text,
                    judge_model=judge_model,
                )
                verdict, reason = judge["verdict"], judge["reason"]
        else:
            verdict, reason = em, "exact-match fallback"

        cost = estimate_cost(
            result.prompt_tokens,
            result.completion_tokens,
            pricing["input_per_1m"],
            pricing["output_per_1m"],
        )
        records.append(
            make_result_record(
                sample=sample,
                generated_answer=result.text,
                model=completion_model,
                latency_ms=result.latency_ms,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                cost_usd=cost,
                exact_match=em,
                judge_verdict=verdict,
                judge_reason=reason,
                judge_model=judge_model,
                semantic_score=sem_score,
                semantic_pass=sem_pass,
                no_evidence=result.no_evidence,
                evidence_score=result.evidence_score,
            )
        )

    summary = aggregate_scores(records)
    run_path = save_run(
        run_name,
        meta={
            **(meta or {}),
            "judge_enabled": judge_enabled,
            "judge_on_fail_only": judge_on_fail_only,
            "completion_model": completion_model,
            "pricing": pricing,
        },
        records=records,
        summary=summary,
    )

    bias = length_bias_report(
        [
            {
                "score": int(r["combined_pass"]),
                "words": r["answer_length_words"],
            }
            for r in records
        ]
    )

    if not skip_final_console:
        print(summarize_for_console(summary, title=f"Run: {run_name}"))
        print(f"\nsaved run artifact: {run_path}")
        print(f"length-bias check : {bias.get('interpretation', 'n/a')}")

    return {
        "records": records,
        "summary": summary,
        "run_path": run_path,
        "bias": bias,
        "judge_enabled": judge_enabled,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_evaluation("baseline")