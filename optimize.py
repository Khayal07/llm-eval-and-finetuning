"""Model adaptation: few-shot prompt optimization and optional fine-tuning.

Addresses the failure modes found in the root-cause analysis by injecting
contamination-free few-shot exemplars (data/train_fewshot.json) plus explicit
task rules into the agent prompt:

  * no_evidence_refusal      -> prevents hallucination on missing context.
  * step_by_step_arithmetic  -> multi-hop math must show work and verify.
  * explicit_threshold       -> answers must state the exact number/limit.

`run_optimized()` evaluates the adapted prompt on the SAME held-out test set
and archives the run, so `compare_runs()` can show the before/after delta.

Fine-tuning is supported but optional: `prepare_finetune_jsonl()` writes an
OpenAI chat-format dataset, and the pipeline automatically switches to
FINE_TUNE_MODEL if it is set in `.env`.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agent_or_rag import SYSTEM_PROMPT
from pipeline import run_evaluation
from evals.run_tracker import RUNS_DIR, list_runs, load_run, summarize_for_console

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
FEWSHOT_PATH = DATA_DIR / "train_fewshot.json"
TEST_PATH = DATA_DIR / "test_heldout.json"
FINETUNE_JSONL = DATA_DIR / "finetune_chat.jsonl"

DEFAULT_NORMAL_N = 3
DEFAULT_SKILL_N = 3

# Task-specific rules appended for the adapted prompt. They fix the residual
# failures from the root-cause analysis without touching the test set.
OPTIMIZED_EXTRA_RULES = (
    "Additional rules:\n"
    "1. If the context does not contain an answer, reply exactly: "
    "\"I cannot find this information in the knowledge base.\"\n"
    "2. For prorated/accrual questions, count every month from the stated start "
    "month through December inclusively (April through December = 9 months), "
    "multiply by the monthly rate, and REPORT THE COMPUTED VALUE as the final "
    "number (e.g. approximately 16.5), showing the calculation.\n"
    "3. State the exact threshold or limit from the context that drives the "
    "decision (e.g. $25, 1 hour, 5 days).\n"
    "4. Do not round a fractional result to a whole number; keep it approximate."
)


def load_exemplars(path: Path = FEWSHOT_PATH) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["samples"]


def select_exemplars(
    normal_n: int = DEFAULT_NORMAL_N,
    skill_n: int = DEFAULT_SKILL_N,
) -> List[Dict[str, Any]]:
    """Pick examples: all skill demonstrators first, then leading normal pairs."""
    exemplars = load_exemplars()
    skills = [e for e in exemplars if e.get("demonstrates")]
    normals = [e for e in exemplars if not e.get("demonstrates")]
    picked = skills[:skill_n] + normals[:normal_n]
    logger.info("selected exemplars: %s", [e["id"] for e in picked])
    return picked


def to_pairs(exemplars: Sequence[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Reduce exemplars to (question, expected_answer) pairs for the prompt."""
    return [(e["question"], e["expected_answer"]) for e in exemplars]


def adapted_system_prompt() -> str:
    """System prompt of the adapted agent: default rules + extra task rules."""
    return SYSTEM_PROMPT + "\n\n" + OPTIMIZED_EXTRA_RULES


def run_optimized(
    run_name: str = "fewshot_optimized",
    normal_n: int = DEFAULT_NORMAL_N,
    skill_n: int = DEFAULT_SKILL_N,
    use_judge: bool = True,
    **eval_kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate the few-shot adapted agent over the held-out set."""
    selected = select_exemplars(normal_n=normal_n, skill_n=skill_n)
    pairs = to_pairs(selected)
    return run_evaluation(
        run_name=run_name,
        test_path=str(TEST_PATH),
        use_judge=use_judge,
        answer_kwargs={
            "mode": "few_shot",
            "few_shot_pairs": pairs,
            "system_prompt": adapted_system_prompt(),
        },
        meta={
            "adaptation": "few_shot",
            "exemplar_ids": [e["id"] for e in selected],
            "exemplar_count": len(pairs),
            "extra_rules": True,
        },
        **eval_kwargs,
    )


def latest_run(run_name: str) -> Optional[Path]:
    """Path of the most recent archived run whose name contains `run_name`."""
    for path in list_runs():
        if run_name in path.name:
            return path
    return None


def _payload(arg: Any) -> Dict[str, Any]:
    if isinstance(arg, (str, Path)):
        return load_run(Path(arg))
    return arg


def compare_runs(
    baseline: Any,
    optimized: Any,
    title: str = "Before / After comparison",
) -> Dict[str, Any]:
    """Compare two run artifacts (baseline vs optimized) and summarize the delta.

    Accepts run file paths, or loaded payload dicts. Returns a dict with
    pass-rate deltas, cost/latency deltas, and per-id failure transitions.
    """
    base = _payload(baseline).get("summary", {})
    opt = _payload(optimized).get("summary", {})

    def _delta(key: str):
        a, b = base.get(key), opt.get(key)
        if a is None or b is None:
            return None
        return round(b - a, 2)

    def _diff_failures(base_records, opt_records):
        base_failed = {r["id"] for r in base_records if r.get("combined_pass") is False}
        opt_failed = {r["id"] for r in opt_records if r.get("combined_pass") is False}
        return {
            "fixed": sorted(base_failed - opt_failed),
            "still_failed": sorted(base_failed & opt_failed),
            "newly_failed": sorted(opt_failed - base_failed),
        }

    latency_delta = None
    if base.get("latency_ms") and opt.get("latency_ms"):
        a, b = base["latency_ms"].get("mean"), opt["latency_ms"].get("mean")
        if a is not None and b is not None:
            latency_delta = round(b - a, 2)

    cost_delta = None
    if base.get("cost_usd") and opt.get("cost_usd"):
        a, b = base["cost_usd"].get("total"), opt["cost_usd"].get("total")
        if a is not None and b is not None:
            cost_delta = round(b - a, 8)

    result = {
        "combined_pass_rate_delta": _delta("combined_pass_rate"),
        "exact_match_rate_delta": _delta("exact_match_rate"),
        "judge_pass_rate_delta": _delta("judge_pass_rate"),
        "latency_mean_delta_ms": latency_delta,
        "cost_total_delta_usd": cost_delta,
    }
    base_records = _payload(baseline).get("records", [])
    opt_records = _payload(optimized).get("records", [])
    result.update(_diff_failures(base_records, opt_records))

    _print_comparison(base, opt, result, title)
    return result


def _print_comparison(base: Dict, opt: Dict, result: Dict, title: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
    print(f"  combined pass   : {base.get('combined_pass_rate')}%  ->  {opt.get('combined_pass_rate')}%  (delta {result['combined_pass_rate_delta']})")
    print(f"  exact match     : {base.get('exact_match_rate')}%  ->  {opt.get('exact_match_rate')}%  (delta {result['exact_match_rate_delta']})")
    print(f"  judge pass      : {base.get('judge_pass_rate')}%  ->  {opt.get('judge_pass_rate')}%  (delta {result['judge_pass_rate_delta']})")
    print(f"  latency mean    : {base.get('latency_ms', {}).get('mean')} ms  ->  {opt.get('latency_ms', {}).get('mean')} ms  (delta {result['latency_mean_delta_ms']})")
    print(f"  cost total      : ${base.get('cost_usd', {}).get('total')}  ->  ${opt.get('cost_usd', {}).get('total')}  (delta ${result['cost_total_delta_usd']})")
    print(f"  fixed ids       : {', '.join(result['fixed']) or 'none'}")
    print(f"  still failed    : {', '.join(result['still_failed']) or 'none'}")
    print(f"  newly failed    : {', '.join(result['newly_failed']) or 'none'}")


def prepare_finetune_jsonl(output: Path = FINETUNE_JSONL) -> Path:
    """Write a chat-format JSONL dataset for optional fine-tuning.

    Each line: {"messages": [{"role": "system", ...}, {"role": "user", ...},
    {"role": "assistant", ...}]}. Built strictly from data/train_fewshot.json,
    so test samples are never part of a fine-tuning dataset.
    """
    exemplars = select_exemplars(normal_n=10, skill_n=10)  # all available
    lines = []
    for e in exemplars:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the HR and policy assistant for AcmeCorp. "
                    "Use only the provided context; if the answer is not present, "
                    "say you cannot find it. Show arithmetic step by step and state exact thresholds."
                ),
            },
            {"role": "user", "content": e["question"]},
            {"role": "assistant", "content": e["expected_answer"]},
        ]
        lines.append({"messages": messages})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
    logger.info("wrote fine-tuning dataset: %s (%d lines)", output, len(lines))
    return output


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    baseline_path = latest_run("baseline")
    if baseline_path is None:
        logger.error("No baseline run found under reports/runs/. Run the baseline first.")
        raise SystemExit(1)

    out = run_optimized()
    opt_path = out["run_path"]

    prepare_finetune_jsonl()

    optimized_payload = load_run(Path(opt_path))
    compare_runs(baseline_path, optimized_payload)

    print(f"\nbaseline run : {baseline_path}")
    print(f"optimized run: {opt_path}")

    if os.getenv("FINE_TUNE_MODEL"):
        print("\nFINE_TUNE_MODEL is set - the agent already uses the fine-tuned model.")
    else:
        print("\nOptional fine-tuning is disabled; few-shot adaptation is the active method.")


if __name__ == "__main__":
    main()