"""End-to-end pipeline runner.

Usage examples:
    python main.py --mode all               # baseline -> optimize -> compare
    python main.py --mode baseline          # run baseline only
    python main.py --mode optimize          # run few-shot adapted evaluation
    python main.py --mode compare           # compare last baseline vs last optimized
    python main.py --mode validate          # dataset integrity + contamination check
    python main.py --mode all --skip-judge  # exact-match only (no LLM-as-Judge)

Every run is archived under reports/runs/ (timestamped) and does not overwrite
history. Run logs are shown as human-readable metric summaries.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from agent_or_rag import load_knowledge_base
from optimize import compare_runs, latest_run, run_optimized
from pipeline import run_evaluation

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_TEST_PATH = DATA_DIR / "test_heldout.json"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("main")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ai-eval-finetune-framework",
        description="Automated LLM/RAG evaluation, metrics tracking, root-cause "
        "analysis and few-shot adaptation.",
    )
    parser.add_argument(
        "--mode",
        choices=["baseline", "optimize", "compare", "all", "validate"],
        default="all",
        help="Pipeline stage to run (default: all).",
    )
    parser.add_argument("--run-name", default=None, help="Override the archived run name.")
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Disable LLM-as-Judge scoring; use exact-match only.",
    )
    parser.add_argument(
        "--test-path",
        default=str(DEFAULT_TEST_PATH),
        help=f"Test set JSON path (default: {DEFAULT_TEST_PATH}).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    return parser.parse_args()


# --------------------------------------------------------------------------- #
# Data integrity
# --------------------------------------------------------------------------- #
def validate_data() -> None:
    """Check dataset integrity and test-set contamination safeguards."""
    full = json.loads((DATA_DIR / "dataset_full.json").read_text(encoding="utf-8"))
    few = json.loads((DATA_DIR / "train_fewshot.json").read_text(encoding="utf-8"))
    test = json.loads((DATA_DIR / "test_heldout.json").read_text(encoding="utf-8"))
    kb_ids = {d["id"] for d in full["knowledge_base"]}

    errors: List[str] = []
    if len(full["knowledge_base"]) != len(load_knowledge_base()):
        errors.append("knowledge_base mismatch between files")

    full_ids = [s["id"] for s in full["samples"]]
    few_ids = [s["id"] for s in few["samples"]]
    test_ids = [s["id"] for s in test["samples"]]

    if sorted(full_ids) != sorted(few_ids + test_ids):
        errors.append("dataset_full ids != train+test ids")
    if few["meta"]["count"] != len(few_ids):
        errors.append("fewshot meta count mismatch")
    if test["meta"]["count"] != len(test_ids):
        errors.append("heldout meta count mismatch")

    overlap = sorted(set(few_ids) & set(test_ids))
    if overlap:
        errors.append(f"id overlap between splits: {overlap}")

    for s in full["samples"] + few["samples"]:
        for doc in s["kb_doc_ids"]:
            if doc not in kb_ids:
                errors.append(f"{s['id']}: unknown kb doc {doc}")
        if s.get("edge_type") == "out_of_scope" and s["kb_doc_ids"]:
            errors.append(f"{s['id']}: out_of_scope must have empty kb_doc_ids")

    few_topics = {frozenset(s["kb_doc_ids"]) for s in few["samples"] if s["kb_doc_ids"]}
    test_topics = {frozenset(s["kb_doc_ids"]) for s in test["samples"] if s["kb_doc_ids"]}
    shared = few_topics & test_topics
    if shared:
        errors.append(f"kb-doc overlap between splits: {shared}")

    status = "PASS" if not errors else "FAIL"
    print(f"[validate] total={len(full_ids)} few={len(few_ids)} heldout={len(test_ids)} -> {status}")
    for err in errors:
        print(f"[validate]   - {err}")
    if errors:
        raise SystemExit(1)


# --------------------------------------------------------------------------- #
# Pipeline stages
# --------------------------------------------------------------------------- #
def cmd_baseline(args: argparse.Namespace) -> Dict[str, Any]:
    return run_evaluation(
        run_name=args.run_name or "baseline",
        test_path=args.test_path,
        use_judge=not args.skip_judge,
    )


def cmd_optimize(args: argparse.Namespace) -> Dict[str, Any]:
    return run_optimized(run_name=args.run_name or "fewshot_optimized", use_judge=not args.skip_judge)


def cmd_compare(args: argparse.Namespace) -> None:
    baseline = latest_run("baseline")
    optimized = latest_run("fewshot_optimized")
    if baseline is None or optimized is None:
        logger.error("Missing runs: baseline=%s optimized=%s", baseline, optimized)
        raise SystemExit(1)
    compare_runs(baseline, optimized)
    print(f"\nbaseline run : {baseline}")
    print(f"optimized run: {optimized}")


def main() -> None:
    args = _args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.mode == "validate":
        validate_data()
        return
    if args.mode == "baseline":
        cmd_baseline(args)
        return
    if args.mode == "optimize":
        cmd_optimize(args)
        return
    if args.mode == "compare":
        cmd_compare(args)
        return

    # all
    cmd_baseline(args)
    cmd_optimize(args)
    cmd_compare(args)


if __name__ == "__main__":
    main()