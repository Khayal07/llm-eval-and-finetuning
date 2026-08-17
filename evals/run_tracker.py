"""Run tracking for evaluation experiments.

Persists per-sample results together with their aggregate summary into
`reports/runs/` so experiments can be compared over time (e.g. baseline vs
few-shot-optimized) and audited later.

Each saved run is a single JSON file whose name encodes the UTC timestamp, so
re-running the pipeline never destroys previous runs.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "reports" / "runs"

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_name(run_name: str) -> str:
    cleaned = _SAFE_NAME_RE.sub("-", run_name).strip("-.")
    return cleaned or "run"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def make_result_record(
    sample: Dict[str, Any],
    generated_answer: str,
    model: str,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
    exact_match: bool,
    judge_verdict: Optional[bool] = None,
    judge_reason: str = "",
    judge_model: Optional[str] = None,
    semantic_score: Optional[float] = None,
    semantic_pass: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build a uniform, JSON-serializable record for one evaluated sample.

    `combined_pass` prefers the LLM-judge verdict (semantic) and falls back to
    exact-match when no judge verdict exists. `semantic_score`/`semantic_pass`
    are the deterministic cosine cross-check on the judge.
    """
    generated = generated_answer or ""
    return {
        "id": sample["id"],
        "question": sample["question"],
        "category": sample.get("category"),
        "edge_type": sample.get("edge_type"),
        "reference_answer": sample["expected_answer"],
        "model": model,
        "generated_answer": generated,
        "answer_length_chars": len(generated),
        "answer_length_words": len(generated.split()) if generated else 0,
        "latency_ms": round(latency_ms, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cost_usd": round(cost_usd, 8),
        "exact_match": bool(exact_match),
        "judge_verdict": judge_verdict,
        "judge_reason": judge_reason,
        "judge_model": judge_model,
        "semantic_score": (
            round(semantic_score, 4) if semantic_score is not None else None
        ),
        "semantic_pass": bool(semantic_pass) if semantic_pass is not None else None,
        "combined_pass": judge_verdict if judge_verdict is not None else bool(exact_match),
    }


def save_run(
    run_name: str,
    meta: Dict[str, Any],
    records: Iterable[Dict[str, Any]],
    summary: Dict[str, Any],
) -> Path:
    """Write a run artifact under reports/runs/ and return its path."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": run_name,
        "timestamp": now_utc(),
        "meta": meta,
        "summary": summary,
        "records": list(records),
    }
    path = RUNS_DIR / f"{now_utc()}__{_safe_name(run_name)}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_run(path: Path) -> Dict[str, Any]:
    """Load a previously saved run artifact."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_runs() -> List[Path]:
    """Return run files sorted newest-first."""
    if not RUNS_DIR.exists():
        return []
    return sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def summarize_for_console(summary: Dict[str, Any], title: str = "Evaluation summary") -> str:
    """Human-readable console rendering of an aggregate summary."""
    lines = ["=" * 60, title, "=" * 60]
    if summary.get("n") == 0:
        lines.append(f"  n: 0 ({summary.get('note', 'no results')})")
        return "\n".join(lines)

    lines.append(f"  samples          : {summary['n']}")
    lines.append(f"  exact match rate : {_fmt(summary['exact_match_rate'])}")
    lines.append(f"  judge pass rate  : {_fmt(summary['judge_pass_rate'])}")
    lines.append(f"  combined pass    : {_fmt(summary['combined_pass_rate'])}")

    lat = summary.get("latency_ms") or {}
    lines.append(f"  latency mean/med : {_fmt(lat.get('mean'))} / {_fmt(lat.get('median'))} ms (p95 {_fmt(lat.get('p95'))})")

    cost = summary.get("cost_usd") or {}
    lines.append(f"  cost total/mean  : ${_fmt(cost.get('total'))} / ${_fmt(cost.get('mean'))}")

    failed = summary.get("failed_ids") or []
    lines.append(f"  failed ids       : {', '.join(failed) if failed else 'none'}")

    categories = summary.get("by_category") or {}
    lines.append("  by category:")
    for name, bucket in sorted(categories.items()):
        lines.append(
            f"    {str(name):<14} n={bucket['count']:<3} pass_rate={_fmt(bucket.get('pass_rate'))}%"
        )

    edges = summary.get("by_edge_type") or {}
    interesting = {k: v for k, v in edges.items() if str(k) not in ("unspecified", "None", "NoneType")}
    if interesting:
        lines.append("  by edge type:")
        for name, bucket in sorted(interesting.items()):
            lines.append(
                f"    {str(name):<14} n={bucket['count']:<3} pass_rate={_fmt(bucket.get('pass_rate'))}%"
            )

    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)