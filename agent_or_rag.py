"""The AI model / prompt chain under evaluation: a retrieval-augmented policy agent.

Answers employee questions about AcmeCorp's fictional knowledge base using a
deterministic term-overlap retriever plus a chat model. Retrieval is intentionally
simple (rare-term weighted token overlap) so weak-retrieval failure modes can be
reproduced and then fixed later in the adaptation step.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from evals.llm_client import LLMClient
from evals.metrics import timer
from evals.text_utils import cosine_similarity

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_K = 3
DEFAULT_MIN_SIMILARITY = 0.12  # cosine gate below which retrieval counts as "no evidence".

_PUNCT_RE = re.compile(r"[\W_]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")

SYSTEM_PROMPT = """You are the HR and policy assistant for AcmeCorp (a fictional software company).

Answer the employee's question using ONLY the context documents provided below.

Rules:
- Reply with exact numbers, thresholds, and limits taken from the context.
- If the context documents do NOT contain the answer, reply exactly with:
  "I cannot find this information in the knowledge base."
  Never invent a policy, benefit, or number that is not in the context.
- Keep the answer concise and factual. Do not add opinions or advice beyond the policy."""


def _tokens(text: str) -> List[str]:
    if not text:
        return []
    reduced = _WS_RE.sub(" ", _PUNCT_RE.sub(" ", text.lower()))
    return [t for t in reduced.split() if t]


def min_similarity_from_env() -> float:
    """Retrieval evidence gate threshold from RETRIEVER_MIN_SIMILARITY (.env)."""
    raw = os.getenv("RETRIEVER_MIN_SIMILARITY", "")
    if not raw.strip():
        return DEFAULT_MIN_SIMILARITY
    try:
        return float(raw)
    except ValueError:
        logger.warning("Ignoring invalid RETRIEVER_MIN_SIMILARITY=%r", raw)
        return DEFAULT_MIN_SIMILARITY


def load_knowledge_base(path: Optional[Path] = None) -> List[Dict]:
    path = path or (DATA_DIR / "dataset_full.json")
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["knowledge_base"]


def build_index(docs: List[Dict]) -> Counter:
    """Document-frequency index used to give rare query terms more weight."""
    df: Counter = Counter()
    for doc in docs:
        for term in set(_tokens(doc["content"])):
            df[term] += 1
    return df


def retrieve(
    docs: List[Dict],
    query: str,
    k: int = DEFAULT_K,
    df_index: Optional[Counter] = None,
) -> List[Tuple[str, str, str, float]]:
    """Rank docs by rare-term-weighted token overlap. Pure, deterministic, testable."""
    df = df_index or build_index(docs)
    n = len(docs)
    query_terms = set(_tokens(query))
    scored: List[Tuple[float, Dict]] = []

    for doc in docs:
        doc_counts = Counter(_tokens(doc["content"]))
        score = 0.0
        for term in query_terms:
            if doc_counts[term]:
                idf = math.log((n + 1) / (df.get(term, 0) + 1)) + 1.0
                score += idf * doc_counts[term]
        if score > 0:
            scored.append((score, doc))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [
        (doc["id"], doc["title"], doc["content"], round(score, 6))
        for score, doc in scored[:k]
    ]


def retrieve_with_evidence_gate(
    docs: List[Dict],
    query: str,
    k: int = DEFAULT_K,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    df_index: Optional[Counter] = None,
    corpus: Optional[Sequence[str]] = None,
) -> Tuple[List[Tuple[str, str, str, float]], bool, Optional[float]]:
    """Top-k retrieval plus a no-evidence cosine gate.

    Returns (context, no_evidence, evidence_score). When the best-matching
    document's cosine similarity (IDF-weighted) falls below `min_similarity`,
    the context is emptied so the answer step receives the safe fallback
    message instead of weak, possibly irrelevant context. This protects against
    hallucination on out-of-scope queries.
    """
    df = df_index or build_index(docs)
    context = retrieve(docs, query, k=k, df_index=df)
    if not context:
        return [], True, 0.0
    if corpus is None:
        corpus = [d["content"] for d in docs]
    evidence_score = cosine_similarity(query, context[0][2], corpus=corpus)
    if evidence_score < min_similarity:
        return [], True, round(evidence_score, 4)
    return context, False, round(evidence_score, 4)


@dataclass
class AnswerResult:
    text: str
    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    model: str
    mode: str
    retrieved_docs: List[Tuple[str, str, str, float]] = field(default_factory=list)
    no_evidence: bool = False
    evidence_score: Optional[float] = None


def build_zero_shot_messages(
    question: str,
    context: List[Tuple[str, str, str, float]],
    system_prompt: Optional[str] = None,
) -> List[Dict]:
    blocks = []
    for doc_id, title, content, _score in context:
        blocks.append(f"[{doc_id}] {title}: {content}")
    context_text = (
        "\n".join(blocks)
        if blocks
        else "(no documents retrieved above the evidence similarity threshold)"
    )
    user = (
        "CONTEXT DOCUMENTS:\n"
        f"{context_text}\n\n"
        "QUESTION:\n"
        f"{question}\n\n"
        "Answer using only the context above."
    )
    return [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def answer(
    question: str,
    client: Optional[LLMClient] = None,
    model: Optional[str] = None,
    k: int = DEFAULT_K,
    mode: str = "zero_shot",
    few_shot_pairs: Optional[List[Tuple[str, str]]] = None,
    system_prompt: Optional[str] = None,
    kb: Optional[List[Dict]] = None,
    min_similarity: Optional[float] = None,
) -> AnswerResult:
    """Run the RAG chain for one question and return the structured result.

    `min_similarity` is the no-evidence cosine gate; when None it falls back to
    the RETRIEVER_MIN_SIMILARITY env variable / DEFAULT_MIN_SIMILARITY.
    """
    docs = list(kb) if kb is not None else load_knowledge_base()

    gate = min_similarity if min_similarity is not None else min_similarity_from_env()
    index = build_index(docs)
    corpus = [d["content"] for d in docs]
    context, no_evidence, evidence_score = retrieve_with_evidence_gate(
        docs,
        question,
        k=k,
        min_similarity=gate,
        df_index=index,
        corpus=corpus,
    )

    messages = build_zero_shot_messages(question, context, system_prompt=system_prompt)
    if mode == "few_shot" and few_shot_pairs:
        messages = _build_few_shot_messages(
            question, context, few_shot_pairs, system_prompt=system_prompt
        )

    client = client or LLMClient(default_model=model)
    with timer() as t:
        completion = client.complete(messages, model=model)

    return AnswerResult(
        text=completion.text.strip(),
        latency_ms=t["ms"],
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        model=completion.model or (model or client.default_model),
        mode=mode,
        retrieved_docs=context,
        no_evidence=no_evidence,
        evidence_score=evidence_score,
    )


def _build_few_shot_messages(
    question: str,
    context: List[Tuple[str, str, str, float]],
    few_shot_pairs: List[Tuple[str, str]],
    system_prompt: Optional[str] = None,
) -> List[Dict]:
    blocks = [f"[{doc_id}] {title}: {content}" for doc_id, title, content, _ in context]
    context_text = (
        "\n".join(blocks)
        if blocks
        else "(no documents retrieved above the evidence similarity threshold)"
    )
    user_parts = [
        "CONTEXT DOCUMENTS:",
        context_text,
        "",
        "Here are examples of expected answer style:",
    ]
    for i, (q, a) in enumerate(few_shot_pairs, start=1):
        user_parts.append(f"EXAMPLE {i}\nQ: {q}\nA: {a}")
    user_parts.extend(["", "Question:", question])
    return [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_parts)},
    ]


if __name__ == "__main__":
    sample_q = "How many vacation days do full-time employees receive each year?"
    import os

    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set; cannot answer without a model call.")
    else:
        result = answer(sample_q)
        print(f"mode={result.mode} model={result.model} latency={result.latency_ms:.0f}ms")
        print("retrieved:", [d[0] for d in result.retrieved_docs])
        print("answer:", result.text)