"""Pure text/vector helpers shared by scoring and retrieval.

Kept deterministic and dependency-free so they are unit-testable offline and
can be reused by the evaluator (semantic similarity) and the retriever
(no-evidence cosine gate) without duplicating tokenization logic.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Optional, Sequence, Tuple

_PUNCT_RE = re.compile(r"[\W_]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation and collapse whitespace into tokens."""
    if not text:
        return []
    reduced = _WS_RE.sub(" ", _PUNCT_RE.sub(" ", text.lower()))
    return [t for t in reduced.split() if t]


def build_idf(corpus: Sequence[str]) -> Tuple[Counter, int]:
    """Document-frequency counter plus corpus size for IDF weighting."""
    n = len(corpus)
    df: Counter = Counter()
    for doc in corpus:
        for term in set(tokenize(doc)):
            df[term] += 1
    return df, n


def cosine_similarity(
    text_a: str,
    text_b: str,
    corpus: Optional[Sequence[str]] = None,
) -> float:
    """Cosine similarity over token vectors with optional IDF weighting.

    Returns a float in [0, 1]. With no corpus, plain bag-of-words cosine is
    used; when a corpus is given, shared rare terms get higher weight.
    """
    counts_a = Counter(tokenize(text_a))
    counts_b = Counter(tokenize(text_b))
    if not counts_a or not counts_b:
        return 0.0

    df: Counter = Counter()
    n = 0
    use_idf = bool(corpus)
    if use_idf:
        df, n = build_idf(corpus)

    def _weight(term: str) -> float:
        if not use_idf:
            return 1.0
        return math.log((n + 1) / (df.get(term, 0) + 1)) + 1.0

    vec_a = {t: c * _weight(t) for t, c in counts_a.items()}
    vec_b = {t: c * _weight(t) for t, c in counts_b.items()}

    dot = sum(vec_a[t] * vec_b.get(t, 0.0) for t in vec_a)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)