"""Cross-repeat consistency.

Temperature is deliberately non-zero, so a model's answer varies between
identical calls. How much it varies is itself a quality signal: an unstable
model is hard to rely on in production even when its average score is good.

Groq exposes no embedding model, so similarity here is lexical: TF-IDF cosine
over word n-grams plus ROUGE-L (longest-common-subsequence F1). Both are
reported so the report can be honest that this measures wording overlap, not
semantic equivalence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations

from sklearn.feature_extraction.text import TfidfVectorizer

_TOKEN = re.compile(r"[a-z0-9][a-z0-9.,%$-]*")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def rouge_l(a: str, b: str) -> float:
    """LCS-based F1 between two token sequences."""
    x, y = tokenize(a), tokenize(b)
    if not x or not y:
        return 0.0

    # Rolling two-row LCS table keeps memory linear for long reports.
    prev = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        cur = [0] * (len(y) + 1)
        xi = x[i - 1]
        for j in range(1, len(y) + 1):
            if xi == y[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = cur[j - 1] if cur[j - 1] >= prev[j] else prev[j]
        prev = cur

    lcs = prev[len(y)]
    if lcs == 0:
        return 0.0
    precision = lcs / len(x)
    recall = lcs / len(y)
    return round(2 * precision * recall / (precision + recall), 4)


def tfidf_cosine(texts: list[str]) -> float:
    """Mean pairwise TF-IDF cosine similarity across the samples."""
    usable = [t for t in texts if t and t.strip()]
    if len(usable) < 2:
        return 1.0 if usable else 0.0

    try:
        matrix = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(usable)
    except ValueError:
        # Degenerate vocabulary (e.g. all stop words).
        return 0.0

    normed = matrix.multiply(1 / (_row_norms(matrix) + 1e-12))
    sims = (normed @ normed.T).toarray()
    pairs = [sims[i][j] for i, j in combinations(range(len(usable)), 2)]
    return round(float(sum(pairs) / len(pairs)), 4) if pairs else 1.0


def _row_norms(matrix):
    import numpy as np

    return np.sqrt(matrix.multiply(matrix).sum(axis=1))


def mean_pairwise_rouge(texts: list[str]) -> float:
    usable = [t for t in texts if t and t.strip()]
    if len(usable) < 2:
        return 1.0 if usable else 0.0
    pairs = [rouge_l(a, b) for a, b in combinations(usable, 2)]
    return round(sum(pairs) / len(pairs), 4) if pairs else 1.0


@dataclass
class ConsistencyResult:
    n_samples: int
    tfidf_cosine: float
    rouge_l: float

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "tfidf_cosine": self.tfidf_cosine,
            "rouge_l": self.rouge_l,
        }


def evaluate(texts: list[str]) -> ConsistencyResult:
    return ConsistencyResult(
        n_samples=len([t for t in texts if t and t.strip()]),
        tfidf_cosine=tfidf_cosine(texts),
        rouge_l=mean_pairwise_rouge(texts),
    )
