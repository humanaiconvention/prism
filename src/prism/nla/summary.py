"""Aggregate per-sample NLA explanations into a per-layer summary block.

The scanner samples several positions from a layer's hidden states, asks
the NLA explainer to verbalize each one, and then needs a single
human-readable summary for the layer.  This module provides that
reduction.  The implementation is intentionally simple — token frequency
over the explanation texts — because PRISM is not in the business of
generating new prose.  Callers who want a richer summary can run a
real summariser over ``[e.text for e in result.explanations]``.
"""

from __future__ import annotations

import re
from collections import Counter
from statistics import fmean, pstdev
from typing import Iterable, List

from .types import NLABatchResult, NLAExplanation


_STOPWORDS = frozenset(
    """
    a about an and are as at be been being but by can come could did do does
    for from had has have he her him his how i if in into is it its just like
    may me more most my no not of on or other our out over she some such than
    that the their them then there these they this those through to up was
    we were what when where which while who whom why will with would you your
    activation appears encode explanation layer mock model nla
    """.split()
)

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-]+")


def _content_tokens(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOPWORDS]


def summarize_layer(
    layer_idx: int,
    explanations: Iterable[NLAExplanation],
    *,
    top_k: int = 5,
) -> NLABatchResult:
    """Bundle per-sample explanations into a :class:`NLABatchResult`.

    The textual summary is a comma-joined list of the *top_k* most common
    content tokens across the batch's explanations, prefixed with a count
    of distinct themes.  When the batch is empty the summary is the
    string ``"no samples"``.
    """
    exps = list(explanations)
    n = len(exps)
    if n == 0:
        return NLABatchResult(
            layer_idx=layer_idx,
            n_samples=0,
            explanations=[],
            summary="no samples",
            mean_fve=float("nan"),
            fve_std=float("nan"),
        )

    fves = [e.reconstruction_fve for e in exps]
    mean_fve = float(fmean(fves))
    fve_std = float(pstdev(fves)) if n > 1 else 0.0

    counter: Counter[str] = Counter()
    for e in exps:
        counter.update(_content_tokens(e.text))

    if counter:
        top = [tok for tok, _ in counter.most_common(top_k)]
        summary = f"common themes ({len(top)}): " + ", ".join(top)
    else:
        summary = "no salient tokens"

    return NLABatchResult(
        layer_idx=layer_idx,
        n_samples=n,
        explanations=exps,
        summary=summary,
        mean_fve=mean_fve,
        fve_std=fve_std,
    )
