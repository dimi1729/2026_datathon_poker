"""
fasteval.py - vectorized PLO hand evaluation for Monte Carlo at play time.

omaha_score() costs ~60 us per call in pure Python, far too slow for hundreds
of samples per decision. Here we precompute, once at startup (~1-2 s):

    PT[pair_index, triple_index] = strength of (2 hole cards + 3 board cards)

for all 630 hole pairs x 7140 board triples (uint16, 9 MB). Strength is the
dense rank of evaluator.score_five's score, so higher is better and equal is
a tie, exactly like omaha_score. A full Omaha hand is then the max over
10 pairs x 10 triples = 100 array gathers, done for many hands at once in numpy.

Indices are combinatorial-number-system ranks of SORTED cards:
    pair (a<b):     a + C(b,2)
    triple (a<b<c): a + C(b,2) + C(c,3)
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

from evaluator import NUM_CARDS, get_table

N_PAIRS = comb(NUM_CARDS, 2)     # 630
N_TRIPLES = comb(NUM_CARDS, 3)   # 7140

_C2 = np.array([comb(c, 2) for c in range(NUM_CARDS)], dtype=np.int64)
_C3 = np.array([comb(c, 3) for c in range(NUM_CARDS)], dtype=np.int64)

# Index combos of 5 hole cards (pairs) and of 3/4/5 board cards (triples).
PAIR_POS = np.array(list(combinations(range(5), 2)), dtype=np.int64)          # (10, 2)
TRIPLE_POS = {n: np.array(list(combinations(range(n), 3)), dtype=np.int64)    # (1|4|10, 3)
              for n in (3, 4, 5)}


def _build() -> np.ndarray:
    table = get_table()  # {36-bit mask: score}
    pairs = np.array(list(combinations(range(NUM_CARDS), 2)), dtype=np.int64)
    triples = np.array(list(combinations(range(NUM_CARDS), 3)), dtype=np.int64)
    p_idx = pairs[:, 0] + _C2[pairs[:, 1]]
    t_idx = triples[:, 0] + _C2[triples[:, 1]] + _C3[triples[:, 2]]
    assert (np.sort(p_idx) == np.arange(N_PAIRS)).all()
    assert (np.sort(t_idx) == np.arange(N_TRIPLES)).all()

    # Dense-rank the distinct scores so they fit in uint16.
    masks = np.fromiter(table.keys(), dtype=np.int64, count=len(table))
    scores = np.fromiter(table.values(), dtype=np.int64, count=len(table))
    uniq, dense = np.unique(scores, return_inverse=True)
    order = np.argsort(masks)
    masks, dense = masks[order], (dense[order] + 1).astype(np.uint16)  # 0 = impossible

    pmask = (np.int64(1) << pairs[:, 0]) | (np.int64(1) << pairs[:, 1])
    tmask = ((np.int64(1) << triples[:, 0]) | (np.int64(1) << triples[:, 1])
             | (np.int64(1) << triples[:, 2]))
    out = np.zeros((N_PAIRS, N_TRIPLES), dtype=np.uint16)
    for i in range(N_PAIRS):
        m = pmask[i] | tmask
        ok = (pmask[i] & tmask) == 0
        pos = np.searchsorted(masks, m[ok])
        out[p_idx[i], t_idx[ok]] = dense[pos]
    return out


PT = _build()
PT_FLAT = PT.reshape(-1)


def pair_indices(hands: np.ndarray) -> np.ndarray:
    """(..., 5) SORTED hole cards -> (..., 10) pair indices."""
    a = hands[..., PAIR_POS[:, 0]]
    b = hands[..., PAIR_POS[:, 1]]
    return a + _C2[b]


def triple_indices(boards: np.ndarray) -> np.ndarray:
    """(..., n) SORTED board cards (n = 3, 4 or 5) -> (..., C(n,3)) triple indices."""
    pos = TRIPLE_POS[boards.shape[-1]]
    return boards[..., pos[:, 0]] + _C2[boards[..., pos[:, 1]]] + _C3[boards[..., pos[:, 2]]]


def best(pairs: np.ndarray, triples: np.ndarray) -> np.ndarray:
    """Best strength for each hand: pairs (..., 10), triples (..., T) -> (...)."""
    idx = pairs[..., :, None] * N_TRIPLES + triples[..., None, :]
    return PT_FLAT[idx].max(axis=(-2, -1))


def score(hole, board) -> int:
    """Strength of one hand on a 3-5 card board (comparable like omaha_score)."""
    h = np.sort(np.asarray(hole, dtype=np.int64))
    b = np.sort(np.asarray(board, dtype=np.int64))
    return int(best(pair_indices(h), triple_indices(b)))
