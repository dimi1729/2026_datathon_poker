"""
preflop.py - preflop hand strength by table lookup (no simulation at play time).

The tables are built offline by tools/build_preflop_table.py and loaded once
when this module is imported:

    preflop_equity(hole)      equity vs a random opponent hand (0..1)
    preflop_percentile(hole)  share of all starting hands with LOWER equity (0..1)

`hole` is the 5 card ints from obs.hole_cards (any order).

Indexing: a sorted hand c0 < c1 < c2 < c3 < c4 has rank
    C(c0,1) + C(c1,2) + C(c2,3) + C(c3,4) + C(c4,5)
(the combinatorial number system), a unique index in 0..376,991.
"""

from __future__ import annotations

import os
from math import comb

import numpy as np

NUM_HANDS = comb(36, 5)  # 376,992

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EQUITY_PATH = os.path.join(_DIR, "preflop_equity.npy")
SORTED_PATH = os.path.join(_DIR, "preflop_equity_sorted.npy")

# _BINOM[k][c] = C(c, k+1) for c in 0..35
_BINOM = [[comb(c, k + 1) for c in range(36)] for k in range(5)]
_BINOM_NP = np.array(_BINOM, dtype=np.int64)


def hand_rank(hole) -> int:
    """Combinatorial-number-system rank of a 5-card hand (any order)."""
    a, b, c, d, e = sorted(hole)
    return _BINOM[0][a] + _BINOM[1][b] + _BINOM[2][c] + _BINOM[3][d] + _BINOM[4][e]


def ranks_of(hands: np.ndarray) -> np.ndarray:
    """Vectorized hand_rank for an (n, 5) array of SORTED hands."""
    return sum(_BINOM_NP[k][hands[:, k]] for k in range(5))


def _load(path: str):
    try:
        arr = np.load(path)
    except FileNotFoundError:
        return None
    if arr.shape != (NUM_HANDS,):
        raise ValueError(f"{path} has shape {arr.shape}, expected ({NUM_HANDS},)")
    return arr


# float16 on disk; keep float32 in memory so lookups return plain fast floats.
_EQ = _load(EQUITY_PATH)
_SORTED = _load(SORTED_PATH)
EQUITY: list[float] | None = _EQ.astype(np.float32).tolist() if _EQ is not None else None
SORTED_EQUITY: np.ndarray | None = _SORTED.astype(np.float32) if _SORTED is not None else None


def preflop_equity(hole) -> float:
    """Equity of this hand vs a random 5-card hand, all-in preflop (0..1)."""
    if EQUITY is None:
        raise RuntimeError(f"missing {EQUITY_PATH}; run tools/build_preflop_table.py")
    return EQUITY[hand_rank(hole)]


def equity_percentile(equity: float) -> float:
    """Fraction of all starting hands with strictly lower equity (0..1)."""
    if SORTED_EQUITY is None:
        raise RuntimeError(f"missing {SORTED_PATH}; run tools/build_preflop_table.py")
    return float(np.searchsorted(SORTED_EQUITY, np.float32(equity), side="left")) / NUM_HANDS


def preflop_percentile(hole) -> float:
    """Percentile of this hand among all starting hands (0 = worst, ~1 = best)."""
    return equity_percentile(preflop_equity(hole))
