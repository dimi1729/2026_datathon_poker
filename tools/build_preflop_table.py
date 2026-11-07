"""
build_preflop_table.py - build the offline preflop equity table (run once, offline).

    python tools/build_preflop_table.py [--trials 1000] [--workers N] [--seed 0]

Steps:
  1. Enumerate all C(36,5) = 376,992 starting hands (sorted 5-card tuples).
  2. Canonicalize each one under suit symmetry: apply all 24 suit
     permutations, sort the cards, keep the lexicographically smallest tuple.
  3. Monte Carlo the equity of each canonical class vs a random opponent:
     deal the opponent 5 cards and a 5-card board from the remaining 31,
     compare omaha_score, ties count 0.5. Parallelized with multiprocessing.
  4. Expand back to every hand and save, indexed by the combinatorial-number-
     system rank of the sorted cards (see preflop.hand_rank):
        data/preflop_equity.npy         float16[376992]  equity per hand
        data/preflop_equity_sorted.npy  float16[376992]  same values, sorted
                                                         (equity -> percentile)
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from itertools import combinations, permutations
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from evaluator import NUM_CARDS, get_table  # noqa: E402
from preflop import EQUITY_PATH, NUM_HANDS, SORTED_PATH, ranks_of  # noqa: E402

_PAIRS = tuple(combinations(range(5), 2))
_TRIPLES = tuple(combinations(range(5), 3))


def all_hands() -> np.ndarray:
    """All sorted 5-card hands, shape (376992, 5), in the order of their rank."""
    hands = np.array(list(combinations(range(NUM_CARDS), 5)), dtype=np.int64)
    order = np.argsort(ranks_of(hands))      # put row i at rank i
    hands = hands[order]
    assert (ranks_of(hands) == np.arange(NUM_HANDS)).all()
    return hands


def canonical_keys(hands: np.ndarray) -> np.ndarray:
    """For each hand, the base-36 key of its lexicographically smallest
    suit-permuted sorted tuple (numeric order of the key == lex order)."""
    rank, suit = hands >> 2, hands & 3
    weights = 36 ** np.arange(4, -1, -1, dtype=np.int64)   # first card most significant
    best = None
    for perm in permutations(range(4)):
        mapped = np.sort(rank * 4 + np.array(perm)[suit], axis=1)
        key = mapped @ weights
        best = key if best is None else np.minimum(best, key)
    return best


def key_to_hand(key: int) -> tuple[int, ...]:
    out = []
    for _ in range(5):
        key, c = divmod(key, 36)
        out.append(c)
    return tuple(reversed(out))


# ----- Monte Carlo (worker side) ---------------------------------------------

_TABLE: dict[int, int] | None = None


def _init_worker() -> None:
    global _TABLE
    _TABLE = get_table()


def _best(hole_masks, board_masks, table) -> int:
    best = 0
    for hm in hole_masks:
        for bm in board_masks:
            s = table[hm | bm]
            if s > best:
                best = s
    return best


def equity_of(args: tuple[tuple[int, ...], int, int]) -> float:
    """Equity of one hand vs a random opponent over `trials` random deals."""
    hand, trials, seed = args
    table = _TABLE
    rng = random.Random(seed)
    deck = [c for c in range(NUM_CARDS) if c not in hand]
    bits = [1 << c for c in hand]
    hero_masks = [bits[i] | bits[j] for i, j in _PAIRS]
    won = 0.0
    for _ in range(trials):
        cards = rng.sample(deck, 10)
        ob = [1 << c for c in cards[:5]]
        bb = [1 << c for c in cards[5:]]
        board_masks = [bb[i] | bb[j] | bb[k] for i, j, k in _TRIPLES]
        opp_masks = [ob[i] | ob[j] for i, j in _PAIRS]
        h = _best(hero_masks, board_masks, table)
        o = _best(opp_masks, board_masks, table)
        won += 1.0 if h > o else 0.5 if h == o else 0.0
    return won / trials


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trials", type=int, default=1000, help="Monte Carlo deals per class")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    t0 = time.perf_counter()
    hands = all_hands()
    keys = canonical_keys(hands)
    class_keys, class_of_hand, class_sizes = np.unique(keys, return_inverse=True, return_counts=True)
    print(f"{NUM_HANDS} hands -> {len(class_keys)} suit-isomorphic classes "
          f"({time.perf_counter() - t0:.1f}s)", flush=True)

    jobs = [(key_to_hand(int(k)), args.trials, args.seed * 1_000_003 + i)
            for i, k in enumerate(class_keys)]
    t1 = time.perf_counter()
    class_eq = np.empty(len(jobs), dtype=np.float64)
    with Pool(args.workers, initializer=_init_worker) as pool:
        for i, eq in enumerate(pool.imap(equity_of, jobs, chunksize=64)):
            class_eq[i] = eq
            if (i + 1) % 2000 == 0:
                el = time.perf_counter() - t1
                print(f"  {i + 1}/{len(jobs)} classes, {el:.0f}s elapsed, "
                      f"~{el / (i + 1) * (len(jobs) - i - 1):.0f}s left", flush=True)
    print(f"Monte Carlo done in {time.perf_counter() - t1:.0f}s "
          f"({args.trials} trials/class, {args.workers} workers)")

    equity = class_eq[class_of_hand].astype(np.float16)   # expand to every hand
    os.makedirs(os.path.dirname(EQUITY_PATH), exist_ok=True)
    np.save(EQUITY_PATH, equity)
    np.save(SORTED_PATH, np.sort(equity))
    print(f"Saved {EQUITY_PATH} ({os.path.getsize(EQUITY_PATH) / 1024:.0f} KB) and "
          f"{SORTED_PATH} ({os.path.getsize(SORTED_PATH) / 1024:.0f} KB)")
    print(f"Mean equity over all hands: {equity.astype(np.float64).mean():.4f} (should be ~0.5)")
    w = class_sizes / class_sizes.sum()
    print(f"Class sizes: min {class_sizes.min()}, max {class_sizes.max()}, "
          f"weighted mean equity {np.dot(w, class_eq):.4f}")


if __name__ == "__main__":
    main()
