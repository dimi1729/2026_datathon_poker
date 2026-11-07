"""
arena.py - play strategies from strategies.py against each other, in-process.

Uses the real game engine (game.py) and the tournament's duplicate format and
scoring: each match is 2 halves on the same cards with seats swapped, scored
as mean bb/deal with a 95% interval (win / lose / chop), exactly like
judge_engine.py, but without HTTP, so it is ~100x faster.

    python arena.py                                  # round robin of every strategy
    python arena.py random always_allin --deals 1000 # one pairing
    python arena.py --seeds 5                        # 5 matches per pairing (different cards)

It also times every choose_action() call per strategy, so you can see how much
of the 8 s per-half time bank a strategy would use.
"""

from __future__ import annotations

import argparse
import itertools
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from game import (BIG_BLIND, CHOP_Z, NUM_DEALS, TIME_BANK_SECONDS, AgentConfig, Game,
                  IllegalActionError, button_for_deal, default_action, make_deal)
from strategies import STRATEGIES, make_strategy


REPORTS: list[str] = []  # end-of-half summaries from strategies that have report()


@dataclass
class Timing:
    """Thinking time of one strategy, in seconds per call."""

    samples: list[float] = field(default_factory=list)
    errors: int = 0
    end_seconds: float = 0.0  # time spent in on_hand_end
    halves: int = 0

    def add(self, other: "Timing") -> None:
        self.samples += other.samples
        self.errors += other.errors
        self.end_seconds += other.end_seconds
        self.halves += other.halves


def play_half(names: tuple[str, str], seed: str, half: int, deals: int,
              timings: dict[str, Timing], threads: int = 1) -> list[int]:
    """Play one half. names[0] is 'agent1'. Returns agent1's net chips per deal."""
    a1_seat = 0 if half == 1 else 1
    seat_names = (names[0], names[1]) if a1_seat == 0 else (names[1], names[0])
    bots = [make_strategy(n) for n in seat_names]
    mems: list[dict] = [{}, {}]
    t = [Timing(), Timing()]
    for bot, mem in zip(bots, mems):
        bot.new_half(AgentConfig(), mem)

    def play_deal(d: int) -> int:
        game = Game(make_deal(seed, d), button_for_deal(d), hand_index=d)
        while not game.is_terminal:
            seat = game.to_act
            obs = game.observation(seat)
            start = time.perf_counter()
            try:
                action = bots[seat].choose_action(obs, mems[seat])
            except Exception as exc:
                action = None
                first_errors.append(f"{seat_names[seat]}: {type(exc).__name__}: {exc}")
            t[seat].samples.append(time.perf_counter() - start)
            try:
                game.apply(action)
            except (IllegalActionError, ValueError, TypeError):
                t[seat].errors += 1
                game.apply(default_action(obs.to_call))
        for seat in (0, 1):
            start = time.perf_counter()
            bots[seat].on_hand_end(game.hand_result(seat), mems[seat])
            t[seat].end_seconds += time.perf_counter() - start
        return game.result().nets[a1_seat]

    first_errors: list[str] = []
    if threads > 1:
        # Deals run concurrently: only safe for strategies that keep no state
        # across hands (e.g. network-bound ones like jev).
        with ThreadPoolExecutor(threads) as pool:
            nets = list(pool.map(play_deal, range(deals)))
    else:
        nets = [play_deal(d) for d in range(deals)]
    for e in first_errors[:3]:
        print("  error:", e)

    for seat in (0, 1):
        t[seat].halves = 1
        if hasattr(bots[seat], "report"):
            REPORTS.append(f"--- {seat_names[seat]} (seat {seat}) vs {seat_names[1 - seat]}, "
                           f"seed {seed}, half {half} ---\n" + bots[seat].report(mems[seat]))
        timings.setdefault(seat_names[seat], Timing()).add(t[seat])
    return nets


@dataclass
class MatchResult:
    agent1: str
    agent2: str
    mean_bb: float
    ci: tuple[float, float]
    verdict: str  # "W" (agent1 wins), "L", or "C" (chop)


def play_match(a1: str, a2: str, seed: str, deals: int,
               timings: dict[str, Timing], threads: int = 1) -> MatchResult:
    """Duplicate match: both halves on the same cards, seats swapped."""
    h1 = play_half((a1, a2), seed, 1, deals, timings, threads)
    h2 = play_half((a1, a2), seed, 2, deals, timings, threads)
    paired = [(x + y) / BIG_BLIND for x, y in zip(h1, h2)]
    mean = sum(paired) / deals
    se = statistics.stdev(paired) / math.sqrt(deals) if deals > 1 else 0.0
    lo, hi = mean - CHOP_Z * se, mean + CHOP_Z * se
    verdict = "W" if lo > 0 else "L" if hi < 0 else "C"
    return MatchResult(a1, a2, mean, (lo, hi), verdict)


def pct(sorted_samples: list[float], q: float) -> float:
    return sorted_samples[min(len(sorted_samples) - 1, int(q * len(sorted_samples)))]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("strategies", nargs="*", help=f"default: all of {sorted(STRATEGIES)}")
    p.add_argument("--deals", type=int, default=NUM_DEALS, help="deals per half (default 1000)")
    p.add_argument("--seeds", type=int, default=1, help="matches per pairing, each on new cards")
    p.add_argument("--threads", type=int, default=1,
                   help="play deals concurrently (only for stateless, network-bound strategies)")
    p.add_argument("--first-vs-all", action="store_true",
                   help="only play the first strategy against each of the others")
    p.add_argument("--reports", choices=("none", "last", "all"), default="last",
                   help="print end-of-half summaries of adaptive bots (default: the last one per pairing)")
    p.add_argument("--seed", default="arena", help="base seed (same seed = same cards)")
    args = p.parse_args()

    names = args.strategies or list(STRATEGIES)
    for n in names:
        make_strategy(n)  # fail fast on typos

    timings: dict[str, Timing] = {}
    points = {n: 0.0 for n in names}
    bb_sum = {n: 0.0 for n in names}
    games = {n: 0 for n in names}
    matrix: dict[tuple[str, str], list[MatchResult]] = {}
    start = time.perf_counter()

    pairings = ([(names[0], n) for n in names[1:]] if args.first_vs_all
                else list(itertools.combinations(names, 2)))
    for a1, a2 in pairings:
        for k in range(args.seeds):
            r = play_match(a1, a2, f"{args.seed}-{k}", args.deals, timings, args.threads)
            matrix.setdefault((a1, a2), []).append(r)
            pts = {"W": 1.0, "L": 0.0, "C": 0.5}[r.verdict]
            points[a1] += pts
            points[a2] += 1 - pts
            bb_sum[a1] += r.mean_bb
            bb_sum[a2] -= r.mean_bb
            games[a1] += 1
            games[a2] += 1
    elapsed = time.perf_counter() - start

    print(f"{args.deals} deals/half, {args.seeds} match(es) per pairing, "
          f"done in {elapsed:.1f}s\n")
    print("Matches (agent1 bb/deal, 95% interval, verdict for agent1):")
    for (a1, a2), rs in matrix.items():
        for r in rs:
            print(f"  {a1:>13} vs {a2:<13} {r.mean_bb:+8.3f}  "
                  f"[{r.ci[0]:+8.3f}, {r.ci[1]:+8.3f}]  {r.verdict}")

    print("\nStandings (points: win 1, chop 0.5; avg bb/deal over all matches):")
    for n in sorted(names, key=lambda n: (-points[n], -bb_sum[n])):
        print(f"  {n:<13} {points[n]:5.1f} pts  {bb_sum[n] / max(1, games[n]):+8.3f} bb/deal")

    if REPORTS and args.reports != "none":
        print("\nAdaptive-bot reports:")
        shown = REPORTS if args.reports == "all" else REPORTS[1::2]
        for r in shown:
            print(r)

    print("\nTiming per strategy (in-process, no HTTP; s/half includes on_hand_end):")
    print(f"  {'strategy':<13} {'calls':>8} {'mean us':>9} {'p99 us':>8} {'max us':>8} "
          f"{'s/half':>8} {'bank used':>9} {'errors':>6}")
    for n in names:
        t = timings.get(n)
        if not t or not t.samples:
            continue
        s = sorted(t.samples)
        per_half = (sum(s) + t.end_seconds) / max(1, t.halves)
        print(f"  {n:<13} {len(s):>8} {statistics.mean(s) * 1e6:>9.2f} {pct(s, .99) * 1e6:>8.2f} "
              f"{s[-1] * 1e6:>8.1f} {per_half:>8.4f} {per_half / TIME_BANK_SECONDS:>8.3%} {t.errors:>6}")


if __name__ == "__main__":
    main()
