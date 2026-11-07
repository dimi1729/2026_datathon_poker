"""
zoo.py - "realistic" opponents: the kinds of bots other participants plausibly write.

    eq_honest       Monte Carlo equity vs a random hand; bets strong hands, calls
                    when equity beats pot odds, never bluffs. (The most common design.)
    eq_balanced     Same, plus bluffs at roughly balanced frequencies.
    naive_adaptive  eq_balanced + fast opponent modelling: a 50-hand window, no
                    prior, adapts after 10 observations. Calls wider vs opponents it
                    has seen bluff at showdown, bluffs more vs opponents who fold.
                    (The type an "advertising" bluff phase is meant to fool.)
    switcher        Attacker: bluffs heavily for its first 150 hands (pot-sized river
                    bluffs that get shown down), then plays value-only. Built to
                    poison opponent models that never forget.

All per-half state lives in `memory`; registered in strategies.STRATEGIES.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np

import fasteval
from game import Action, Street
from strategies import PreflopThreshold

RAISES = (Action.RAISE_HALF, Action.RAISE_POT)


def mc_equity(hole, board, n, rng) -> float:
    """Equity vs ONE uniformly random opponent hand (the naive, common approach)."""
    hero = np.array(sorted(hole), dtype=np.int64)
    known = np.array(sorted(board), dtype=np.int64)
    dead = set(hole) | set(board)
    deck = np.array([c for c in range(36) if c not in dead], dtype=np.int64)
    m = 5 - len(known)
    pick = deck[np.argsort(rng.random((n, len(deck))), axis=1)[:, :m + 5]]
    boards = np.sort(np.concatenate([np.broadcast_to(known, (n, len(known))), pick[:, :m]], axis=1), axis=1)
    opp = np.sort(pick[:, m:], axis=1)
    tri = fasteval.triple_indices(boards)
    hs = fasteval.best(fasteval.pair_indices(hero)[None, :], tri)
    os_ = fasteval.best(fasteval.pair_indices(opp), tri)
    return float(((hs > os_) + 0.5 * (hs == os_)).mean())


def river_strength(hole, board, rng, n=150) -> float:
    """Share of random hands that `hole` beats on a complete board."""
    return mc_equity(hole, board, n, rng)


class EquityBot(PreflopThreshold):
    name = "eq_honest"

    def __init__(self, bet_eq=0.65, pot_eq=0.82, bluff=0.0, river_bluff=0.0, call_margin=0.0,
                 n=150, seed=11) -> None:
        super().__init__(raise_pcts=(0.65, 0.90, 0.96), fold_pct=0.25, margin=0.02)
        self.base = dict(bet_eq=bet_eq, pot_eq=pot_eq, bluff=bluff, river_bluff=river_bluff,
                         call_margin=call_margin)
        self.n, self.seed = n, seed

    def new_half(self, config, memory):
        memory["rng"] = random.Random(self.seed)
        memory["np"] = np.random.default_rng(self.seed)
        memory["p"] = dict(self.base)
        memory["hands"] = 0

    def params(self, memory):
        return memory["p"]

    def choose_action(self, obs, memory):
        if "p" not in memory:
            self.new_half(None, memory)
        if obs.street == Street.PREFLOP:
            return super().choose_action(obs, memory)
        p = self.params(memory)
        rng = memory["rng"]
        eq = mc_equity(obs.hole_cards, obs.board, self.n, memory["np"])
        legal = obs.legal_actions
        half = Action.RAISE_HALF if Action.RAISE_HALF in legal else None
        pot = Action.RAISE_POT if Action.RAISE_POT in legal else None
        river = obs.street == Street.RIVER
        if obs.to_call == 0:
            if eq >= p["pot_eq"] and pot:
                return pot
            if eq >= p["bet_eq"] and half:
                return half
            if eq < 0.45:
                if river and pot and rng.random() < p["river_bluff"]:
                    return pot
                if not river and half and rng.random() < p["bluff"]:
                    return half
            return Action.CHECK_CALL
        if eq >= p["pot_eq"] and pot:
            return pot
        pot_odds = obs.to_call / (obs.pot + obs.to_call)
        return Action.CHECK_CALL if eq >= pot_odds + p["call_margin"] else Action.FOLD

    def on_hand_end(self, result, memory):
        memory["hands"] = memory.get("hands", 0) + 1


class NaiveAdaptive(EquityBot):
    """Tracks the opponent over a short window with no prior and adapts fast."""

    name = "naive_adaptive"

    def __init__(self, window=50, min_obs=10) -> None:
        super().__init__(bluff=0.12, river_bluff=0.10)
        self.window, self.min_obs = window, min_obs

    def new_half(self, config, memory):
        super().new_half(config, memory)
        memory["bluffs"] = deque(maxlen=self.window)   # 1/0 per aggressive showdown of opp
        memory["folds"] = deque(maxlen=self.window)    # 1/0 per our postflop bet
        memory["cur"] = None

    def params(self, memory):
        p = dict(self.base)
        b, f = memory["bluffs"], memory["folds"]
        if len(b) >= 3:  # "they showed a few bluffs -> they bluff a lot"
            share = sum(b) / len(b)
            p["call_margin"] = max(-0.25, min(0.15, -0.6 * (share - 0.33)))
        if len(f) >= self.min_obs:
            rate = sum(f) / len(f)
            p["bluff"] = max(0.0, min(0.6, 0.12 + 1.0 * (rate - 0.35)))
            p["river_bluff"] = max(0.0, min(0.5, 0.10 + 1.0 * (rate - 0.45)))
        return p

    def choose_action(self, obs, memory):
        if "p" not in memory:
            self.new_half(None, memory)
        cur = memory["cur"]
        if cur is None or cur["idx"] != obs.hand_index:
            cur = memory["cur"] = {"idx": obs.hand_index, "seen": 0, "opp_aggr": False, "pending": False}
        for a in obs.action_history[cur["seen"]:]:
            if not a.is_me:
                if cur["pending"]:
                    memory["folds"].append(0)
                    cur["pending"] = False
                if a.street != Street.PREFLOP and a.action in RAISES:
                    cur["opp_aggr"] = True
        cur["seen"] = len(obs.action_history)
        action = super().choose_action(obs, memory)
        if obs.street != Street.PREFLOP and action in RAISES:
            cur["pending"] = True
        return action

    def on_hand_end(self, result, memory):
        super().on_hand_end(result, memory)
        cur = memory.get("cur")
        if cur is None or cur["idx"] != result.hand_index:
            return
        if cur["pending"]:
            memory["folds"].append(1 if (not result.went_to_showdown and result.my_net > 0) else 0)
        if result.went_to_showdown and cur["opp_aggr"] and result.opp_hole and len(result.board) == 5:
            weak = river_strength(result.opp_hole, result.board, memory["np"]) < 0.5
            memory["bluffs"].append(1 if weak else 0)
        memory["cur"] = None


class Switcher(EquityBot):
    """Bluff-heavy for `switch_at` hands, then value-only (never bluffs, calls tight)."""

    name = "switcher"

    def __init__(self, switch_at=150) -> None:
        super().__init__()
        self.switch_at = switch_at

    def params(self, memory):
        if memory.get("hands", 0) < self.switch_at:
            return dict(self.base, bluff=0.6, river_bluff=0.7, bet_eq=0.6, call_margin=0.0)
        return dict(self.base, bluff=0.0, river_bluff=0.0, bet_eq=0.62, pot_eq=0.75, call_margin=0.05)


def register(strategies: dict, named) -> None:
    strategies["eq_honest"] = EquityBot
    strategies["eq_balanced"] = named("eq_balanced", EquityBot, bluff=0.15, river_bluff=0.15)
    strategies["naive_adaptive"] = NaiveAdaptive
    strategies["switcher"] = Switcher
