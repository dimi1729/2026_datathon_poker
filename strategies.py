"""
strategies.py - a library of poker strategies with the same interface as agent.py.

Every strategy is a class with three methods that mirror agent.py's hooks:

    new_half(config, memory)        start of each half (memory is a fresh dict)
    choose_action(obs, memory)      return one of obs.legal_actions
    on_hand_end(result, memory)     after every hand (optional)

agent.py plays one of these (picked with the STRATEGY env var), and arena.py
plays them against each other in-process, without HTTP.

Add a new strategy by subclassing Strategy and listing it in STRATEGIES.
"""

from __future__ import annotations

import random

from game import Action, AgentConfig, HandResult, Observation, Street
from preflop import preflop_equity, preflop_percentile


class Strategy:
    """Base class. Defaults: no setup, check/fold, learn nothing."""

    name = "base"

    def new_half(self, config: AgentConfig, memory: dict) -> None:
        pass

    def choose_action(self, obs: Observation, memory: dict) -> Action:
        return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD

    def on_hand_end(self, result: HandResult, memory: dict) -> None:
        pass


class RandomStrategy(Strategy):
    """Uniformly random legal action. Seeded per half so runs are repeatable."""

    name = "random"

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed

    def new_half(self, config, memory):
        memory["rng"] = random.Random(self.seed)

    def choose_action(self, obs, memory):
        rng = memory.get("rng")
        if rng is None:  # defensive: never crash if new_half was skipped
            rng = memory["rng"] = random.Random(self.seed)
        return rng.choice(obs.legal_actions)


class AlwaysFold(Strategy):
    """Folds whenever it owes chips. FOLD is illegal when nothing is owed,
    so it checks then (this is the 'check/fold' bot)."""

    name = "always_fold"

    def choose_action(self, obs, memory):
        return Action.FOLD if Action.FOLD in obs.legal_actions else Action.CHECK_CALL


class AlwaysCall(Strategy):
    """Calling station: always checks or calls, never raises or folds."""

    name = "always_call"

    def choose_action(self, obs, memory):
        return Action.CHECK_CALL


class AlwaysHalfPot(Strategy):
    """Raises half pot whenever allowed, otherwise calls."""

    name = "always_half"

    def choose_action(self, obs, memory):
        return Action.RAISE_HALF if Action.RAISE_HALF in obs.legal_actions else Action.CHECK_CALL


class AlwaysAllIn(Strategy):
    """Shoves as hard as pot-limit allows: raises pot whenever it can, else
    calls. With 40bb stacks, repeated pot raises put everyone all-in within
    a street or two, so this is the closest legal thing to 'always all-in'."""

    name = "always_allin"

    def choose_action(self, obs, memory):
        return Action.RAISE_POT if Action.RAISE_POT in obs.legal_actions else Action.CHECK_CALL


class PreflopThreshold(Strategy):
    """Card-aware bot driven only by the offline preflop table (preflop.py).

    Preflop:
      * raise pot if the hand's percentile clears the bar for this raise
        number (open / 3-bet / 4-bet+ get stricter bars);
      * otherwise check if free;
      * otherwise call if equity vs a random hand beats the pot odds by
        `margin` (and the hand isn't in the bottom `fold_pct`), else fold.
    Postflop (no board reading yet): bet half pot with the top `value_pct`
    hands when checked to, otherwise check; facing a bet, use the same
    equity-vs-pot-odds rule with `post_margin`.
    """

    name = "preflop"

    def __init__(self, raise_pcts=(0.70, 0.88, 0.95), margin=0.03, fold_pct=0.15,
                 value_pct=0.90, post_margin=0.05) -> None:
        self.raise_pcts = raise_pcts
        self.margin = margin
        self.fold_pct = fold_pct
        self.value_pct = value_pct
        self.post_margin = post_margin

    def choose_action(self, obs, memory):
        eq = preflop_equity(obs.hole_cards)
        pct = preflop_percentile(obs.hole_cards)
        legal = obs.legal_actions
        pot_odds = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0

        if obs.street == Street.PREFLOP:
            bar = self.raise_pcts[min(obs.raises_this_street, len(self.raise_pcts) - 1)]
            if pct >= bar and Action.RAISE_POT in legal:
                return Action.RAISE_POT
            if obs.to_call == 0:
                return Action.CHECK_CALL
            if pct >= self.fold_pct and eq >= pot_odds + self.margin:
                return Action.CHECK_CALL
            return Action.FOLD

        if obs.to_call == 0:
            if pct >= self.value_pct and Action.RAISE_HALF in legal:
                return Action.RAISE_HALF
            return Action.CHECK_CALL
        if eq >= pot_odds + self.post_margin:
            return Action.CHECK_CALL
        return Action.FOLD


def _named(name: str, cls, **kwargs):
    """A factory for a parameter variant of a strategy, registered under `name`."""
    def factory() -> Strategy:
        s = cls(**kwargs)
        s.name = name
        return s
    return factory


# name -> zero-argument factory
STRATEGIES: dict = {
    cls.name: cls
    for cls in (RandomStrategy, AlwaysFold, AlwaysCall, AlwaysHalfPot, AlwaysAllIn, PreflopThreshold)
}
STRATEGIES["preflop_tight"] = _named("preflop_tight", PreflopThreshold,
                                     raise_pcts=(0.80, 0.92, 0.97), fold_pct=0.35, margin=0.06)
STRATEGIES["preflop_loose"] = _named("preflop_loose", PreflopThreshold,
                                     raise_pcts=(0.55, 0.80, 0.92), fold_pct=0.05, margin=0.0)


def make_strategy(name: str) -> Strategy:
    """Build a strategy by name (see STRATEGIES)."""
    try:
        return STRATEGIES[name]()
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}") from None
