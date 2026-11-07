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

from game import Action, AgentConfig, HandResult, Observation


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


STRATEGIES: dict[str, type[Strategy]] = {
    cls.name: cls
    for cls in (RandomStrategy, AlwaysFold, AlwaysCall, AlwaysHalfPot, AlwaysAllIn)
}


def make_strategy(name: str) -> Strategy:
    """Build a strategy by name (see STRATEGIES)."""
    try:
        return STRATEGIES[name]()
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}") from None
