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


class Maniac(Strategy):
    """Zoo bot: hyper-aggressive, never folds. Raises pot 50%, half pot 25%, calls 25%."""

    name = "maniac"

    def new_half(self, config, memory):
        memory["rng"] = random.Random(7)

    def choose_action(self, obs, memory):
        rng = memory.setdefault("rng", random.Random(7))
        x = rng.random()
        if x < 0.5 and Action.RAISE_POT in obs.legal_actions:
            return Action.RAISE_POT
        if x < 0.75 and Action.RAISE_HALF in obs.legal_actions:
            return Action.RAISE_HALF
        return Action.CHECK_CALL


class OverFolder(PreflopThreshold):
    """Zoo bot: plays preflop like `preflop`, but postflop folds to any bet
    unless its starting hand was top 10%. Never bets postflop without one."""

    name = "overfolder"

    def choose_action(self, obs, memory):
        if obs.street == Street.PREFLOP:
            return super().choose_action(obs, memory)
        strong = preflop_percentile(obs.hole_cards) >= 0.90
        if obs.to_call == 0:
            return Action.RAISE_HALF if strong and Action.RAISE_HALF in obs.legal_actions else Action.CHECK_CALL
        return Action.CHECK_CALL if strong else Action.FOLD


class AgentStrategy(Strategy):
    """The submission bot (agent.py) as a library strategy, for arena.py."""

    name = "agent"

    def __init__(self, exploit: bool = True, advert=None, robust=None, **flags) -> None:
        import agent  # lazy: agent.py builds its tables at import
        self.mod = agent
        self.exploit = exploit
        self.advert, self.robust = advert, robust  # None = use agent.CONFIG
        self.flags = flags                         # other per-half memory overrides

    def new_half(self, config, memory):
        self.mod.new_half(config, memory)
        if not self.exploit:
            memory["no_exploit"] = True  # opponent model still tracked, never used
        if self.advert is not None:
            memory["advert"] = self.advert
        if self.robust is not None:
            memory["robust"] = self.robust
        memory.update(self.flags)

    def choose_action(self, obs, memory):
        return self.mod.choose_action(obs, memory)

    def on_hand_end(self, result, memory):
        self.mod.on_hand_end(result, memory)

    def report(self, memory) -> str:
        return self.mod.report(memory)


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
    for cls in (RandomStrategy, AlwaysFold, AlwaysCall, AlwaysHalfPot, AlwaysAllIn, PreflopThreshold,
                Maniac, OverFolder, AgentStrategy)
}
STRATEGIES["preflop_tight"] = _named("preflop_tight", PreflopThreshold,
                                     raise_pcts=(0.80, 0.92, 0.97), fold_pct=0.35, margin=0.06)


def _jev() -> Strategy:
    from jev_strategy import JevStrategy  # lazy: needs network + httpx, never used in the tournament
    return JevStrategy()


STRATEGIES["jev"] = _jev
STRATEGIES["agent_noexploit"] = _named("agent_noexploit", AgentStrategy, exploit=False)
STRATEGIES["agent_advert"] = _named("agent_advert", AgentStrategy, advert=True)  # until bluffs are shown
STRATEGIES["agent_advert_fixed"] = _named("agent_advert_fixed", AgentStrategy, advert=True,
                                          advert_until_shown=False)          # fixed 60 hands
STRATEGIES["agent_noadvert"] = _named("agent_noadvert", AgentStrategy, advert=False)
STRATEGIES["agent_bluff2"] = _named("agent_bluff2", AgentStrategy, intermittent_bluff=0.02)
STRATEGIES["agent_combo"] = _named("agent_combo", AgentStrategy, advert=True, intermittent_bluff=0.02)
STRATEGIES["agent_naive"] = _named("agent_naive", AgentStrategy, robust=False)      # no recency/robust model
STRATEGIES["agent_robust"] = _named("agent_robust", AgentStrategy, robust=True)


def _zoo(name: str):
    def factory() -> Strategy:
        import zoo  # lazy: builds fasteval's table
        reg: dict = {}
        zoo.register(reg, _named)
        return reg[name]()
    return factory


for _n in ("eq_honest", "eq_balanced", "naive_adaptive", "switcher"):
    STRATEGIES[_n] = _zoo(_n)
STRATEGIES["preflop_loose"] = _named("preflop_loose", PreflopThreshold,
                                     raise_pcts=(0.55, 0.80, 0.92), fold_pct=0.05, margin=0.0)


def make_strategy(name: str) -> Strategy:
    """Build a strategy by name (see STRATEGIES)."""
    try:
        return STRATEGIES[name]()
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}") from None
