"""
jev_strategy.py - a strategy that asks TypeSafe's Jev model to pick the action.

Every decision is one Choice question over the legal actions, with the game
situation as structured state. OFFLINE EXPERIMENT ONLY: the tournament has no
network access and an 8 s time bank per half, so this can never be submitted.

Needs JEV_API (or TYPESAFE_API_KEY) in the environment or in .env.
Every request and response is appended to logs/jev_calls.jsonl.
"""

from __future__ import annotations

import json
import os
import threading
import time

import httpx

from evaluator import cards_to_str, category_name, omaha_score
from game import Action, Street
from strategies import Strategy

API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-1.14.0"
ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(ROOT, "logs", "jev_calls.jsonl")

# ----- all question wording lives here --------------------------------------

RULES = ("Heads-up Short Deck 5-card Pot-Limit Omaha. 36-card deck (6 through A). "
         "Each player has 5 hole cards and must use exactly 2 of them plus exactly 3 "
         "of the 5 board cards. Short-deck ranking: a flush beats a full house, and "
         "three of a kind beats a straight. Both players started the hand with 80 chips "
         "(big blind 2). Chips won are what counts, over many hands.")

INSTRUCTIONS = "Which action wins the most chips in the long run for me in this spot?"

STREET_NAMES = {Street.PREFLOP: "preflop", Street.FLOP: "flop", Street.TURN: "turn", Street.RIVER: "river"}
ACTION_WORDS = {Action.FOLD: "folded", Action.CHECK_CALL: "checked/called",
                Action.RAISE_HALF: "raised half pot", Action.RAISE_POT: "raised pot"}


def build_state(obs) -> dict:
    """The situation as a person would read it: named cards, named actions."""
    history = [
        f"{STREET_NAMES[a.street]}: {'I' if a.is_me else 'opponent'} "
        f"{ACTION_WORDS[a.action]} (street bet now {a.street_total} chips)"
        for a in obs.action_history
    ]
    state = {
        "rules": RULES,
        "street": STREET_NAMES[obs.street],
        "my_hole_cards": cards_to_str(obs.hole_cards),
        "board": cards_to_str(obs.board) if obs.board else "no board cards yet",
        "my_position": ("button: small blind, acts first preflop and last after the flop"
                        if obs.is_button else
                        "big blind: acts last preflop and first after the flop"),
        "pot_chips": obs.pot,
        "chips_i_must_call": obs.to_call,
        "my_stack_chips": obs.my_stack,
        "opponent_stack_chips": obs.opp_stack,
        "actions_this_hand": history or ["none yet (only the blinds are posted)"],
    }
    if len(obs.board) == 5:  # a fact code already knows
        state["my_best_hand"] = category_name(omaha_score(obs.hole_cards, obs.board))
    return state


def build_criteria(obs) -> dict[str, str]:
    """One option per legal action, described with the exact chip amounts."""
    out = {}
    for a in obs.legal_actions:
        if a == Action.FOLD:
            out["fold"] = f"Give up the hand and the {obs.pot} chips in the pot."
        elif a == Action.CHECK_CALL and obs.to_call == 0:
            out["check"] = "Put in no chips and see what the opponent does."
        elif a == Action.CHECK_CALL:
            out["call"] = f"Put in {obs.to_call} chips to match the opponent's bet."
        elif a == Action.RAISE_HALF:
            out["raise_half_pot"] = (f"Raise to {obs.raise_to[a]} chips on this street "
                                     f"(about half the pot).")
        elif a == Action.RAISE_POT:
            total = obs.raise_to[a]
            allin = " This puts all my chips in." if total - obs.my_bet >= obs.my_stack else ""
            out["raise_pot"] = f"Raise to {total} chips on this street (the pot-limit maximum).{allin}"
    return out


OPTION_TO_ACTION = {"fold": Action.FOLD, "check": Action.CHECK_CALL, "call": Action.CHECK_CALL,
                    "raise_half_pot": Action.RAISE_HALF, "raise_pot": Action.RAISE_POT}

# ----- plumbing --------------------------------------------------------------


def _api_key() -> str:
    key = os.environ.get("JEV_API") or os.environ.get("TYPESAFE_API_KEY")
    if not key:
        try:
            with open(os.path.join(ROOT, ".env")) as f:
                for line in f:
                    name, _, value = line.strip().partition("=")
                    if name in ("JEV_API", "TYPESAFE_API_KEY") and value:
                        key = value.strip().strip('"').strip("'")
        except FileNotFoundError:
            pass
    if not key:
        raise RuntimeError("no JEV_API / TYPESAFE_API_KEY in the environment or .env")
    return key


class JevStrategy(Strategy):
    """Asks Jev to choose among the legal actions. A failed call RAISES (it is
    never turned into a silent fold), so the arena counts it as an error."""

    name = "jev"
    _lock = threading.Lock()
    stats = {"calls": 0, "input_tokens": 0, "failures": 0, "seconds": 0.0}

    def __init__(self) -> None:
        self.client = httpx.Client(timeout=30.0, headers={"Authorization": f"Bearer {_api_key()}"})
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    def ask(self, obs) -> tuple[str, dict]:
        payload = {"model": MODEL, "state": build_state(obs),
                   "questions": {"action": {"type": "choice", "instructions": INSTRUCTIONS,
                                            "criteria": build_criteria(obs)}}}
        start = time.perf_counter()
        body, error = None, None
        for attempt in range(4):
            try:
                r = self.client.post(API_URL, json=payload)
                if r.status_code == 429 or r.status_code >= 500:
                    error = f"HTTP {r.status_code}: {r.text[:200]}"
                    time.sleep(float(r.headers.get("retry-after", 2 ** attempt)))
                    continue
                r.raise_for_status()
                body, error = r.json(), None
                break
            except httpx.HTTPError as exc:
                error = f"{type(exc).__name__}: {exc}"
                time.sleep(2 ** attempt)
        elapsed = time.perf_counter() - start
        with self._lock:
            self.stats["calls"] += 1
            self.stats["seconds"] += elapsed
            if body:
                self.stats["input_tokens"] += body.get("usage", {}).get("input_tokens", 0)
            else:
                self.stats["failures"] += 1
            with open(LOG_PATH, "a") as f:
                f.write(json.dumps({"request": payload, "response": body, "error": error,
                                    "seconds": round(elapsed, 3)}) + "\n")
        if body is None:
            raise RuntimeError(f"Jev call failed: {error}")
        answer = body["answers"]["action"]
        return answer["choice"], answer["probabilities"]

    def choose_action(self, obs, memory):
        choice, _ = self.ask(obs)
        return OPTION_TO_ACTION[choice]
