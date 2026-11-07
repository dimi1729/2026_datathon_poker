"""
agent.py - adaptive heads-up short-deck PLO5 bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port
    STRATEGY=always_allin PORT=5009 python agent.py   # play a strategies.py bot instead

Strategy (all tunable numbers are in CONFIG below):
  * Preflop: percentile thresholds from the offline table (data/preflop_equity.npy).
  * Postflop: one vectorized Monte Carlo pass (fasteval.py) vs an opponent range
    narrowed by their actions -> equity + "nuttiness" -> hand class
    (NUTTED / GOOD / NUT_DRAW / WEAK / AIR) -> action, with board-texture,
    position and river-polarization adjustments.
  * Opponent model in `memory` (fold-to-bet, river bluff share, aggression,
    VPIP, PFR), Bayesian-smoothed; bounded exploit adjustments after
    CONFIG["exploit_start"] hands; stop-loss reverts to defaults if the
    luck-adjusted exploit results go bad.

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament
"""

import math
import os
import random
import time
import traceback

import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, warmup  # noqa: F401
from game import Action, AgentConfig, HandResult, Observation, Street

# Slow setup happens here, at startup (limit 15 s): the evaluator table, the
# preflop tables, and fasteval's pair x triple table (~1 s).
warmup()
import fasteval  # noqa: E402
import preflop  # noqa: E402
from preflop import preflop_equity, preflop_percentile  # noqa: E402

# rename as your team/bot's name
PARTICIPANT = "TotallyAwesomeDatathonParticipant"  # TODO: your team name
AGENT_NAME = os.environ.get("STRATEGY", "AdaptiveBot")


# ===== YOUR CODE HERE =====

CONFIG = {
    # ---- time-bank guard (seconds of OUR compute per half; the judge also
    #      charges ~0.3 ms of HTTP overhead per call on top of this) ----
    "time_soft_s": 3.0,          # above this: smaller Monte Carlo
    "time_hard_s": 5.0,          # above this: tiny Monte Carlo
    "mc_boards": 30, "mc_opps": 12, "mc_river_opps": 300,          # normal
    "mc_soft": (12, 8, 100),     # (boards, opps, river opps) after time_soft_s
    "mc_hard": (5, 5, 30),       # after time_hard_s
    "seed": 20261107,

    # ---- preflop (percentile of the hand among all 376,992 starting hands) ----
    "pf_open_raise_pct": 0.65,   # button unopened: RAISE_POT top 35%
    "pf_open_limp_pct": 0.35,    # ... next 30% limp (or RAISE_HALF, see below), rest fold
    "pf_open_mid_raise_half": False,
    "pf_iso_raise_pct": 0.85,    # big blind vs a limp: RAISE_POT top 15%, else check
    "pf_3bet_pct": 0.93,         # facing one raise: RAISE_POT (3-bet) top 7%
    "pf_defend_pct": 0.35,       # ... call with the top 65% if equity beats pot odds
    "pf_defend_margin": 0.0,
    "pf_4bet_pct": 0.96,         # facing a 3-bet+: RAISE_POT top 4% (effectively all-in)
    "pf_vs3bet_call_pct": 0.90,  # ... call with the top 10%, fold the rest

    # ---- postflop classification ----
    "nut_top_frac": 0.10,        # "nut" = top 10% of hands on the final board
    "nutted_river_pct": 0.95,    # river: beats >= 95% of random hands
    "nutted_nuttiness": 0.60,    # before river: P(ending in the top 10%) >= 0.60
    "nutted_min_equity": 0.65,   # ... and equity vs their range >= this
    "good_equity": 0.60,
    "draw_nuttiness": 0.30,      # NUT_DRAW: below good_equity but P(nuts) >= this
    "strong_draw_nuttiness": 0.45,
    "weak_equity": 0.40,         # WEAK: equity >= this; below is AIR

    # ---- postflop betting (probabilities) ----
    "good_bet_freq": 0.50,       # GOOD hands bet half pot this often (else pot control)
    "draw_semibluff_ip": 0.60,   # NUT_DRAW semi-bluff half pot, in position
    "draw_semibluff_oop": 0.35,  # ... out of position
    "draw_raise_freq": 0.25,     # strongest draws raise a bet this often
    "air_bluff_freq": 0.12,      # AIR bets half pot (flop/turn)
    "dry_cbet_freq": 0.60,       # button + preflop raiser on a dry high board: c-bet WEAK/AIR
    "river_bluff_freq": 0.12,    # AIR pot-sized river bluff (~1 bluff per 2 value bets)
    "wet_bet_scale": 0.50,       # preflop raiser on a wet board: bet non-nut hands half as often
    "trap_freq": 0.0,            # NUTTED out of position: check (to check-raise) this often

    # ---- postflop calling ----
    "call_margin": 0.0,          # call if equity >= pot odds + margin
    "river_call_margin": 0.0,
    "draw_implied_bonus": 0.08,  # equity credit for nut draws (implied odds)
    "stackoff_spr": 1.5,         # GOOD hands: above this SPR, don't stack off ...
    "stackoff_equity": 0.60,     # ... unless equity >= this
    "stackoff_commit": 0.5,      # "stacking off" = calling >= this share of our stack

    # ---- range narrowing for the Monte Carlo ----
    "range_floor_weight": 0.15,  # weight of hands outside the assumed range
    "range_pfr_clamp": (0.15, 0.90),
    "range_vpip_clamp": (0.30, 1.00),
    "range_bet_default": 0.55,   # opp bet this street: top 55% of made hands (scaled by aggression)
    "range_bet_clamp": (0.30, 0.90),

    # ---- opponent model: Bayesian prior (prior_n pseudo-observations at the benchmark) ----
    "prior_n": 10,
    "bench_fold_to_half": 0.33,  # GTO: defend >= 67% vs half pot
    "bench_fold_to_pot": 0.50,   # GTO: defend >= 50% vs pot
    "bench_river_bluff": 0.33,   # balanced pot-size river bluff share
    "bench_aggression": 0.35,
    "bench_vpip": 0.65,
    "bench_pfr": 0.45,
    "bluff_weak_pct": 0.50,      # opp river hand "weak" = beats < 50% of random hands

    # ---- exploits: start after N hands; param = clamp(default + gain * (rate - bench)) ----
    "exploit_start": 75,
    "exploit_log_delta": 0.03,   # log a parameter change when it moved this much
    "exploits": {
        # risky = +1 if raising the param makes us more exploitable (bluff more,
        # value-bet thinner, trap), -1 if lowering it does (call wider, open wider).
        # param:             (driver,           gain,  lo,    hi,   risky)
        "air_bluff_freq":     ("fold_to_half",   1.00,  0.00,  0.45,  +1),
        "dry_cbet_freq":      ("fold_to_half",   0.80,  0.30,  0.90,  +1),
        "good_bet_freq":      ("fold_to_half",  -1.00,  0.30,  0.85,  +1),
        "river_bluff_freq":   ("fold_to_pot",    0.80,  0.00,  0.40,  +1),
        "river_call_margin":  ("river_bluff",   -0.30, -0.08,  0.10,  -1),
        "call_margin":        ("aggression",    -0.15, -0.05,  0.08,  -1),
        "trap_freq":          ("aggression",     1.00,  0.00,  0.40,  +1),
        "pf_3bet_pct":        ("vpip",          -0.10,  0.88,  0.96,  -1),
        "pf_open_raise_pct":  ("vpip",           0.30,  0.50,  0.75,  -1),
        "pf_defend_pct":      ("vpip",          -0.30,  0.25,  0.50,  -1),
    },

    # ---- robustness to opponents who change gears (e.g. bluff early to build
    #      an image, then value-bet). Every stat is also kept with exponential
    #      decay; each exploit uses the LESS risky of the long-run and recent
    #      estimates: trust slowly, distrust quickly. Ranges assume the tighter one.
    "robust": True,
    "recent_half_life": 60,      # hands
    "regime_shift_delta": 0.20,  # log when recent and long-run rates differ this much
    "regime_min_obs": 8,

    # ---- advertising: bluff a lot early so adaptive opponents call wider,
    #      then cut bluffs and value-bet more while their read lags behind.
    "advert": False,
    "advert_until_shown": True,  # True: bluff phase lasts until opponents have SEEN bluffs
    "advert_shown_target": 2,    # ... this many of our bluffs shown down (they only see
    "advert_max_hands": 150,     #     our cards at showdown), capped at this many hands
    "advert_hands": 60,          # fixed-length phase when advert_until_shown is False
    "advert_air_bluff": 0.55,    # AIR half-pot bluff frequency during the phase
    "advert_river_bluff": 0.50,  # AIR pot-sized river bluff frequency
    "advert_semibluff": 0.85,
    "post_advert_hands": 250,    # payoff phase length
    "post_advert_bluff_scale": 0.25,
    "post_advert_value_boost": 0.25,  # added to good_bet_freq (thinner value bets)

    # ---- intermittent bluffing: with a bad hand (AIR postflop, a would-be fold
    #      preflop) that we'd check or fold, raise pot instead this often ----
    "intermittent_bluff": 0.0,

    # ---- self-evaluation / stop-loss ----
    "stoploss_window": 150,      # exploit hands before judging, and revert length
    "stoploss_se": 1.0,          # revert if exploit mean < -stoploss_se * standard error
}

# Override for local testing: STRATEGY=<name> plays a strategies.py bot instead.
_OVERRIDE = None
if os.environ.get("STRATEGY"):
    from strategies import make_strategy
    _OVERRIDE = make_strategy(os.environ["STRATEGY"])

RAISES = (Action.RAISE_HALF, Action.RAISE_POT)
NUTTED, GOOD, NUT_DRAW, WEAK, AIR = "NUTTED", "GOOD", "NUT_DRAW", "WEAK", "AIR"
_DEBUG = bool(os.environ.get("AGENT_DEBUG"))  # re-raise internal errors (tests)

# Percentile of every starting hand, indexed by its colex rank (for range weights).
_PCT_ALL = (np.searchsorted(preflop.SORTED_EQUITY, np.asarray(preflop.EQUITY, dtype=np.float32),
                            side="left") / preflop.NUM_HANDS).astype(np.float32)
_BINOM5 = preflop._BINOM_NP


def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


# ----- opponent model ----------------------------------------------------------

STATS = ("fold_to_half", "fold_to_pot", "river_bluff", "aggression", "vpip", "pfr")


def _rate(memory, key, recent=False):
    """Bayesian-smoothed opponent rate for one stat (long-run or recency-weighted)."""
    k, n = memory["opp_recent" if recent else "opp"][key]
    bench = CONFIG["bench_" + key]
    return (k + CONFIG["prior_n"] * bench) / (n + CONFIG["prior_n"])


def _rates(memory, recent=False):
    return {key: _rate(memory, key, recent) for key in STATS}


def _robust(memory):
    return memory.get("robust", CONFIG["robust"])


def _add(memory, key, hit, n=1):
    """Record n observations (hit of them positive) in both the long and recent stats."""
    for table in (memory["opp"], memory["opp_recent"]):
        table[key][0] += hit
        table[key][1] += n


def _decay(memory):
    d = 0.5 ** (1.0 / CONFIG["recent_half_life"])
    for kn in memory["opp_recent"].values():
        kn[0] *= d
        kn[1] *= d


def _exploit_active(memory):
    if memory.get("no_exploit"):
        return False
    return memory["hands"] >= CONFIG["exploit_start"] and memory["sl"]["mode"] == "exploit"


def _update_params(memory):
    """Recompute the parameters in use from the opponent model (bounded)."""
    P = memory["P"]
    active = _exploit_active(memory)
    rates = _rates(memory)
    recent = _rates(memory, recent=True)
    robust = _robust(memory)
    if robust:
        _regime_log(memory, rates, recent)
    for name, (driver, gain, lo, hi, risky) in CONFIG["exploits"].items():
        default = CONFIG[name]
        bench = CONFIG["bench_" + driver]
        new = default
        if active:
            new = _clamp(default + gain * (rates[driver] - bench), lo, hi)
            if robust:  # the less risky of long-run and recent: trust slowly, distrust quickly
                alt = _clamp(default + gain * (recent[driver] - bench), lo, hi)
                if risky * (alt - default) < risky * (new - default):
                    new = alt
        last = memory["logged"].get(name, default)
        if abs(new - last) >= CONFIG["exploit_log_delta"] or (not active and new != last):
            memory["log"].append(
                f"hand {memory['hands']}: {name} {last:.3f} -> {new:.3f} "
                f"({driver}={rates[driver]:.2f} vs bench {CONFIG['bench_' + driver]:.2f}"
                f"{'' if active else ', defaults restored'})")
            memory["logged"][name] = new
            if len(memory["log"]) > 400:
                del memory["log"][:100]
        P[name] = new


def _regime_log(memory, rates, recent):
    last = memory.setdefault("regime_logged", {})
    for key in STATS:
        n_recent = memory["opp_recent"][key][1]
        if (n_recent >= CONFIG["regime_min_obs"]
                and abs(recent[key] - rates[key]) >= CONFIG["regime_shift_delta"]
                and memory["hands"] - last.get(key, -10**9) >= 100):
            last[key] = memory["hands"]
            memory["log"].append(f"hand {memory['hands']}: regime shift in {key}: "
                                 f"long-run {rates[key]:.2f}, recent {recent[key]:.2f}")


# ----- per-hand tracking ---------------------------------------------------------

def _hand(memory, hand_index):
    h = memory.get("hand")
    if h is None or h["idx"] != hand_index:
        h = memory["hand"] = {"idx": hand_index, "seen": 0, "pending": None, "vpip": False,
                              "pfr": False, "opp_bet_street": -1, "opp_river_pot": False,
                              "allin_street": None, "our_pf_raise_last": False,
                              "mode": "exploit" if _exploit_active(memory) else "default"}
    return h


def _observe(obs, memory, h):
    """Process the actions we haven't seen yet in this hand's history."""
    for a in obs.action_history[h["seen"]:]:
        if a.is_me:
            if a.street == Street.PREFLOP:
                h["our_pf_raise_last"] = a.action in RAISES
            continue
        if h["pending"] is not None:  # opponent acted after our postflop bet: didn't fold
            _add(memory, h["pending"], 0)
            h["pending"] = None
        if a.street == Street.PREFLOP:
            if a.action in RAISES:
                h["pfr"] = h["vpip"] = True
            elif a.action == Action.CHECK_CALL and a.amount > 0:
                h["vpip"] = True
        else:
            _add(memory, "aggression", 1 if a.action in RAISES else 0)
            if a.action in RAISES:
                h["opp_bet_street"] = int(a.street)
                if a.street == Street.RIVER and a.action == Action.RAISE_POT:
                    h["opp_river_pot"] = True
    h["seen"] = len(obs.action_history)


# ----- Monte Carlo features ------------------------------------------------------

def _sizes(memory):
    t = memory["think"]
    if t > CONFIG["time_hard_s"]:
        return CONFIG["mc_hard"]
    if t > CONFIG["time_soft_s"]:
        return CONFIG["mc_soft"]
    return CONFIG["mc_boards"], CONFIG["mc_opps"], CONFIG["mc_river_opps"]


def _features(obs, memory, h):
    """One MC pass -> (equity vs narrowed range, nuttiness)."""
    rng = memory["nprng"]
    hero = np.array(sorted(obs.hole_cards), dtype=np.int64)
    known = np.array(sorted(obs.board), dtype=np.int64)
    dead = set(obs.hole_cards) | set(obs.board)
    deck = np.array([c for c in range(36) if c not in dead], dtype=np.int64)
    R, m = len(deck), 5 - len(known)
    nb, no, nr = _sizes(memory)
    B, H = (1, nr) if m == 0 else (nb, no)

    okeys = rng.random((B, H, R))
    if m:
        comp = np.argpartition(rng.random((B, R)), m, axis=1)[:, :m]               # (B, m)
        okeys[np.arange(B)[:, None, None], np.arange(H)[None, :, None], comp[:, None, :]] = 2.0
        boards = np.sort(np.concatenate([np.broadcast_to(known, (B, len(known))), deck[comp]],
                                        axis=1), axis=1)
    else:
        boards = known[None, :]
    opp = np.sort(deck[np.argpartition(okeys, 5, axis=2)[..., :5]], axis=2)       # (B, H, 5)

    tri = fasteval.triple_indices(boards)                                          # (B, 10)
    hero_s = fasteval.best(fasteval.pair_indices(hero)[None, :], tri)              # (B,)
    opp_pairs = fasteval.pair_indices(opp)                                         # (B, H, 10)
    opp_s = fasteval.best(opp_pairs, tri[:, None, :])                              # (B, H)
    win = (hero_s[:, None] > opp_s) + 0.5 * (hero_s[:, None] == opp_s)

    # Range narrowing: preflop action, then betting on this street.
    w = np.ones((B, H))
    floor = CONFIG["range_floor_weight"]
    rates = _rates(memory)
    if _robust(memory):
        recent = _rates(memory, recent=True)
        rates = {k: min(rates[k], recent[k]) for k in rates}
    if h["pfr"] or h["vpip"]:
        frac = (_clamp(rates["pfr"], *CONFIG["range_pfr_clamp"]) if h["pfr"]
                else _clamp(rates["vpip"], *CONFIG["range_vpip_clamp"]))
        rank = sum(_BINOM5[k][opp[..., k]] for k in range(5))
        w = np.where(_PCT_ALL[rank] >= 1.0 - frac, 1.0, floor)
    if h["opp_bet_street"] == int(obs.street):
        frac = _clamp(CONFIG["range_bet_default"] * rates["aggression"] / CONFIG["bench_aggression"],
                      *CONFIG["range_bet_clamp"])
        cur = opp_s if m == 0 else fasteval.best(opp_pairs, fasteval.triple_indices(known))
        pct = cur.ravel().argsort().argsort().reshape(cur.shape) / cur.size
        w = w * np.where(pct >= 1.0 - frac, 1.0, floor)
    equity = float((w * win).sum() / w.sum())

    if m == 0:
        nuttiness = float(win.mean())
    else:
        nuttiness = float((win.mean(axis=1) >= 1.0 - CONFIG["nut_top_frac"] - 1e-9).mean())
    return equity, nuttiness


def _texture(board):
    ranks = [c >> 2 for c in board]
    suits = [c & 3 for c in board]
    monotone = max(suits.count(s) for s in range(4)) >= 3
    rs = set(ranks) | ({-1} if 8 in ranks else set())  # ace also plays low (A-6-7-8-9)
    straighty = any(len(rs & set(range(lo, lo + 5))) >= 3 for lo in range(-1, 5))
    paired = len(set(ranks)) < len(ranks)
    wet = monotone or straighty
    dry_high = not wet and not paired and max(ranks) >= 6  # queen or higher
    return wet, dry_high


def _classify(street, equity, nut):
    if street == Street.RIVER:
        if nut >= CONFIG["nutted_river_pct"] and equity >= CONFIG["nutted_min_equity"]:
            return NUTTED
    elif nut >= CONFIG["nutted_nuttiness"] and equity >= CONFIG["nutted_min_equity"]:
        return NUTTED
    if equity >= CONFIG["good_equity"]:
        return GOOD
    if street != Street.RIVER and nut >= CONFIG["draw_nuttiness"]:
        return NUT_DRAW
    if equity >= CONFIG["weak_equity"]:
        return WEAK
    return AIR


# ----- decisions -----------------------------------------------------------------

def _preflop(obs, memory):
    P = memory["P"]
    pct = preflop_percentile(obs.hole_cards)
    eq = preflop_equity(obs.hole_cards)
    legal = obs.legal_actions
    r = obs.raises_this_street
    pot_odds = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
    can_pot = Action.RAISE_POT in legal

    if r == 0 and obs.to_call > 0:                     # button, unopened
        if pct >= P["pf_open_raise_pct"] and can_pot:
            return Action.RAISE_POT
        if pct >= CONFIG["pf_open_limp_pct"]:
            if CONFIG["pf_open_mid_raise_half"] and Action.RAISE_HALF in legal:
                return Action.RAISE_HALF
            return Action.CHECK_CALL
        return Action.FOLD
    if r == 0:                                         # big blind vs a limp
        return Action.RAISE_POT if pct >= CONFIG["pf_iso_raise_pct"] and can_pot else Action.CHECK_CALL
    if obs.to_call == 0:
        return Action.CHECK_CALL
    if r == 1:                                         # facing one raise
        if pct >= P["pf_3bet_pct"] and can_pot:
            return Action.RAISE_POT
        if pct >= P["pf_defend_pct"] and eq >= pot_odds + CONFIG["pf_defend_margin"]:
            return Action.CHECK_CALL
        return Action.FOLD
    if pct >= CONFIG["pf_4bet_pct"] and can_pot:       # facing a 3-bet or more
        return Action.RAISE_POT
    return Action.CHECK_CALL if pct >= CONFIG["pf_vs3bet_call_pct"] else Action.FOLD


def _advert_phase(memory):
    """'advert' -> 'payoff' -> 'done'. The advert phase ends once enough bluffs
    have been SHOWN DOWN (or at a fixed/maximum hand count)."""
    a = memory["adv"]
    n = memory["hands"]
    until_shown = memory.get("advert_until_shown", CONFIG["advert_until_shown"])
    if a["phase"] == "advert":
        done = (a["shown"] >= CONFIG["advert_shown_target"] or n >= CONFIG["advert_max_hands"]
                if until_shown else n >= CONFIG["advert_hands"])
        if done:
            a.update(phase="payoff", start=n)
            memory["log"].append(f"hand {n}: advert phase over ({a['shown']} bluffs shown down, "
                                 f"{a['bluff_bets']} bluff bets made); payoff phase starts")
    if a["phase"] == "payoff" and n >= a["start"] + CONFIG["post_advert_hands"]:
        a["phase"] = "done"
        memory["log"].append(f"hand {n}: payoff phase over")
    return a["phase"]


def _phase_params(memory):
    """Parameters for this hand, with the advertising / payoff phases applied."""
    P = memory["P"]
    if not memory.get("advert", CONFIG["advert"]):
        return P
    phase = _advert_phase(memory)
    if phase == "advert":
        return dict(P, air_bluff_freq=max(P["air_bluff_freq"], CONFIG["advert_air_bluff"]),
                    river_bluff_freq=max(P["river_bluff_freq"], CONFIG["advert_river_bluff"]),
                    draw_semibluff_ip=CONFIG["advert_semibluff"],
                    draw_semibluff_oop=CONFIG["advert_semibluff"],
                    dry_cbet_freq=max(P["dry_cbet_freq"], CONFIG["advert_air_bluff"]))
    if phase == "payoff":
        sc = CONFIG["post_advert_bluff_scale"]
        return dict(P, air_bluff_freq=P["air_bluff_freq"] * sc,
                    river_bluff_freq=P["river_bluff_freq"] * sc,
                    dry_cbet_freq=P["dry_cbet_freq"] * sc,
                    good_bet_freq=min(0.95, P["good_bet_freq"] + CONFIG["post_advert_value_boost"]))
    return P


def _postflop(obs, memory, h):
    P, rng = _phase_params(memory), memory["rng"]
    equity, nut = _features(obs, memory, h)
    cls = _classify(obs.street, equity, nut)
    memory["last_cls"] = cls
    memory["class_counts"][cls] = memory["class_counts"].get(cls, 0) + 1
    legal = obs.legal_actions
    half = Action.RAISE_HALF if Action.RAISE_HALF in legal else None
    pot = Action.RAISE_POT if Action.RAISE_POT in legal else None
    river = obs.street == Street.RIVER
    ip = obs.is_button
    pf_raises = [a for a in obs.action_history if a.street == Street.PREFLOP and a.action in RAISES]
    we_raised_pf = bool(pf_raises) and pf_raises[-1].is_me
    wet, dry_high = _texture(obs.board)
    scale = P["wet_bet_scale"] if (wet and we_raised_pf and cls != NUTTED) else 1.0

    if obs.to_call == 0:
        if cls == NUTTED:
            if not ip and not river and rng.random() < P["trap_freq"]:
                return Action.CHECK_CALL
            return pot or half or Action.CHECK_CALL
        if river:
            if cls == GOOD and half and rng.random() < P["good_bet_freq"] * 0.6:
                return half                             # thin value
            if cls == AIR and pot and rng.random() < P["river_bluff_freq"]:
                return pot                              # polarized bluff
            return Action.CHECK_CALL
        cbet = P["dry_cbet_freq"] if (dry_high and ip and we_raised_pf) else 0.0
        freq = {GOOD: P["good_bet_freq"],
                NUT_DRAW: P["draw_semibluff_ip"] if ip else P["draw_semibluff_oop"],
                WEAK: cbet,
                AIR: max(P["air_bluff_freq"], cbet)}[cls]
        if half and rng.random() < freq * scale:
            return half
        return Action.CHECK_CALL

    pot_odds = obs.to_call / (obs.pot + obs.to_call)
    need = pot_odds + (P["river_call_margin"] if river else P["call_margin"])
    if cls == NUTTED:
        return pot or Action.CHECK_CALL
    if cls == GOOD:
        spr = min(obs.my_stack, obs.opp_stack) / max(obs.pot, 1)
        if (obs.to_call >= CONFIG["stackoff_commit"] * obs.my_stack
                and spr > CONFIG["stackoff_spr"]):
            need = max(need, CONFIG["stackoff_equity"])
        return Action.CHECK_CALL if equity >= need else Action.FOLD
    if cls == NUT_DRAW:
        if nut >= CONFIG["strong_draw_nuttiness"] and pot and rng.random() < CONFIG["draw_raise_freq"]:
            return pot
        return Action.CHECK_CALL if equity + CONFIG["draw_implied_bonus"] >= need else Action.FOLD
    return Action.CHECK_CALL if equity >= need else Action.FOLD


def _decide(obs, memory):
    h = _hand(memory, obs.hand_index)
    _observe(obs, memory, h)
    preflop_hand = obs.street == Street.PREFLOP
    action = _preflop(obs, memory) if preflop_hand else _postflop(obs, memory, h)
    p = memory.get("intermittent_bluff", CONFIG["intermittent_bluff"])
    if p > 0 and action in (Action.CHECK_CALL, Action.FOLD) and (action == Action.FOLD or obs.to_call == 0):
        bad = (preflop_percentile(obs.hole_cards) < CONFIG["pf_open_limp_pct"] if preflop_hand
               else memory.get("last_cls") == AIR)
        if bad and memory["rng"].random() < p:
            raise_ = Action.RAISE_POT if Action.RAISE_POT in obs.legal_actions else Action.RAISE_HALF
            if raise_ in obs.legal_actions:
                action = raise_
                memory["intermittent_count"] = memory.get("intermittent_count", 0) + 1
    if action not in obs.legal_actions:
        action = Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD

    # Bookkeeping for the opponent model and the luck-adjusted results.
    if (action in RAISES and obs.street != Street.PREFLOP
            and memory.get("last_cls") in (AIR, WEAK) and not h.get("bluffed")):
        h["bluffed"] = True
        memory["adv"]["bluff_bets"] += 1
    if action in RAISES and obs.street != Street.PREFLOP:
        h["pending"] = "fold_to_half" if action == Action.RAISE_HALF else "fold_to_pot"
    if obs.street == Street.PREFLOP:
        h["our_pf_raise_last"] = action in RAISES
    allin = ((action == Action.CHECK_CALL and obs.to_call > 0
              and (obs.to_call >= obs.my_stack or obs.opp_stack == 0))
             or (action in RAISES and obs.raise_to[action] - obs.my_bet >= obs.my_stack))
    if allin and h["allin_street"] is None:
        h["allin_street"] = int(obs.street)
    return action


# ----- hand-end analysis -------------------------------------------------------------

def _equity_vs_hand(memory, hero, opp, board):
    """Our equity vs a KNOWN opponent hand from `board` (0-4 cards) on."""
    rng = memory["nprng"]
    dead = set(hero) | set(opp) | set(board)
    deck = np.array([c for c in range(36) if c not in dead], dtype=np.int64)
    m = 5 - len(board)
    if m == 0:
        boards = np.array([sorted(board)], dtype=np.int64)
    elif math.comb(len(deck), m) <= 400:
        from itertools import combinations
        comps = np.array(list(combinations(deck, m)), dtype=np.int64)
        boards = np.sort(np.concatenate([np.broadcast_to(np.array(board, dtype=np.int64),
                                                          (len(comps), len(board))), comps], axis=1), axis=1)
    else:
        comps = deck[np.argpartition(rng.random((400, len(deck))), m, axis=1)[:, :m]]
        boards = np.sort(np.concatenate([np.broadcast_to(np.array(board, dtype=np.int64),
                                                          (400, len(board))), comps], axis=1), axis=1)
    tri = fasteval.triple_indices(boards)
    hs = fasteval.best(fasteval.pair_indices(np.array(sorted(hero)))[None, :], tri)
    os_ = fasteval.best(fasteval.pair_indices(np.array(sorted(opp)))[None, :], tri)
    return float(((hs > os_) + 0.5 * (hs == os_)).mean())


def _opp_river_pct(memory, opp, board, hero):
    """Share of random hands the opponent's river hand beats."""
    rng = memory["nprng"]
    dead = set(opp) | set(board) | set(hero)
    deck = np.array([c for c in range(36) if c not in dead], dtype=np.int64)
    others = np.sort(deck[np.argpartition(rng.random((200, len(deck))), 5, axis=1)[:, :5]], axis=1)
    tri = fasteval.triple_indices(np.array(sorted(board), dtype=np.int64))
    o = fasteval.best(fasteval.pair_indices(np.array(sorted(opp)))[None, :], tri[None, :])
    x = fasteval.best(fasteval.pair_indices(others), tri[None, :])
    return float(((o > x) + 0.5 * (o == x)).mean())


def _stoploss(memory, mode, adj_bb):
    sl = memory["sl"]
    sl["results"][mode].append(adj_bb)
    if sl["mode"] == "revert":
        sl["count"] += 1
        if sl["count"] >= CONFIG["stoploss_window"]:
            sl.update(mode="exploit", count=0, window=[])
            memory["log"].append(f"hand {memory['hands']}: stop-loss over, exploits back on")
        return
    if mode != "exploit":
        return
    w = sl["window"]
    w.append(adj_bb)
    if len(w) >= CONFIG["stoploss_window"]:
        mean = sum(w) / len(w)
        se = (sum((x - mean) ** 2 for x in w) / (len(w) - 1)) ** 0.5 / len(w) ** 0.5
        if mean < -CONFIG["stoploss_se"] * se:
            sl.update(mode="revert", count=0, window=[])
            memory["log"].append(f"hand {memory['hands']}: STOP-LOSS: exploit mean {mean:+.2f} bb/hand "
                                 f"< -{CONFIG['stoploss_se']} SE ({se:.2f}); defaults for "
                                 f"{CONFIG['stoploss_window']} hands")


def _hand_end(result, memory):
    h = memory.get("hand")
    if h is None or h["idx"] != result.hand_index:
        h = _hand(memory, result.hand_index)       # we never acted: opp folded the button
    if h["pending"] is not None:                    # our last postflop bet: did they fold?
        folded = not result.went_to_showdown and result.my_net > 0
        _add(memory, h["pending"], int(folded))
        if folded:
            _add(memory, "aggression", 0)
    if result.went_to_showdown and h["our_pf_raise_last"]:
        h["vpip"] = True                            # they called our preflop raise
    _add(memory, "vpip", int(h["vpip"]))
    _add(memory, "pfr", int(h["pfr"]))
    hero = memory.get("last_hole")
    if result.went_to_showdown and result.opp_hole and hero:
        if h["opp_river_pot"] and len(result.board) == 5:
            weak = _opp_river_pct(memory, result.opp_hole, result.board, hero) < CONFIG["bluff_weak_pct"]
            _add(memory, "river_bluff", int(weak))

    # Did the opponent just SEE one of our bluffs? (only possible at showdown)
    if (h.get("bluffed") and result.went_to_showdown and hero and result.opp_hole
            and len(result.board) == 5
            and _opp_river_pct(memory, hero, result.board, result.opp_hole) < CONFIG["bluff_weak_pct"]):
        memory["adv"]["shown"] += 1

    # Luck-adjusted result (bb): all-in before the river -> equity share of the pot.
    adj = result.my_net
    s = h["allin_street"]
    if result.went_to_showdown and result.opp_hole and hero and s is not None and s < Street.RIVER:
        n_board = {Street.PREFLOP: 0, Street.FLOP: 3, Street.TURN: 4}[Street(s)]
        eq = _equity_vs_hand(memory, hero, result.opp_hole, result.board[:n_board])
        adj = eq * result.pot - result.pot / 2
    memory["hands"] += 1
    _decay(memory)
    _stoploss(memory, h["mode"], adj / 2.0)
    _update_params(memory)
    memory["hand"] = None


def _fresh_memory(memory):
    memory.update(
        hands=0, think=0.0, errors=[],
        rng=random.Random(CONFIG["seed"]), nprng=np.random.default_rng(CONFIG["seed"]),
        opp={k: [0, 0] for k in STATS}, opp_recent={k: [0.0, 0.0] for k in STATS},
        P={k: v for k, v in CONFIG.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
        logged={}, log=[],
        sl={"mode": "exploit", "count": 0, "window": [],
            "results": {"default": [], "exploit": []}},
        class_counts={}, hand=None, last_hole=None, last_cls=None,
        adv={"phase": "advert", "start": 0, "shown": 0, "bluff_bets": 0},
    )


def _record_error(memory, exc):
    memory.setdefault("errors", [])
    if len(memory["errors"]) < 20:
        memory["errors"].append(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")
    if _DEBUG:
        raise exc


def new_half(config: AgentConfig, memory: dict) -> None:
    """Start of each half (memory is a fresh dict)."""
    if _OVERRIDE is not None:
        return _OVERRIDE.new_half(config, memory)
    _fresh_memory(memory)


def choose_action(obs: Observation, memory: dict) -> Action:
    """Return one of obs.legal_actions. Internal errors fall back to check/fold
    (an exception here would count as an error; 5 errors forfeit the half)."""
    if _OVERRIDE is not None:
        action = _OVERRIDE.choose_action(obs, memory)
        return action if action in obs.legal_actions else (
            Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD)
    start = time.perf_counter()
    try:
        if "P" not in memory:
            _fresh_memory(memory)
        memory["last_hole"] = tuple(obs.hole_cards)
        return _decide(obs, memory)
    except Exception as exc:
        _record_error(memory, exc)
        return Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD
    finally:
        memory["think"] = memory.get("think", 0.0) + time.perf_counter() - start


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Update the opponent model, luck-adjusted results and exploit parameters."""
    if _OVERRIDE is not None:
        return _OVERRIDE.on_hand_end(result, memory)
    start = time.perf_counter()
    try:
        if "P" not in memory:
            _fresh_memory(memory)
        _hand_end(result, memory)
    except Exception as exc:
        _record_error(memory, exc)
    finally:
        memory["think"] = memory.get("think", 0.0) + time.perf_counter() - start


def report(memory: dict) -> str:
    """A human-readable summary of one half (used by arena.py)."""
    if "P" not in memory:
        return "(no adaptive state)"
    r = _rates(memory)
    res = memory["sl"]["results"]

    def ms(xs):
        if len(xs) < 2:
            return f"n={len(xs)}"
        m = sum(xs) / len(xs)
        se = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 / len(xs) ** 0.5
        return f"n={len(xs)} {m:+.3f}±{se:.3f} bb/hand"

    lines = [f"hands {memory['hands']}, think {memory['think']:.2f}s, internal errors {len(memory['errors'])}",
             "opp model (long/recent): " + ", ".join(
                 f"{k} {v:.2f}/{_rate(memory, k, True):.2f}" for k, v in r.items()),
             f"luck-adjusted: default {ms(res['default'])}; exploit {ms(res['exploit'])}",
             "classes: " + ", ".join(f"{k} {v}" for k, v in sorted(memory["class_counts"].items())),
             f"exploit log ({len(memory['log'])} entries):"]
    lines += ["  " + x for x in memory["log"][:6]]
    if len(memory["log"]) > 12:
        lines.append("  ...")
    lines += ["  " + x for x in memory["log"][max(6, len(memory["log"]) - 6):]]
    if memory["errors"]:
        lines.append("first internal error: " + memory["errors"][0])
    return "\n".join(lines)

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
