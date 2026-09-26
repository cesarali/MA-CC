"""Post-hoc Santa Fe v3 microscopic and mean-field validation.

No information estimator is implemented here. MI/CMI, uncertainty, nulls and
support stay in the shared MA-CC round-information engine. This module checks
known synthetic transition and sensor kernels against retained event records.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import hashlib
from itertools import combinations
import json
import math
from math import comb
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import load_config
from .game import sigmoid
from .state import SimulationParameters
from .v3_game import replay_next_day, _json

CATEGORIES = ("+1_+1", "+1_0", "-1_-1", "-1_0")


def hypergeometric_category_law(counts: dict[str, int], q: int) -> dict[tuple[int, ...], float]:
    """Exact category counts for distinct messages, including an empty board."""
    sizes = tuple(int(counts.get(category, 0)) for category in CATEGORIES)
    M = sum(sizes)
    q_eff = min(q, M)
    if q < 0 or any(size < 0 for size in sizes):
        raise ValueError("q and category counts must be nonnegative")
    law = {}
    for a in range(min(sizes[0], q_eff) + 1):
        for b in range(min(sizes[1], q_eff - a) + 1):
            for c in range(min(sizes[2], q_eff - a - b) + 1):
                d = q_eff - a - b - c
                if 0 <= d <= sizes[3]:
                    draw = (a, b, c, d)
                    law[draw] = math.prod(comb(n, k) for n, k in zip(sizes, draw)) / comb(M, q_eff)
    return law


def multinomial_category_law(counts: dict[str, int], q: int) -> dict[tuple[int, ...], float]:
    """Paper's with-replacement approximation at the same effective sample size."""
    sizes = tuple(int(counts.get(category, 0)) for category in CATEGORIES)
    M = sum(sizes)
    q_eff = min(q, M)
    if q < 0 or any(size < 0 for size in sizes):
        raise ValueError("q and category counts must be nonnegative")
    if M == 0:
        return {(0, 0, 0, 0): 1.0}
    law = {}
    for a in range(q_eff + 1):
        for b in range(q_eff - a + 1):
            for c in range(q_eff - a - b + 1):
                draw = (a, b, c, q_eff - a - b - c)
                probability = math.factorial(q_eff) / math.prod(math.factorial(k) for k in draw)
                probability *= math.prod((n / M) ** k for n, k in zip(sizes, draw))
                if probability:
                    law[draw] = probability
    return law


def fact_encounter_probability(M: int, copies: int, q: int, *, replacement: bool = False) -> float:
    """Encounter one identity in a message sample; repeated fact IDs remain possible."""
    if M < 0 or copies < 0 or copies > M or q < 0:
        raise ValueError("invalid board/sample counts")
    if M == 0:
        return 0.0
    q_eff = min(q, M)
    if replacement:
        return 1 - (1 - copies / M) ** q_eff
    return 1 - (comb(M - copies, q_eff) / comb(M, q_eff) if M - copies >= q_eff else 0.0)


def sensor_averaged_propensity(peer_target_count: int, N: int, q_c: int,
                              policy_beta: float, policy_threshold: float) -> float:
    """P(U=1 | closed peer board), averaged over its sensor sample."""
    if not (0 <= peer_target_count <= N and 0 < q_c <= N):
        raise ValueError("invalid peer target count or sensor sample")
    return sum(comb(peer_target_count, y) * comb(N-peer_target_count, q_c-y)
               / comb(N, q_c) * sigmoid(policy_beta * (policy_threshold - y / q_c))
               for y in range(max(0, q_c-(N-peer_target_count)), min(q_c, peer_target_count)+1))


def _binary_entropy_nats(p: float) -> float:
    return -sum(x * math.log(x) for x in (p, 1-p) if x > 0)


def _records(value):
    return json.loads(value) if isinstance(value, str) else value


def _state_class(facts: set[int], vote: int, weights: list[int]) -> tuple[int, int, int]:
    r = sum(weights[f] == 1 for f in facts)
    return r, len(facts) - r, vote


def exact_one_step(facts: set[int], vote: int, board: list[dict], weights: list[int],
                   q: int, beta_evidence: float, beta_social: float):
    """Enumerate unordered message samples; return exact class and emission laws."""
    q_eff = min(q, len(board))
    subsets = combinations(range(len(board)), q_eff)
    probability_sample = 1 / comb(len(board), q_eff)
    classes = defaultdict(float)
    emissions = defaultdict(float)
    joint = defaultdict(float)
    for indices in subsets:
        sample = [board[i] for i in indices]
        next_facts = facts | {m["fact_id"] for m in sample if m["fact_id"] is not None}
        e = sum(weights[f] for f in next_facts) / len(next_facts) if next_facts else 0.0
        h = sum(m["vote"] for m in sample) / len(sample) if sample else 0.0
        p_plus = sigmoid(beta_evidence * e + beta_social * h)
        for new_vote, p_vote in ((1, p_plus), (-1, 1 - p_plus)):
            state = _state_class(next_facts, new_vote, weights)
            aligned = any(weights[f] == new_vote for f in next_facts)
            category = f"{new_vote:+d}_{new_vote:+d}" if aligned else f"{new_vote:+d}_0"
            prob = probability_sample * p_vote
            classes[state] += prob
            emissions[category] += prob
            joint[(state, category)] += prob
    return dict(classes), dict(emissions), dict(joint)


def _new_fact_count_law(held: int, total: int, draws: int) -> dict[int, float]:
    """Identity-exchangeable independent message IDs, allowing duplicates."""
    law = {0: 1.0}
    for _ in range(draws):
        following = defaultdict(float)
        for new, probability in law.items():
            gain = (total - held - new) / total
            following[new] += probability * (1 - gain)
            following[new + 1] += probability * gain
        law = dict(following)
    return law


def hmf_one_step(start_class: tuple[int, int, int], board_counts: dict[str, int],
                 F_plus: int, F_minus: int, q: int, beta_evidence: float,
                 beta_social: float, *, sampling: str = "hypergeometric"):
    """Class kernel with exchangeable fact IDs; board identities are hidden.

    The two sampling modes isolate the finite-board participant-sampling law.
    Neither mode is the exact identity-resolved MICRO transition.
    """
    r, s, _ = start_class
    if not (0 <= r <= F_plus and 0 <= s <= F_minus and F_plus > 0 and F_minus > 0):
        raise ValueError("invalid class or fact counts")
    if sampling == "hypergeometric":
        samples = hypergeometric_category_law(board_counts, q)
    elif sampling == "multinomial":
        samples = multinomial_category_law(board_counts, q)
    else:
        raise ValueError("sampling must be hypergeometric or multinomial")
    classes = defaultdict(float)
    emissions = defaultdict(float)
    for draw, sample_probability in samples.items():
        positive_posts = draw[0]
        negative_posts = draw[2]
        q_eff = sum(draw)
        h = ((draw[0] + draw[1]) - (draw[2] + draw[3])) / q_eff if q_eff else 0.0
        for gained_plus, plus_probability in _new_fact_count_law(r, F_plus, positive_posts).items():
            for gained_minus, minus_probability in _new_fact_count_law(s, F_minus, negative_posts).items():
                next_r, next_s = r + gained_plus, s + gained_minus
                e = (next_r - next_s) / (next_r + next_s) if next_r + next_s else 0.0
                p_plus = sigmoid(beta_evidence * e + beta_social * h)
                base = sample_probability * plus_probability * minus_probability
                for vote, p_vote in ((1, p_plus), (-1, 1-p_plus)):
                    probability = base * p_vote
                    classes[(next_r, next_s, vote)] += probability
                    category = f"{vote:+d}_{vote:+d}" if (next_r if vote == 1 else next_s) else f"{vote:+d}_0"
                    emissions[category] += probability
    return dict(classes), dict(emissions)


def _local_coarse_drift(rounds, config, max_states, seed):
    """Independent local HMF and reduced predictions from matched after-night projections."""
    candidates = rounds.loc[rounds["round"] > 0]
    chosen = candidates.sample(n=min(max_states, len(candidates)), random_state=seed)
    rows = []
    for row in chosen.itertuples():
        params = config.cells[int(row.cell_id)].params
        agents = _records(row.agent_states_after_persistence_json)
        weights = _records(row.fact_weights_json)
        Fp = weights.count(1); Fm = weights.count(-1)
        board = _records(row.front_page_json)
        board_counts = Counter(f"{m['vote']:+d}_{m['fact_sign']:+d}" if m["fact_sign"] else
                               f"{m['vote']:+d}_0" for m in board)
        exact = np.zeros(3)
        for agent in agents:
            start = tuple(agent["class"])
            kernel, _, _ = exact_one_step(set(agent["active_fact_ids"]), agent["vote"], board,
                                           weights, params.q, params.beta_evidence, params.beta_social)
            for end, probability in kernel.items():
                exact += probability * np.array([int(end[2] == 1)-int(start[2] == 1),
                                                  (end[0]-start[0])/Fp, (end[1]-start[1])/Fm]) / params.N
        classes = Counter(tuple(agent["class"]) for agent in agents)
        r_mean = sum(agent["class"][0] for agent in agents) / params.N
        s_mean = sum(agent["class"][1] for agent in agents) / params.N
        x = sum(agent["vote"] == 1 for agent in agents) / params.N
        reduced = {}
        for r in range(Fp+1):
            pr = comb(Fp,r)*(r_mean/Fp)**r*(1-r_mean/Fp)**(Fp-r)
            for s in range(Fm+1):
                ps = comb(Fm,s)*(s_mean/Fm)**s*(1-s_mean/Fm)**(Fm-s)
                for vote, pv in ((1,x),(-1,1-x)):
                    reduced[(r,s,vote)] = pr*ps*pv
        for model, density in (("HMF", {key:value/params.N for key,value in classes.items()}),
                               ("REDUCED", reduced)):
            for sampling in ("hypergeometric", "multinomial"):
                prediction = np.zeros(3)
                for start, mass in density.items():
                    if not mass:
                        continue
                    kernel, _ = hmf_one_step(start, board_counts, Fp, Fm, params.q,
                                             params.beta_evidence, params.beta_social,
                                             sampling=sampling)
                    for end, probability in kernel.items():
                        prediction += mass*probability*np.array([
                            int(end[2] == 1)-int(start[2] == 1),
                            (end[0]-start[0])/Fp, (end[1]-start[1])/Fm])
                for index, coordinate in enumerate(("truth_share", "kappa_plus", "kappa_minus")):
                    rows.append({"cell_id":row.cell_id, "seed":row.seed, "round":row.round,
                                 "conditioning":"matched_after_night_projection_and_board_categories",
                                 "model":model, "sampling":sampling, "coordinate":coordinate,
                                 "predicted_A":prediction[index], "micro_exact_A":exact[index],
                                 "signed_error":prediction[index]-exact[index],
                                 "N":params.N, "F_plus":Fp, "F_minus":Fm,
                                 "board_size":len(board), "q_effective":min(params.q,len(board))})
    return pd.DataFrame(rows)


def _plot_xy(table, x, y, path, title, xlabel=None, ylabel=None):
    if table.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(table[x], table[y], alpha=.7)
    values = pd.concat([table[x], table[y]]).dropna()
    if len(values):
        low, high = values.min(), values.max()
        ax.plot([low, high], [low, high], color="gray", linestyle="--")
    ax.set(xlabel=xlabel or x, ylabel=ylabel or y, title=title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _semantic_checks(rounds: pd.DataFrame, micro: pd.DataFrame) -> pd.DataFrame:
    checks = Counter()
    for row in rounds.loc[rounds["round"] > 0].itertuples():
        persistence = _records(row.persistence_events_json)
        before = _records(row.agent_states_before_persistence_json)
        after = _records(row.agent_states_after_persistence_json)
        peer = _records(row.peer_board_json)
        ctrl = _records(row.controller_board_json)
        sensed = _records(row.sensor_sample_ids_json)
        checks["persistence_one_record_per_agent"] += len(persistence) == row.N and len(before) == len(after) == row.N
        checks["peer_count_N"] += len(peer) == row.N
        checks["peer_aligned"] += all(m["fact_sign"] in (0, m["vote"]) for m in peer)
        checks["controller_aligned"] += all(m["vote"] == m["fact_sign"] == row.controller_target for m in ctrl)
        checks["controller_budget"] += len(ctrl) == row.budget_used == row.budget * row.controller_effective_U
        checks["sensor_count"] += len(sensed) == row.controller_sensed_messages
        checks["board_conservation"] += len(_records(row.next_board_json)) == row.N + len(ctrl)
        checks["persistence_boundary"] += all(set(item["after_fact_ids"]) <= set(item["before_fact_ids"])
                                                for item in persistence)
        checks["persistence_extremes"] += (row.rho not in (0, 1) or all(
            (not item["after_fact_ids"] if row.rho == 0 else
             item["after_fact_ids"] == item["before_fact_ids"])
            for item in persistence))
        expected_action = sigmoid(row.policy_beta *
                                  (row.policy_threshold - row.controller_observed_target_share))
        checks["action_probability"] += math.isclose(row.controller_p_act, expected_action,
                                                       rel_tol=0, abs_tol=1e-12)
    for row in micro.itertuples():
        ids = _records(row.sampled_message_ids_json)
        checks["no_same_day_leakage"] += all(not mid.startswith(f"r{row.round}-peer-") for mid in ids)
        checks["distinct_sampled_messages"] += len(ids) == len(set(ids)) == row.q_effective
        checks["vote_probability"] += math.isclose(row.p_vote_plus,
            sigmoid(row.beta_evidence * row.evidence_signal + row.beta_social * row.social_signal),
            rel_tol=0, abs_tol=1e-12)
        eligible = _records(row.eligible_post_fact_ids_json)
        checks["uniform_aligned_post"] += math.isclose(row.p_peer_fact_selection,
            1 / len(eligible) if eligible else 1.0, rel_tol=0, abs_tol=1e-12)
        checks["posted_fact_aligned"] += row.posted_fact_sign in (0, row.vote_after)
        checks["factless_only_if_no_aligned"] += (pd.isna(row.posted_fact_id)) == (not _records(row.eligible_post_fact_ids_json))
        checks["deterministic_acquisition"] += row.p_fact_acquisition_given_sample == 1.0
    round_count = len(rounds.loc[rounds["round"] > 0])
    micro_count = len(micro)
    return pd.DataFrame([{"invariant": key, "passed": value,
                          "checked": micro_count if key in {"no_same_day_leakage", "posted_fact_aligned",
                              "factless_only_if_no_aligned", "deterministic_acquisition", "distinct_sampled_messages",
                              "vote_probability", "uniform_aligned_post"} else round_count,
                          "all_passed": value == (micro_count if key in {"no_same_day_leakage", "posted_fact_aligned",
                              "factless_only_if_no_aligned", "deterministic_acquisition", "distinct_sampled_messages",
                              "vote_probability", "uniform_aligned_post"} else round_count)}
                         for key, value in sorted(checks.items())])


def _kernel_checks(rounds, micro, max_events, seed):
    lookup = {(int(r.cell_id), int(r.seed), int(r.round)): r for r in rounds.itertuples() if r.round > 0}
    rng = np.random.default_rng(seed)
    sample_indices = rng.choice(len(micro), min(max_events, len(micro)), replace=False)
    transition = defaultdict(lambda: [0.0, 0, 0.0])
    emission = defaultdict(lambda: [0.0, 0, 0.0])
    individual = []
    for row in micro.iloc[sample_indices].itertuples():
        parent = lookup[(int(row.cell_id), int(row.seed), int(row.round))]
        weights = _records(parent.fact_weights_json)
        board = _records(parent.front_page_json)
        facts = set(_records(row.facts_before_json))
        vote = int(row.vote_before)
        class_probs, emission_probs, joint = exact_one_step(facts, vote, board, weights,
            int(row.q), float(row.beta_evidence), float(row.beta_social))
        observed_class = tuple(_records(row.class_after_json))
        cat = f"{row.vote_after:+d}_{int(row.posted_fact_sign):+d}" if row.posted_fact_sign else f"{row.vote_after:+d}_0"
        prefix = tuple(_records(row.class_before_json))
        for dest, probability in class_probs.items():
            transition[(prefix, dest)][0] += probability
            transition[(prefix, dest)][2] += probability*(1-probability)
        transition[(prefix, observed_class)][1] += 1
        for category, probability in emission_probs.items():
            emission[(prefix, category)][0] += probability
            emission[(prefix, category)][2] += probability*(1-probability)
        emission[(prefix, cat)][1] += 1
        individual.append({"cell_id": row.cell_id, "seed": row.seed, "round": row.round, "slot": row.slot,
                           "p_observed_transition": class_probs.get(observed_class, 0.0),
                           "p_observed_emission": emission_probs.get(cat, 0.0),
                           "p_observed_joint": joint.get((observed_class, cat), 0.0)})
    transitions = pd.DataFrame([{"start_class": str(key[0]), "end_class": str(key[1]),
        "expected_count": value[0], "observed_count": value[1],
        "residual": value[1] - value[0], "binomial_sd": math.sqrt(value[2]),
        "residual_z": (value[1]-value[0])/math.sqrt(value[2]) if value[2]>0 else math.nan}
        for key, value in transition.items()])
    emissions = pd.DataFrame([{"start_class": str(key[0]), "category": key[1],
        "expected_count": value[0], "observed_count": value[1],
        "residual": value[1] - value[0], "binomial_sd": math.sqrt(value[2]),
        "residual_z": (value[1]-value[0])/math.sqrt(value[2]) if value[2]>0 else math.nan}
        for key, value in emission.items()])
    return pd.DataFrame(individual), transitions, emissions


def _closure(rounds, *, null_draws=32, seed=7):
    entries = []
    rng = np.random.default_rng(seed)
    for row in rounds.itertuples():
        if row.round == 0:
            continue
        agents = _records(row.agent_states_json)
        observed = Counter((a["class"][0], a["class"][1]) for a in agents)
        p = {(r, s): comb(row.F_plus, r) * row.kappa_plus ** r * (1-row.kappa_plus) ** (row.F_plus-r)
             * comb(row.F_minus, s) * row.kappa_minus ** s * (1-row.kappa_minus) ** (row.F_minus-s)
             for r in range(row.F_plus+1) for s in range(row.F_minus+1)}
        tv = .5 * sum(abs(observed.get(key, 0)/row.N - value) for key, value in p.items())
        keys = list(p)
        probabilities = np.array([p[key] for key in keys], dtype=float)
        null_counts = rng.multinomial(row.N, probabilities, size=null_draws)
        null_tv = .5 * np.abs(null_counts/row.N - probabilities).sum(axis=1)
        kl = sum((count/row.N) * math.log((count/row.N)/p[key]) for key, count in observed.items() if p.get(key, 0) > 0)
        if any(p.get(key, 0) == 0 for key in observed):
            kl = math.inf
        entries.append({"cell_id": row.cell_id, "seed": row.seed, "round": row.round,
                        "rho": row.rho, "budget": row.budget, "beta_evidence": row.beta_evidence,
                        "beta_social": row.beta_social, "total_variation": tv, "kl_nats": kl,
                        "tv_null_draws": null_draws, "tv_null_mean": float(null_tv.mean()),
                        "tv_null_q95": float(np.quantile(null_tv, .95)),
                        "tv_excess_over_null_mean": float(tv-null_tv.mean()),
                        "tv_above_null_q95": bool(tv > np.quantile(null_tv, .95)),
                        "r_mean_error": sum(r*observed.get((r,s),0)/row.N for r in range(row.F_plus+1) for s in range(row.F_minus+1)) - row.F_plus*row.kappa_plus,
                        "s_mean_error": sum(s*observed.get((r,s),0)/row.N for r in range(row.F_plus+1) for s in range(row.F_minus+1)) - row.F_minus*row.kappa_minus})
    return pd.DataFrame(entries)


def _board_field(rounds):
    rows = []
    for row in rounds.itertuples():
        if row.round == 0:
            continue
        expected = _records(row.R_bar_json)
        actual = _records(row.B_peer_counts_json)
        for cat in CATEGORIES:
            prob = expected.get(cat, 0.0)
            rows.append({"cell_id": row.cell_id, "seed": row.seed, "round": row.round,
                         "category": cat, "observed_count": actual.get(cat, 0),
                         "expected_count": row.N*prob, "residual": actual.get(cat, 0)-row.N*prob,
                         "multinomial_variance": row.N*prob*(1-prob)})
    return pd.DataFrame(rows)


def _acquisition(rounds, micro):
    lookup = {(int(r.cell_id), int(r.seed), int(r.round)): r for r in rounds.itertuples() if r.round > 0}
    rows = []
    for event in micro.itertuples():
        parent = lookup[(int(event.cell_id), int(event.seed), int(event.round))]
        weights = _records(parent.fact_weights_json)
        board = _records(parent.front_page_json)
        M = len(board)
        before = set(_records(event.facts_before_json))
        acquired = set(_records(event.newly_acquired_fact_ids_json))
        for sign in (-1, 1):
            Fsign = sum(w == sign for w in weights)
            if not Fsign:
                continue
            missing = [f for f, w in enumerate(weights) if w == sign and f not in before]
            csign = sum(m["fact_sign"] == sign for m in board) / M if M else 0.0
            exchangeable = 1 - (1 - csign / Fsign) ** event.q_effective
            for fact in missing:
                copies = sum(m["fact_id"] == fact for m in board)
                rows.append({"cell_id": event.cell_id, "seed": event.seed,
                             "round": event.round, "slot": event.slot,
                             "rho": event.rho, "budget": event.budget,
                             "sign": sign, "fact_id": fact, "board_size": M,
                             "copies_on_board": copies, "q_effective": event.q_effective,
                             "acquired": int(fact in acquired),
                             "p_exact_sample": fact_encounter_probability(M, copies, event.q_effective),
                             "p_with_replacement": fact_encounter_probability(M, copies, event.q_effective, replacement=True),
                             "p_exchangeability": exchangeable,
                             "controller_exposed": int(event.sampled_controller_messages > 0)})
    return pd.DataFrame(rows)

def _hypergeom_sensing(rounds):
    rows = []
    for cell, data in rounds.loc[rounds["round"] > 0].groupby("cell_id"):
        N = int(data.N.iloc[0]); q = int(data.controller_sensed_messages.iloc[0])
        peer_target = data.peer_board_json.map(lambda value: sum(m["vote"] == int(data.controller_target.iloc[0]) for m in _records(value)))
        distribution = peer_target.value_counts(normalize=True)
        joint = defaultdict(float)
        for n, pn in distribution.items():
            for y in range(q+1):
                if y <= n and q-y <= N-n:
                    joint[(n,y)] += pn * comb(n,y)*comb(N-n,q-y)/comb(N,q)
        py = Counter()
        for (_, y), prob in joint.items(): py[y] += prob
        exact = sum(prob*math.log2(prob/(distribution[n]*py[y])) for (n,y),prob in joint.items() if prob > 0)
        rows.append({"cell_id": cell, "n_episodes": data.seed.nunique(), "rounds": len(data),
                     "sensor_sample_size": q, "exact_hypergeometric_mi_bits": exact,
                     "sensor_absolute_error": data.sensor_absolute_error.mean(),
                     "sensor_squared_error": data.sensor_squared_error.mean()})
    return pd.DataFrame(rows)



def _paired_counterfactuals(rounds, config, max_states: int, repetitions: int, seed: int):
    candidates = rounds.loc[(rounds["round"] > 0) & (rounds["round"] < config.params.rounds)]
    if candidates.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    chosen = candidates.sample(n=min(max_states, len(candidates)), random_state=seed)
    lookup = {(int(row.cell_id), int(row.seed), int(row.round)): row
              for row in rounds.itertuples()}
    results, replay_checks, distributions = [], [], []
    for row in chosen.itertuples():
        params = config.cells[int(row.cell_id)].params
        weights = _records(row.fact_weights_json)
        agents = _records(row.agent_states_json)
        peer = _records(row.peer_board_json)
        factual_board = _records(row.next_board_json)
        factual_state = _records(row.rng_state_next_day_json)
        replay_agents, replay_peer = replay_next_day(agents, factual_board, weights, params,
                                                      factual_state, row.round + 1)
        successor = lookup[(int(row.cell_id), int(row.seed), int(row.round)+1)]
        replay_checks.append({"cell_id": row.cell_id, "seed": row.seed, "round": row.round,
                              "agents_match": _json(replay_agents) == successor.agent_states_json,
                              "board_match": _json(replay_peer) == successor.peer_board_json})
        pool = _records(row.controller_fact_pool_json)
        outcomes = {0: [], 1: []}
        for replicate in range(repetitions):
            pair_seed = seed + int(row.cell_id)*100_000_003 + int(row.seed) % 10_000_019 + row.round*100_003 + replicate
            downstream_state = np.random.default_rng(pair_seed).bit_generator.state
            fact_rng = np.random.default_rng(pair_seed + 91_000_003)
            forced = [{"message_id": f"r{row.round}-forced-controller-{j}", "author": params.N,
                       "vote": params.controller_target, "fact_id": int(fact_rng.choice(pool)),
                       "fact_sign": params.controller_target, "source": "controller"}
                      for j in range(params.budget)]
            for action, board in ((0, peer), (1, peer+forced)):
                next_agents, _ = replay_next_day(agents, board, weights, params,
                                                  downstream_state, row.round+1)
                target = sum(a["vote"] == params.controller_target for a in next_agents)
                Fplus = sum(w == 1 for w in weights); Fminus = sum(w == -1 for w in weights)
                kplus = sum(a["class"][0] for a in next_agents)/(params.N*Fplus)
                kminus = sum(a["class"][1] for a in next_agents)/(params.N*Fminus) if Fminus else math.nan
                outcomes[action].append((target, kplus, kminus))
        action0 = np.array(outcomes[0]); action1 = np.array(outcomes[1])
        q_c = int(row.controller_sensed_messages)
        a = sensor_averaged_propensity(int(row.peer_board_target_count), params.N, q_c,
                                       params.policy_beta, params.policy_threshold)
        count0 = Counter(action0[:,0].astype(int)); count1 = Counter(action1[:,0].astype(int))
        q0 = {k: v/repetitions for k,v in count0.items()}; q1 = {k: v/repetitions for k,v in count1.items()}
        mixture = {k: (1-a)*q0.get(k,0)+a*q1.get(k,0) for k in set(q0)|set(q1)}
        t_nats = (1-a)*sum(v*math.log(v/mixture[k]) for k,v in q0.items()) + a*sum(v*math.log(v/mixture[k]) for k,v in q1.items())
        chi_x = float((action1[:,0].mean()-action0[:,0].mean())/params.N)
        paired_difference = (action1[:,0] - action0[:,0]) / params.N
        chi_se = float(np.std(paired_difference, ddof=1) / math.sqrt(repetitions)) if repetitions > 1 else math.nan
        for target_count in sorted(set(q0) | set(q1)):
            distributions.append({"cell_id": row.cell_id, "seed": row.seed, "round": row.round,
                                  "conditioning": "complete_pre_action_snapshot",
                                  "target_count_next": target_count, "Q0": q0.get(target_count, 0.0),
                                  "Q1": q1.get(target_count, 0.0), "Qbar": mixture[target_count],
                                  "n_branch_pairs": repetitions})
        numerator = 2*a*(1-a)*chi_x**2
        results.append({"cell_id":row.cell_id, "seed":row.seed, "round":row.round,
                        "rho":row.rho, "budget":row.budget, "action_propensity":a,
                        "conditioning":"complete_pre_action_snapshot", "n_branches":repetitions,
                        "chi_x":chi_x, "chi_target":chi_x, "chi_target_se":chi_se,
                        "chi_kappa_plus":float(action1[:,1].mean()-action0[:,1].mean()),
                        "chi_kappa_minus":float(action1[:,2].mean()-action0[:,2].mean()),
                        "chi_kappa_ctrl":float((action1[:,1] if params.controller_target==1 else action1[:,2]).mean() -
                                               (action0[:,1] if params.controller_target==1 else action0[:,2]).mean()),
                        "T_pi_nats":t_nats, "H_action_nats":_binary_entropy_nats(a),
                        "eta_IF":t_nats/_binary_entropy_nats(a) if 0 < a < 1 else math.nan,
                        "pinsker_numerator_nats":numerator,
                        "eta_IR":numerator/t_nats if t_nats>0 else math.nan,
                        "bound_satisfied":t_nats+1e-12>=numerator})
    return pd.DataFrame(results), pd.DataFrame(replay_checks), pd.DataFrame(distributions)


def _paired_summary(paired):
    """Aggregate selected complete-state quantities by their sampling weights."""
    if paired.empty:
        return pd.DataFrame()
    rows = []
    for cell, group in paired.groupby("cell_id"):
        # Snapshots are selected uniformly without replacement within the pooled
        # cell's pre-action rounds. This is an equal-weight selected-state mean.
        weight = 1 / len(group)
        T = float(group.T_pi_nats.sum() * weight)
        H = float(group.H_action_nats.sum() * weight)
        B = float(group.pinsker_numerator_nats.sum() * weight)
        rows.append({"cell_id": cell, "conditioning": "complete_pre_action_snapshot",
                     "snapshot_selection": "uniform_without_replacement_from_saved_pre_action_rounds",
                     "n_snapshots": len(group), "snapshot_weight": weight,
                     "mean_chi_target": float(group.chi_target.mean()),
                     "mean_T_pi_nats": T, "mean_H_action_nats": H,
                     "mean_B_IR_nats": B,
                     "eta_IF_ratio_of_sums": T/H if H > 0 else math.nan,
                     "eta_IR_ratio_of_sums": B/T if T > 0 else math.nan,
                     "information_estimator": "branch_distribution_plugin",
                     "bias_warning": "JSD plug-in biased upward at finite branch repetitions"})
    return pd.DataFrame(rows)


def _drift_diffusion(rounds, micro, config, max_states: int, seed: int):
    """Exact MICRO conditional one-slot moments, with both clock conventions.

    These use the identity-resolved saved state and board. They verify the
    simulator's local clock; they are not independently predicted HMF moments.
    """
    candidates = rounds.loc[rounds["round"] > 0]
    chosen = candidates.sample(n=min(max_states, len(candidates)), random_state=seed)
    first = {(int(row.cell_id), int(row.seed), int(row.round)): row
             for row in micro.loc[micro.slot == 0].itertuples()}
    drift, diffusion = [], []
    coordinates = ("truth_share", "kappa_plus", "kappa_minus")
    for row in chosen.itertuples():
        params = config.cells[int(row.cell_id)].params
        weights = _records(row.fact_weights_json)
        board = _records(row.front_page_json)
        agents = _records(row.agent_states_after_persistence_json)
        Fplus = sum(w == 1 for w in weights)
        Fminus = sum(w == -1 for w in weights)
        if not Fplus or not Fminus:
            raise ValueError("v3 local moments require both fact signs")
        mean = np.zeros(3)
        raw = np.zeros((3, 3))
        def change(before, after):
            return np.array([int(after[2] == 1) - int(before[2] == 1),
                             (after[0] - before[0]) / Fplus,
                             (after[1] - before[1]) / Fminus], dtype=float)
        for agent in agents:
            before = tuple(agent["class"])
            classes, _, _ = exact_one_step(set(agent["active_fact_ids"]), agent["vote"], board,
                weights, params.q, params.beta_evidence, params.beta_social)
            for after, probability in classes.items():
                vector = change(before, after)
                mean += probability * vector / params.N
                raw += probability * np.outer(vector, vector) / params.N
        event = first[(int(row.cell_id), int(row.seed), int(row.round))]
        observed = change(tuple(_records(event.class_before_json)),
                          tuple(_records(event.class_after_json)))
        fixed = raw - np.outer(mean, mean)
        common = {"cell_id": row.cell_id, "seed": row.seed, "round": row.round,
                  "model": "MICRO_exact_conditional", "conditioning": "complete_after_night_state_and_frozen_board",
                  "N": params.N, "q_effective": min(params.q, len(board))}
        for i, coordinate in enumerate(coordinates):
            drift.append({**common, "coordinate": coordinate, "A_per_unit_time": mean[i],
                          "observed_one_slot_v": observed[i],
                          "observed_population_increment": observed[i]/params.N})
            for j, other in enumerate(coordinates):
                diffusion.append({**common, "coordinate_i": coordinate, "coordinate_j": other,
                                  "D_P_raw_v_second_moment": raw[i,j],
                                  "D_F_centered_v_covariance": fixed[i,j],
                                  "poisson_population_covariance_per_unit_time": raw[i,j]/params.N,
                                  "fixed_slot_population_covariance_per_unit_time": fixed[i,j]/params.N,
                                  "observed_one_slot_outer_v": observed[i]*observed[j]})
    return pd.DataFrame(drift), pd.DataFrame(diffusion)

def _board_covariance(rounds):
    entries = []
    for cell, data in rounds.loc[rounds["round"] > 0].groupby("cell_id"):
        residuals = []
        theory = []
        for row in data.itertuples():
            p = _records(row.R_bar_json)
            counts = _records(row.B_peer_counts_json)
            residuals.append(np.array([counts.get(k,0)-row.N*p.get(k,0) for k in CATEGORIES]))
            vector = np.array([p.get(k,0) for k in CATEGORIES])
            theory.append(row.N*(np.diag(vector)-np.outer(vector,vector)))
        empirical = np.mean([np.outer(x,x) for x in residuals],axis=0)
        expected = np.mean(theory,axis=0)
        for i,a in enumerate(CATEGORIES):
            for j,b in enumerate(CATEGORIES):
                entries.append({"cell_id":cell,"category_i":a,"category_j":b,
                                "observed_residual_second_moment":empirical[i,j],
                                "multinomial_covariance":expected[i,j],"n_days":len(data)})
    return pd.DataFrame(entries)


def _information_response(rounds, info):
    pooled = info.loc[info.scope == "pooled"]
    rows = []
    for cell, data in rounds.groupby("cell_id"):
        index = pooled.loc[pooled.cell_id == cell].set_index("statistic")
        if "round_target_actuation_cmi" not in index.index or "round_target_susceptibility" not in index.index:
            continue
        t = index.loc["round_target_actuation_cmi"]
        chi = index.loc["round_target_susceptibility"]
        a = float(data.loc[(data["round"] > 0) & (data["round"] < data["round"].max()), "controller_p_act"].mean())
        t_nats = float(t.estimate)*math.log(2)
        numerator = 2*a*(1-a)*float(chi.estimate)**2
        rows.append({"cell_id":cell,"action_propensity":a,"chi_x":chi.estimate,
                     "chi_ci_low":chi.bootstrap_ci_low,"chi_ci_high":chi.bootstrap_ci_high,
                     "T_pi_bits":t.estimate,"T_pi_nats":t_nats,
                     "T_pi_null_mean_bits":t.null_mean,"T_pi_null_p_value":t.null_p_value,
                     "T_pi_ci_low_bits":t.bootstrap_ci_low,"T_pi_ci_high_bits":t.bootstrap_ci_high,
                     "pinsker_numerator_nats":numerator,
                     "eta_IR":numerator/t_nats if t_nats>0 else math.nan,
                     "plug_in_bound_violation":bool(t_nats+1e-12<numerator),
                     "dual_action_event_fraction":t.round_dual_action_event_fraction,
                     "n_rounds":t.n_rounds})
    return pd.DataFrame(rows)


def finite_size_scaling(config, *, sizes=(24,48,96,192), repetitions=100, checkpoint_round=1):
    """Clone one exact initial macrostate across N and sample one v3 day."""
    from .game import SyntheticGame
    if repetitions < 3 or len(sizes) < 2 or checkpoint_round != 1:
        raise ValueError("fixed-macrostate scaling needs >=3 repetitions, >=2 sizes, and checkpoint_round=1")
    base=config.cells[0].params
    if any(int(N) % base.N for N in sizes):
        raise ValueError("each scaling N must be an integer multiple of the base N")
    root=config.results_dir/"validation";root.mkdir(parents=True,exist_ok=True)
    initial=SyntheticGame(replace(base,rounds=1)).run_episode(config.seed+81_000_003)
    state=_records(initial.rounds[0]["agent_states_json"])
    board=_records(initial.rounds[0]["front_page_json"])
    weights=initial.fact_weights
    outcomes=[]
    for N in sizes:
        N=int(N);factor=N//base.N
        params=replace(base,N=N,rounds=1)
        agents=[{**agent,"agent_id":j*base.N+i} for j in range(factor) for i,agent in enumerate(state)]
        front=[{**message,"message_id":f"clone-{j}-{message['message_id']}",
                "author":j*base.N+message["author"]}
               for j in range(factor) for message in board]
        for rep in range(repetitions):
            rng_state=np.random.default_rng(config.seed+N*1_000_003+rep).bit_generator.state
            next_agents,_=replay_next_day(agents,front,weights,params,rng_state,1)
            Fplus=sum(w==1 for w in weights);Fminus=sum(w==-1 for w in weights)
            outcomes.append({"N":N,"repetition":rep,"round":1,
                "truth_share":sum(a["vote"]==1 for a in next_agents)/N,
                "kappa_plus":sum(a["class"][0] for a in next_agents)/(N*Fplus),
                "kappa_minus":sum(a["class"][1] for a in next_agents)/(N*Fminus)})
    details=pd.DataFrame(outcomes);details.to_parquet(root/"finite_size_episodes.parquet",index=False)
    summary=details.groupby("N",as_index=False).agg(**{f"{k}_std":(k,"std") for k in
        ("truth_share","kappa_plus","kappa_minus")})
    slopes=[];fig,ax=plt.subplots(figsize=(7,5))
    for key in ("truth_share","kappa_plus","kappa_minus"):
        subset=summary.loc[summary[f"{key}_std"]>0]
        if len(subset)<2:continue
        x=np.log(subset.N.to_numpy());y=np.log(subset[f"{key}_std"].to_numpy())
        slope,intercept=np.polyfit(x,y,1)
        ax.scatter(subset.N,subset[f"{key}_std"],label=f"{key}: slope {slope:.2f}")
        ax.plot(subset.N,np.exp(intercept)*subset.N**slope)
        bootstrap_slopes=[]
        rng=np.random.default_rng(config.seed+sum(map(ord,key)))
        for _ in range(300):
            stds=[]
            for Nvalue in subset.N:
                values=details.loc[details.N==Nvalue,key].to_numpy()
                stds.append(np.std(rng.choice(values,size=len(values),replace=True),ddof=1))
            if all(value>0 for value in stds):
                bootstrap_slopes.append(np.polyfit(x,np.log(stds),1)[0])
        ci_low,ci_high=(np.quantile(bootstrap_slopes,[.025,.975]) if bootstrap_slopes
                        else (math.nan,math.nan))
        slopes.append({"observable":key,"slope":float(slope),"intercept":float(intercept),
                       "slope_ci_low":float(ci_low),"slope_ci_high":float(ci_high),
                       "n_sizes":len(subset),"repetitions_per_size":repetitions})
    ax.set(xscale="log",yscale="log",xlabel="N",ylabel="standard deviation",
           title="Fixed-macrostate one-day fluctuation scaling")
    ax.legend();fig.tight_layout();fig.savefig(root/"finite_size_scaling.png",dpi=150);plt.close(fig)
    summary.to_csv(root/"finite_size_summary.csv",index=False)
    pd.DataFrame(slopes).to_csv(root/"finite_size_slopes.csv",index=False)
    return {"sizes":list(sizes),"repetitions":repetitions,"slopes":slopes}

def sensing_sample_size(config, *, counts=(10,20,30,50,75,100,200), repetitions=100,
                        null_permutations=99, seed=7):
    """Subsample episodes and reuse the shared sensing MI/null estimator."""
    from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import round_information_analysis
    from .llm_parallel import adapt_trajectories, _summarize_nulls
    root=config.results_dir
    rounds=pd.read_parquet(root/"trajectories"/"round_trajectories.parquet")
    reference=_hypergeom_sensing(rounds).set_index("cell_id")
    results=[]
    for cell,data in rounds.groupby("cell_id"):
        events=adapt_trajectories(data,bins=config.bins)
        groups=defaultdict(list)
        for event in events: groups[event.episode_id].append(event)
        ids=list(groups)
        for n in counts:
            if n>len(ids): continue
            for rep in range(repetitions):
                trial_seed=seed+int(cell)*1_000_003+int(n)*10_003+rep
                chosen=np.random.default_rng(trial_seed).choice(ids,int(n),replace=False)
                selected=[event for episode_id in chosen for event in groups[episode_id]]
                estimates,nulls=round_information_analysis(selected,
                    statistics=("round_target_sensing_mi",),bootstrap_resamples=0,
                    null_permutations=null_permutations,seed=trial_seed)
                value=_summarize_nulls(estimates,nulls)[0]
                results.append({"cell_id":cell,"n_episodes":n,"repetition":rep,
                                "estimate_bits":value["estimate"],
                                "exact_reference_bits":reference.loc[cell,"exact_hypergeometric_mi_bits"],
                                "bias_bits":value["estimate"]-reference.loc[cell,"exact_hypergeometric_mi_bits"],
                                "null_mean_bits":value["null_mean"],
                                "null_p_value":value["null_p_value"],
                                "detected":value["null_p_value"]<.05})
    out=root/"validation";out.mkdir(parents=True,exist_ok=True)
    details=pd.DataFrame(results)
    details.to_parquet(out/"sensing_sample_size_repetitions.parquet",index=False)
    if details.empty: return {"repetitions":0}
    summary=details.groupby(["cell_id","n_episodes"],as_index=False).agg(
        estimate_mean=("estimate_bits","mean"),bias=("bias_bits","mean"),
        estimator_variance=("estimate_bits","var"),
        null_mean=("null_mean_bits","mean"),detection_probability=("detected","mean"),
        p_value_median=("null_p_value","median"))
    summary.to_csv(out/"sensing_sample_size_summary.csv",index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for cell,data in summary.groupby("cell_id"):
        axes[0].plot(data.n_episodes,data.bias,marker="o",label=f"cell {cell}")
        axes[1].plot(data.n_episodes,data.detection_probability,marker="o",label=f"cell {cell}")
    axes[0].set(xlabel="episodes",ylabel="MI bias (bits)")
    axes[1].set(xlabel="episodes",ylabel="P(null p<0.05)",ylim=(0,1))
    axes[0].legend();fig.tight_layout();fig.savefig(out/"sensing_sample_size.png",dpi=150);plt.close(fig)
    return {"repetitions":len(details),"summary_rows":len(summary)}


def _phase_maps(rounds, info, out):
    final=rounds.loc[rounds.groupby(["cell_id","seed"])["round"].idxmax()]
    table=final.groupby("cell_id",as_index=False).agg(
        rho=("rho","first"),budget=("budget","first"),beta_regime=("beta_regime","first"),
        final_target_share=("target_share","mean"),
        final_kappa_ctrl=("kappa_ctrl","mean"))
    if info is not None:
        subset=info.loc[(info.scope=="pooled") & info.statistic.isin(
            ("round_target_actuation_cmi","round_target_susceptibility"))]
        values=subset.pivot(index="cell_id",columns="statistic",values="estimate")
        values.columns=["T_pi_bits" if c=="round_target_actuation_cmi" else "chi_x" for c in values.columns]
        table=table.merge(values,left_on="cell_id",right_index=True,how="left")
    table.to_csv(out/"phase_coordinates.csv",index=False)
    for regime,metric,data in ((regime,metric,group) for regime,group in table.groupby("beta_regime")
                                  for metric in ("final_target_share","final_kappa_ctrl","T_pi_bits","chi_x")
                                  if metric in table):
        pivot=data.pivot(index="rho",columns="budget",values=metric)
        fig,ax=plt.subplots(figsize=(7,4))
        image=ax.imshow(pivot.to_numpy(dtype=float),aspect="auto",origin="lower")
        ax.set_xticks(range(len(pivot.columns)),[str(int(x)) for x in pivot.columns])
        ax.set_yticks(range(len(pivot.index)),[f"{x:g}" for x in pivot.index])
        ax.set(xlabel="controller budget b",ylabel="rho",title=f"{regime}: {metric}")
        fig.colorbar(image,ax=ax);fig.tight_layout()
        fig.savefig(out/f"phase_{regime}_{metric}.png",dpi=150);plt.close(fig)
    return table

def _protocol_manifest(config, rounds, micro, *, seed, max_events, max_drift_states,
                       max_paired_states, paired_repetitions):
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, check=False).stdout.strip() or None
    saved_config = config.results_dir / "config.yaml"
    if saved_config.is_file() and saved_config.read_bytes() != config.path.read_bytes():
        raise ValueError("the saved trajectory config differs from the requested validation config")
    cells = []
    for cell in config.cells:
        data = rounds.loc[rounds.cell_id == cell.cell_id]
        active = data.loc[data["round"] > 0]
        if active.empty:
            continue
        initial = data.loc[data["round"] == 0]
        if len(initial) != config.episodes:
            raise ValueError(f"cell {cell.cell_id}: missing saved round-0 initializations")
        expected_seeds = np.random.SeedSequence([config.seed, cell.cell_id]).generate_state(
            config.episodes, dtype=np.uint64)
        observed_seeds = set(map(int, initial.seed))
        if observed_seeds != set(map(int, expected_seeds)):
            raise ValueError(f"cell {cell.cell_id}: saved seeds differ from resolved schedule")
        first = active.iloc[0]
        weights = _records(first.fact_weights_json)
        cells.append({"cell_id": cell.cell_id, "beta_regime": cell.beta_regime,
                      "parameters": asdict(cell.params), "realized_F_plus": weights.count(1),
                      "realized_F_minus": weights.count(-1), "realized_budget_b": int(first.budget),
                      "realized_sensor_q_c": int(first.controller_sensed_messages),
                      "seed_schedule": list(map(int, expected_seeds)),
                      "saved_episodes": len(initial), "saved_round_rows": len(data),
                      "saved_micro_rows": len(micro.loc[micro.cell_id == cell.cell_id])})
    code_paths = (Path(__file__), Path(__file__).with_name("v3_game.py"),
                  Path(__file__).with_name("game.py"))
    return {"model_version": config.params.model_version, "git_commit": commit,
            "config_path": str(config.path),
            "config_sha256": hashlib.sha256(config.path.read_bytes()).hexdigest(),
            "saved_config_match": saved_config.is_file(),
            "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in code_paths},
            "initial_fact_redundancy": config.params.initial_fact_redundancy,
            "stage_definitions": {"round_0": "initialized population and first front page",
                "before_night": "saved previous end-of-day population",
                "after_night": "all-agent independent fact thinning, before first focal slot",
                "after_day": "population and N peer messages, before controller posts",
                "after_controller": "next front page; current population unchanged",
                "next_day_outcome": "population after next N slots, before next controller posts"},
            "sampling": {"focal": "N slots with replacement", "board": "distinct messages without replacement",
                         "fact_identity": "repeated IDs across messages allowed",
                         "controller": "target fact IDs uniformly with replacement"},
            "validation_settings": {"seed": seed, "max_events": max_events,
                "max_drift_states": max_drift_states, "max_paired_states": max_paired_states,
                "paired_repetitions": paired_repetitions}, "cells": cells}


def _category_sampling_checks(rounds, micro):
    first = micro.loc[micro.slot == 0].set_index(["cell_id", "seed", "round"])
    rows = []
    for row in rounds.loc[rounds["round"] > 0].itertuples():
        board = _records(row.front_page_json)
        counts = Counter(f"{m['vote']:+d}_{m['fact_sign']:+d}" if m["fact_sign"] else
                         f"{m['vote']:+d}_0" for m in board)
        if any(key not in CATEGORIES for key in counts):
            raise ValueError("v3 front page contains a cross-sign category")
        exact = hypergeometric_category_law(counts, int(row.q))
        approx = multinomial_category_law(counts, int(row.q))
        tv = .5 * sum(abs(exact.get(key, 0) - approx.get(key, 0))
                         for key in set(exact) | set(approx))
        event = first.loc[(row.cell_id, row.seed, row.round)]
        ids = set(_records(event.sampled_message_ids_json))
        observed = Counter(f"{m['vote']:+d}_{m['fact_sign']:+d}" if m["fact_sign"] else
                           f"{m['vote']:+d}_0" for m in board if m["message_id"] in ids)
        key = tuple(observed.get(category, 0) for category in CATEGORIES)
        rows.append({"cell_id": row.cell_id, "seed": row.seed, "round": row.round,
                     "board_size": len(board), "q_effective": min(int(row.q), len(board)),
                     "sampled_category_counts_json": _json(key),
                     "p_observed_hypergeometric": exact.get(key, 0.0),
                     "p_observed_multinomial": approx.get(key, 0.0),
                     "sampling_law_tv": tv, "model_exact": "MICRO_hypergeometric_categories",
                     "model_approx": "HMF_multinomial_sampling_only"})
    return pd.DataFrame(rows)


def _voting_calibration(micro):
    data = micro[["p_vote_plus", "vote_after"]].copy()
    data["observed_positive"] = (data.vote_after == 1).astype(float)
    data["bin"] = pd.cut(data.p_vote_plus, bins=np.linspace(0, 1, 11), include_lowest=True)
    result = data.groupby("bin", observed=True, as_index=False).agg(
        n=("observed_positive", "size"), predicted=("p_vote_plus", "mean"),
        observed=("observed_positive", "mean")).assign(
        binomial_se=lambda x: np.sqrt(x.predicted * (1-x.predicted) / x.n))
    result["bin"] = result["bin"].astype(str)
    return result


def validate(config, *, max_events=1000, max_drift_states=12,
             max_paired_states=20, paired_repetitions=100, seed=7):
    if config.params.model_version != "santa_fe_epistemic_feedback_v3":
        raise ValueError("theory validation requires an explicit v3 config")
    root = config.results_dir
    rounds = pd.read_parquet(root / "trajectories" / "round_trajectories.parquet")
    micro = pd.read_parquet(root / "trajectories" / "micro_trajectories.parquet")
    if rounds.empty or micro.empty:
        raise ValueError("saved v3 round and micro trajectories are required")
    out = root / "validation"; out.mkdir(parents=True, exist_ok=True)
    manifest = _protocol_manifest(config, rounds, micro, seed=seed, max_events=max_events,
        max_drift_states=max_drift_states, max_paired_states=max_paired_states,
        paired_repetitions=paired_repetitions)
    (out / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    category = _category_sampling_checks(rounds, micro)
    category.to_parquet(out / "category_sampling_validation.parquet", index=False)
    voting = _voting_calibration(micro)
    voting.to_parquet(out / "voting_calibration.parquet", index=False)
    checks = _semantic_checks(rounds, micro)
    checks.to_csv(out / "semantic_invariants.csv", index=False)
    if not checks.all_passed.all():
        raise ValueError("v3 semantic invariant failed; inspect validation/semantic_invariants.csv")
    individual, transition, emission = _kernel_checks(rounds, micro, max_events, seed)
    individual.to_parquet(out / "one_step_probabilities.parquet", index=False)
    transition.to_parquet(out / "transition_kernel_validation.parquet", index=False)
    emission.to_parquet(out / "emission_kernel_validation.parquet", index=False)
    closure = _closure(rounds, seed=seed)
    closure.to_parquet(out / "closure_validation.parquet", index=False)
    board = _board_field(rounds)
    board.to_parquet(out / "board_field_validation.parquet", index=False)
    board_cov = _board_covariance(rounds)
    board_cov.to_parquet(out / "board_covariance_validation.parquet", index=False)
    acquisition = _acquisition(rounds, micro)
    acquisition.to_parquet(out / "acquisition_validation.parquet", index=False)
    sensing = _hypergeom_sensing(rounds)
    info_path = root / "information" / "llm_parallel" / "round_information_estimates.parquet"
    info = None
    if info_path.is_file():
        info = pd.read_parquet(info_path)
        response = _information_response(rounds, info)
        response.to_parquet(out / "response_estimates.parquet", index=False)
        subset = info.loc[(info.scope == "pooled") & (info.statistic == "round_target_sensing_mi"),
                          ["cell_id", "estimate", "null_mean", "null_p_value", "bootstrap_ci_low", "bootstrap_ci_high"]]
        sensing = sensing.merge(subset.rename(columns={"estimate":"empirical_sensing_mi_bits"}), on="cell_id", how="left")
    sensing.to_csv(out / "sensing_validation.csv", index=False)
    _phase_maps(rounds, info, out)
    drift, diffusion = _drift_diffusion(rounds, micro, config, max_drift_states, seed)
    coarse_drift = _local_coarse_drift(rounds, config, max_drift_states, seed)
    coarse_drift.to_parquet(out / "coarse_local_drift_validation.parquet", index=False)
    drift.to_parquet(out / "micro_exact_drift_validation.parquet", index=False)
    diffusion.to_parquet(out / "micro_exact_diffusion_validation.parquet", index=False)
    paired, replay, branch_laws = _paired_counterfactuals(rounds, config, max_paired_states, paired_repetitions, seed)
    paired.to_parquet(out / "paired_counterfactuals.parquet", index=False)
    branch_laws.to_parquet(out / "paired_branch_laws.parquet", index=False)
    paired_summary = _paired_summary(paired)
    paired_summary.to_parquet(out / "paired_summary.parquet", index=False)
    replay.to_csv(out / "path_replay_validation.csv", index=False)
    if not replay.empty and not (replay.agents_match & replay.board_match).all():
        raise ValueError("factual next-day replay failed; inspect path_replay_validation.csv")
    _plot_xy(transition, "expected_count", "observed_count", out / "transition_kernel.png", "One-step transition kernel")
    _plot_xy(emission, "expected_count", "observed_count", out / "emission_kernel.png", "Peer emission kernel")
    _plot_xy(board, "expected_count", "observed_count", out / "board_mean.png", "Peer board counts")
    _plot_xy(board_cov, "multinomial_covariance", "observed_residual_second_moment",
             out / "board_covariance.png", "Peer board covariance approximation")
    if not drift.empty:
        drift_plot = drift.groupby("coordinate", as_index=False)[["A_per_unit_time", "observed_one_slot_v"]].mean()
        _plot_xy(drift_plot, "A_per_unit_time", "observed_one_slot_v", out / "drift.png", "MICRO conditional drift")
    if not diffusion.empty:
        diffusion_plot = diffusion.groupby(["coordinate_i", "coordinate_j"], as_index=False)[["D_P_raw_v_second_moment", "observed_one_slot_outer_v"]].mean()
        _plot_xy(diffusion_plot, "D_P_raw_v_second_moment", "observed_one_slot_outer_v", out / "diffusion.png", "MICRO raw one-slot second moment")
    if not coarse_drift.empty:
        for model, sampling in (("HMF", "hypergeometric"), ("HMF", "multinomial"),
                                ("REDUCED", "hypergeometric"), ("REDUCED", "multinomial")):
            subset = coarse_drift.loc[(coarse_drift.model == model) &
                                      (coarse_drift.sampling == sampling)]
            _plot_xy(subset, "micro_exact_A", "predicted_A",
                     out / f"local_{model.lower()}_{sampling}.pdf",
                     f"{model} local drift: {sampling} sampling")
    if not paired.empty:
        _plot_xy(paired, "pinsker_numerator_nats", "T_pi_nats", out / "pinsker_bound.png", "Paired controller information bound")
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(paired.budget, paired.eta_IR)
        ax.set(xlabel="budget", ylabel="eta_IR", title="Paired information-response efficiency")
        fig.tight_layout(); fig.savefig(out / "eta_IR.png", dpi=150); plt.close(fig)
    acquisition_bins = acquisition.copy()
    acquisition_bins["probability_bin"] = pd.cut(acquisition_bins.p_exchangeability, bins=np.linspace(0,1,11), include_lowest=True)
    acquisition_plot = acquisition_bins.groupby("probability_bin", observed=True, as_index=False).agg(
        p_exchangeability=("p_exchangeability","mean"), acquired=("acquired","mean"))
    _plot_xy(acquisition_plot, "p_exchangeability", "acquired", out / "acquisition.png", "Fact acquisition approximation")
    _plot_xy(sensing.dropna(subset=["empirical_sensing_mi_bits"]) if "empirical_sensing_mi_bits" in sensing else pd.DataFrame(),
             "exact_hypergeometric_mi_bits", "empirical_sensing_mi_bits", out / "sensing_mi.png", "Exact vs estimated sensing MI")
    fig, ax = plt.subplots(figsize=(7,4))
    for cell, group in closure.groupby("cell_id"):
        ax.plot(group.groupby("round").total_variation.mean(), label=f"cell {cell}")
    ax.set(xlabel="round", ylabel="closure total variation", title="Reduced epistemic closure error")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "closure_error.png", dpi=150); plt.close(fig)
    lines = ["# Santa Fe v3 validation", "", f"Cells: {rounds.cell_id.nunique()}; episodes: {rounds.groupby('cell_id').seed.nunique().sum()}; microscopic events: {len(micro)}.",
             f"Semantic invariants: {int(checks.all_passed.sum())}/{len(checks)} passed.",
             f"Exact one-step kernel compared on {len(individual)} sampled events; minimum realized joint probability: {individual.p_observed_joint.min():.4g}.",
             f"Transition entries beyond 3 conditional SD: {int(transition.residual_z.abs().gt(3).sum())}/{len(transition)} overall, {int(transition.loc[transition.expected_count>=1, 'residual_z'].abs().gt(3).sum())}/{int((transition.expected_count>=1).sum())} with expected count >=1.",
             f"Emission entries beyond 3 conditional SD: {int(emission.residual_z.abs().gt(3).sum())}/{len(emission)} overall, {int(emission.loc[emission.expected_count>=1, 'residual_z'].abs().gt(3).sum())}/{int((emission.expected_count>=1).sum())} with expected count >=1.",
             f"Mean reduced-closure total variation: {closure.total_variation.mean():.4f}; same-N binomial null mean {closure.tv_null_mean.mean():.4f}; fraction above snapshot null 95th percentile {closure.tv_above_null_q95.mean():.3f}. Null quantiles use 32 draws per snapshot and are diagnostic, not calibrated cross-time hypothesis tests.",
             f"Mean absolute peer-board residual: {board.residual.abs().mean():.3f} messages.",
             f"Observed fact-acquisition rate {acquisition.acquired.mean():.4f}; exact sampling prediction {acquisition.p_exact_sample.mean():.4f}; exchangeability approximation {acquisition.p_exchangeability.mean():.4f}.",
             "", "The transition and emission comparisons aggregate exact conditional probabilities over the sampled starting states and frozen boards.",
             "The board multinomial and exchangeability acquisition formulas are approximations; their residuals are retained rather than tuned away. Sparse expected transition counts make normal z-scores unstable.",
             f"Mean finite-board category-law TV (hypergeometric versus multinomial): {category.sampling_law_tv.mean():.4f}.",
             f"Matched local drift MAE by model and sampling: {coarse_drift.groupby(['model','sampling']).signed_error.apply(lambda v: round(v.abs().mean(), 4)).to_dict()}.",
             "HMF and REDUCED are independent one-slot class/closure predictions from matched after-night projections. The saved exact MICRO kernel is only a local rule check; none of these rows is an evolved theory trajectory.",
             f"Replay-verified paired branches: {len(paired)} states × {paired_repetitions} replicates; plug-in Pinsker consistency check satisfied in {int(paired.bound_satisfied.sum()) if not paired.empty else 0}/{len(paired)} sampled states.",
             "Paired action propensity averages the hypergeometric sensor over each closed peer board. T_pi and efficiencies are finite-sample plug-in values and may have upward information bias; the paired summary uses ratios of weighted sums.",
             f"Exact MICRO local moments computed at {len(drift) // 3} saved after-night states. Fixed-slot covariance uses D_F=D_P-AA^T; Poisson uses D_P. One observed slot per state cannot validate covariance.",
             "SDE trajectory comparison requires a separately specified reduced SDE and is not inferred from these tables.",
             "A sensing subsample equal to the entire reference episode pool repeats identical data; its across-repetition variance is not a power estimate.", ""]
    (out / "validation_report.md").write_text("\n".join(lines))
    alignment = ["# Santa Fe v3 protocol alignment", "",
        "The resolved protocol and seed schedule are in `protocol_manifest.json`.", "",
        "| Comparison | Alignment and remaining approximation |", "| --- | --- |",
        "| MICRO | Exact finite-N identity-resolved simulator; round-0 states and boards are saved. |",
        "| HMF local | Initialized from the saved after-night class histogram and frozen board categories. It assumes exchangeable fact identities; hypergeometric and multinomial message sampling are reported separately. |",
        "| REDUCED local | Uses the matched x and signed coverages but assumes independent binomial fact counts and vote/count independence. |",
        "| Local covariance | MICRO exact raw D_P and fixed-slot centered D_F are saved separately. One observed slot per state is not a covariance estimate. |",
        "| Paired action | Forced branches start after the peer board closes. Controller IDs are redrawn uniformly per pair. Propensity is averaged over sensor outcomes, and next-day population is measured before its later action. |",
        "| Existing MI/CMI | Calculated by the shared engine from retained trajectories; observational conditioning differs from complete-state paired JSD. |",
        "", "Full stochastic HMF, evolved HMF/REDUCED trajectories, and phase-map residuals are not implemented by this local validator. Do not interpret MICRO-derived one-step moments as those predictions.", ""]
    (out / "protocol_alignment.md").write_text("\n".join(alignment))
    (out / "analysis_recipe.yaml").write_bytes(config.path.read_bytes())
    return {"results_dir": str(out), "checked_events": len(individual), "invariants_passed": bool(checks.all_passed.all()),
            "mean_closure_tv": float(closure.total_variation.mean()),
            "paired_states": len(paired), "drift_rows": len(drift)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-events", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-drift-states", type=int, default=12)
    parser.add_argument("--max-paired-states", type=int, default=20)
    parser.add_argument("--paired-repetitions", type=int, default=100)
    parser.add_argument("--scaling-repetitions", type=int, default=0)
    parser.add_argument("--scaling-sizes", type=int, nargs="+", default=[24,48,96,192])
    parser.add_argument("--scaling-round", type=int, default=1)
    parser.add_argument("--sensing-sample-sizes", type=int, nargs="+")
    parser.add_argument("--sensing-repetitions", type=int, default=100)
    parser.add_argument("--sensing-null-permutations", type=int, default=99)
    args = parser.parse_args(argv)
    result = validate(load_config(args.config), max_events=args.max_events,
                      max_drift_states=args.max_drift_states,
                      max_paired_states=args.max_paired_states,
                      paired_repetitions=args.paired_repetitions, seed=args.seed)
    report = load_config(args.config).results_dir / "validation" / "validation_report.md"
    if args.scaling_repetitions:
        result["finite_size_scaling"] = finite_size_scaling(load_config(args.config),
            sizes=args.scaling_sizes, repetitions=args.scaling_repetitions,
            checkpoint_round=args.scaling_round)
        lines = ["", "## Optional finite-size check", ""]
        for item in result["finite_size_scaling"]["slopes"]:
            lines.append(f"- {item['observable']}: fitted log-log slope {item['slope']:.3f} (episode-bootstrap 95% interval [{item['slope_ci_low']:.3f}, {item['slope_ci_high']:.3f}]) over {item['n_sizes']} sizes, {item['repetitions_per_size']} episodes each.")
        lines.append("All sizes clone the same initial class and board fractions exactly; this measures one-day fluctuations from a fixed macrostate.")
        report.write_text(report.read_text() + "\n".join(lines) + "\n")
    if args.sensing_sample_sizes:
        result["sensing_sample_size"] = sensing_sample_size(load_config(args.config),
            counts=args.sensing_sample_sizes,repetitions=args.sensing_repetitions,
            null_permutations=args.sensing_null_permutations,seed=args.seed)
        sensing = pd.read_csv(load_config(args.config).results_dir / "validation" / "sensing_sample_size_summary.csv")
        lines = ["", "## Optional sensing sample-size check", ""]
        for row in sensing.itertuples():
            lines.append(f"- cell {row.cell_id}, n={row.n_episodes}: mean MI bias {row.bias:+.3f} bits; null detection probability {row.detection_probability:.2f}.")
        report.write_text(report.read_text() + "\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
