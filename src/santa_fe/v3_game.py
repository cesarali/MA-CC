"""Exact finite-population Santa Fe v3 day/night process and event ledger.

The ledger uses JSON strings in canonical round/micro rows so Parquet retains
fact identities and probability terms without relying on inferred nested schemas.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from math import comb, log, prod
import json
import math

import numpy as np

from .game import binary_entropy, make_fact_weights, sigmoid
from .state import EpisodeResult, Message


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _probability_log(probability: float) -> float:
    return math.log(probability) if probability > 0 else -math.inf


def _class(agent, weights) -> tuple[int, int, int]:
    positive = sum(weights[f] == 1 for f in agent.active_facts)
    return int(positive), int(len(agent.active_facts) - positive), int(agent.vote)


def _agents(agents, weights) -> list[dict]:
    return [{"agent_id": i, "vote": int(a.vote), "active_fact_ids": sorted(a.active_facts),
             "class": _class(a, weights)} for i, a in enumerate(agents)]


def _occupancy(agents, weights) -> list[dict]:
    counts = Counter(_class(agent, weights) for agent in agents)
    return [{"r": r, "s": s, "v": v, "count": n, "fraction": n / len(agents)}
            for (r, s, v), n in sorted(counts.items())]


def _message(message: Message, weights) -> dict:
    return {"message_id": message.message_id, "author": int(message.author), "vote": int(message.vote),
            "fact_id": message.fact_id, "fact_sign": int(weights[message.fact_id]) if message.fact_id is not None else 0,
            "source": "controller" if message.is_controller else "peer"}


def _category(message: Message, weights) -> str:
    sign = int(weights[message.fact_id]) if message.fact_id is not None else 0
    return f"{message.vote:+d}_{sign:+d}" if sign else f"{message.vote:+d}_0"


def _counts(messages, weights) -> dict[str, int]:
    source = Counter(_category(m, weights) for m in messages)
    return {key: int(source.get(key, 0)) for key in ("+1_+1", "+1_0", "-1_-1", "-1_0", "+1_-1", "-1_+1")}


def _coverage(agents, weights, target):
    plus = int(sum(weights == 1))
    minus = int(sum(weights == -1))
    plus_counts = np.array([sum(weights[f] == 1 for f in a.active_facts) for a in agents], dtype=float)
    minus_counts = np.array([sum(weights[f] == -1 for f in a.active_facts) for a in agents], dtype=float)
    total = plus_counts + minus_counts
    union = set().union(*(a.active_facts for a in agents))
    balance = np.divide(plus_counts - minus_counts, total, out=np.zeros_like(total), where=total > 0)
    k_plus = float(plus_counts.mean() / plus)
    k_minus = float(minus_counts.mean() / minus) if minus else math.nan
    votes = np.array([a.vote for a in agents], dtype=float)
    def correlation(values):
        return float(np.corrcoef(values, votes)[0,1]) if values.std() > 0 and votes.std() > 0 else math.nan
    return {"kappa_mean_coverage": float(total.mean() / len(weights)),
            "kappa_population_coverage": len(union) / len(weights),
            "kappa_full_agent_fraction": float(np.mean(total == len(weights))),
            "kappa_mean_evidence_balance": float(balance.mean()), "kappa_evidence_sd": float(balance.std()),
            "F_plus": plus, "F_minus": minus, "kappa_plus": k_plus, "kappa_minus": k_minus,
            "kappa_ctrl": k_plus if target == 1 else k_minus,
            "fraction_all_plus_facts": float(np.mean(plus_counts == plus)),
            "fraction_all_minus_facts": float(np.mean(minus_counts == minus)) if minus else math.nan,
            "epistemic_r_variance": float(plus_counts.var()), "epistemic_s_variance": float(minus_counts.var()),
            "epistemic_rs_covariance": float(np.mean((plus_counts - plus_counts.mean()) * (minus_counts - minus_counts.mean()))),
            "epistemic_r_vote_correlation": correlation(plus_counts),
            "epistemic_s_vote_correlation": correlation(minus_counts)}


def _sample_messages(board, count, rng):
    if not count:
        return [], 1.0
    indices = rng.choice(len(board), size=count, replace=False)
    sampled = [board[int(i)] for i in indices]
    # Selection uses ordered draws without replacement.
    probability = 1.0 / prod(range(len(board) - count + 1, len(board) + 1))
    return sampled, probability


def run_v3_episode(game, seed: int) -> EpisodeResult:
    p = game.p
    rng = np.random.default_rng(seed)
    weights = make_fact_weights(p.F, p.truth_fact_fraction, rng, n_truth=p.F_plus)
    controller_pool = [int(f) for f in np.flatnonzero(weights == p.controller_target)]
    if not controller_pool:
        raise ValueError("v3 controller has no target-aligned fact")
    agents = game.initialize(rng, weights)
    board = []
    for i, agent in enumerate(agents):
        eligible = sorted(f for f in agent.active_facts if weights[f] == agent.vote)
        fact = int(rng.choice(eligible)) if eligible else None
        board.append(Message(i, int(agent.vote), fact, False, f"r0-peer-{i}"))
    rounds, micro = [], []
    initial = {"round": 0, "fact_weights_json": _json(weights.tolist()),
               "agent_states_json": _json(_agents(agents, weights)), "occupancy_json": _json(_occupancy(agents, weights)),
               "front_page_json": _json([_message(m, weights) for m in board]),
               "B_total_counts_json": _json(_counts(board, weights)),
               "controller_fact_pool_json": _json(controller_pool),
               "controller_target": p.controller_target, "truth_share": game.truth_share(agents),
               "target_share": game.target_share(agents), "vote_entropy_bits": binary_entropy(game.truth_share(agents)),
               "controller_U": 0, "controller_effective_U": 0, "controller_p_act": 0.0,
               "controller_observed_target_share": math.nan, "controller_sensed_messages": 0,
               "controller_sensor_target_count": 0, "controller_messages_next_board": 0,
               "budget_used": 0, "controller_requested_budget": p.budget,
               "sensor_sample_ids_json": _json([]), "controller_selected_fact_ids_json": _json([]),
               "controller_message_ids_json": _json([]), "peer_board_json": _json([]),
               "rng_state_next_day_json": _json(rng.bit_generator.state),
               "controller_board_json": _json([]), "p_persistence_event": 1.0,
               "log_p_persistence_event": 0.0, "p_sensor_outcome": 1.0,
               "p_action_given_sensor": 1.0, "p_controller_fact_selection": 1.0,
               "p_sensor_sample_ids": 1.0, "log_p_sensor_sample_ids": 0.0,
               **_coverage(agents, weights, p.controller_target)}
    rounds.append(initial)
    for day in range(1, p.rounds + 1):
        B_current = list(board)
        before_persistence = _agents(agents, weights)
        persistence = []
        kept, lost = 0, 0
        plus_lost, minus_lost = 0, 0
        for agent_id, agent in enumerate(agents):
            before = sorted(agent.active_facts)
            retained = {f for f in before if rng.random() < p.rho}
            gone = sorted(set(before) - retained)
            agent.active_facts = retained
            kept += len(retained)
            lost += len(gone)
            plus_lost += sum(weights[f] == 1 for f in gone)
            minus_lost += sum(weights[f] == -1 for f in gone)
            persistence.append({"agent_id": agent_id, "before_fact_ids": before,
                                "after_fact_ids": sorted(retained), "lost_fact_ids": gone,
                                "class_before": before_persistence[agent_id]["class"],
                                "class_after": _class(agent, weights)})
        if p.rho == 0:
            persistence_probability = float(kept == 0)
            persistence_log = 0.0 if kept == 0 else -math.inf
        elif p.rho == 1:
            persistence_probability = float(lost == 0)
            persistence_log = 0.0 if lost == 0 else -math.inf
        else:
            persistence_log = kept * log(p.rho) + lost * log(1 - p.rho)
            persistence_probability = math.exp(persistence_log)
        C_peer = []
        predictive_categories = Counter()
        for slot in range(p.N):
            focal = int(rng.integers(p.N))
            agent = agents[focal]
            before = sorted(agent.active_facts)
            class_before = _class(agent, weights)
            q_eff = min(p.q, len(B_current))
            sampled, p_sample = _sample_messages(B_current, q_eff, rng)
            acquired = sorted({m.fact_id for m in sampled if m.fact_id is not None} - agent.active_facts)
            controller_exposed = {m.fact_id for m in sampled if m.is_controller and m.fact_id is not None}
            peer_exposed = {m.fact_id for m in sampled if not m.is_controller and m.fact_id is not None}
            controller_acquired = sorted(controller_exposed - peer_exposed - agent.active_facts)
            agent.active_facts.update(m.fact_id for m in sampled if m.fact_id is not None)
            class_after_acquisition = _class(agent, weights)
            evidence = game.evidence_signal(agent, weights)
            social = game.social_signal(sampled)
            logit = p.beta_evidence * evidence + p.beta_social * social
            p_truth = sigmoid(logit)
            new_vote = 1 if rng.random() < p_truth else -1
            p_vote = p_truth if new_vote == 1 else 1 - p_truth
            agent.vote = new_vote
            eligible_plus = [f for f in agent.active_facts if weights[f] == 1]
            eligible_minus = [f for f in agent.active_facts if weights[f] == -1]
            predictive_categories["+1_+1" if eligible_plus else "+1_0"] += p_truth
            predictive_categories["-1_-1" if eligible_minus else "-1_0"] += 1 - p_truth
            eligible = sorted(f for f in agent.active_facts if weights[f] == new_vote)
            fact = int(rng.choice(eligible)) if eligible else None
            p_post = 1 / len(eligible) if eligible else 1.0
            emitted = Message(focal, new_vote, fact, False, f"r{day}-peer-{slot}")
            C_peer.append(emitted)
            micro.append({"round": day, "slot": slot, "focal": focal,
                          "class_before_json": _json(class_before), "class_after_acquisition_json": _json(class_after_acquisition),
                          "class_after_json": _json(_class(agent, weights)), "facts_before_json": _json(before),
                          "facts_after_json": _json(sorted(agent.active_facts)), "newly_acquired_fact_ids_json": _json(acquired),
                          "sampled_message_ids_json": _json([m.message_id for m in sampled]),
                          "sampled_vote_signs_json": _json([m.vote for m in sampled]),
                          "sampled_fact_ids_json": _json([m.fact_id for m in sampled]),
                          "sampled_fact_signs_json": _json([int(weights[m.fact_id]) if m.fact_id is not None else 0 for m in sampled]),
                          "sampled_controller_message_ids_json": _json([m.message_id for m in sampled if m.is_controller]),
                          "controller_exposed_fact_ids_json": _json(sorted(controller_exposed)),
                          "controller_only_acquired_fact_ids_json": _json(controller_acquired),
                          "sampled_controller_messages": sum(m.is_controller for m in sampled),
                          "eligible_post_fact_ids_json": _json(eligible), "posted_message_id": emitted.message_id,
                          "posted_fact_id": fact, "posted_fact_sign": int(weights[fact]) if fact is not None else 0,
                          "vote_before": class_before[2], "vote_after": new_vote,
                          "facts_before": len(before), "facts_after": len(agent.active_facts),
                          "facts_acquired": len(acquired), "facts_lost": 0, "q_effective": q_eff,
                          "evidence_signal": evidence, "social_signal": social, "vote_logit": logit,
                          "p_vote_plus": p_truth, "p_focal_agent": 1 / p.N,
                          "p_board_sample": p_sample, "p_fact_acquisition_given_sample": 1.0,
                          "p_vote_realized": p_vote, "p_peer_fact_selection": p_post,
                          "log_p_focal_agent": -math.log(p.N), "log_p_fact_acquisition_given_sample": 0.0,
                          "log_p_board_sample": _probability_log(p_sample),
                          "log_p_vote_realized": _probability_log(p_vote),
                          "log_p_peer_fact_selection": _probability_log(p_post)})
        q_sensor = min(p.N, max(1, round(p.sensing_fraction * p.N)))
        sensed, p_sensor = _sample_messages(C_peer, q_sensor, rng)
        Y = sum(m.vote == p.controller_target for m in sensed)
        peer_target_share = sum(m.vote == p.controller_target for m in C_peer) / p.N
        observed = Y / q_sensor
        peer_target_count = sum(m.vote == p.controller_target for m in C_peer)
        p_sensor_y = comb(peer_target_count, Y) * comb(p.N-peer_target_count, q_sensor-Y) / comb(p.N, q_sensor)
        U, p_act = game.controller_policy(observed, rng)
        p_action = p_act if U else 1 - p_act
        controller = []
        if U and p.budget > 0:
            for index in range(p.budget):
                fact = int(rng.choice(controller_pool))
                controller.append(Message(p.N, p.controller_target, fact, True, f"r{day}-controller-{index}"))
        p_ctrl_facts = (1 / len(controller_pool)) ** len(controller)
        board = C_peer + controller
        peer_counts = _counts(C_peer, weights)
        controller_counts = _counts(controller, weights)
        total_counts = _counts(board, weights)
        truth_share = game.truth_share(agents)
        round_row = {"round": day, "truth_share": truth_share,
                     "target_share": game.target_share(agents), "vote_entropy_bits": binary_entropy(truth_share),
                     "controller_U": U, "controller_effective_U": int(U and p.budget > 0),
                     "controller_p_act": p_act, "controller_observed_target_share": observed,
                     "controller_sensed_messages": q_sensor, "controller_sensor_target_count": Y,
                     "peer_board_target_count": peer_target_count,
                     "controller_target": p.controller_target, "controller_messages_next_board": len(controller),
                     "controller_requested_budget": p.budget, "budget_used": len(controller),
                     "fact_weights_json": _json(weights.tolist()),
                     "agent_states_before_persistence_json": _json(before_persistence),
                     "persistence_events_json": _json(persistence),
                     "agent_states_after_persistence_json": _json(_agents_from_persistence(persistence, agents, weights)),
                     "agent_states_json": _json(_agents(agents, weights)),
                     "occupancy_json": _json(_occupancy(agents, weights)),
                     "vote_share_by_epistemic_class_json": _json(_vote_by_class(agents, weights)),
                     "front_page_json": _json([_message(m, weights) for m in B_current]),
                     "peer_board_json": _json([_message(m, weights) for m in C_peer]),
                     "controller_board_json": _json([_message(m, weights) for m in controller]),
                     "next_board_json": _json([_message(m, weights) for m in board]),
                     "B_peer_counts_json": _json(peer_counts), "B_controller_counts_json": _json(controller_counts),
                     "B_total_counts_json": _json(total_counts),
                     "B_total_fractions_json": _json({k: v / len(board) for k, v in total_counts.items()}),
                     "R_bar_json": _json({k: predictive_categories.get(k, 0.0) / p.N for k in peer_counts}),
                     "C_peer_fractions_json": _json({k: v / p.N for k, v in peer_counts.items()}),
                     "sensor_sample_ids_json": _json([m.message_id for m in sensed]),
                     "controller_fact_pool_json": _json(controller_pool),
                     "controller_selected_fact_ids_json": _json([m.fact_id for m in controller]),
                     "controller_message_ids_json": _json([m.message_id for m in controller]),
                     "rng_state_next_day_json": _json(rng.bit_generator.state),
                     "persistence_plus_lost": plus_lost, "persistence_minus_lost": minus_lost,
                     "p_persistence_event": persistence_probability, "log_p_persistence_event": persistence_log,
                     "p_sensor_outcome": p_sensor_y, "log_p_sensor_outcome": _probability_log(p_sensor_y),
                     "p_sensor_sample_ids": p_sensor, "log_p_sensor_sample_ids": _probability_log(p_sensor),
                     "p_action_given_sensor": p_action, "log_p_action_given_sensor": _probability_log(p_action),
                     "p_controller_fact_selection": p_ctrl_facts,
                     "log_p_controller_fact_selection": _probability_log(p_ctrl_facts),
                     "sensor_absolute_error": abs(observed - peer_target_share),
                     "sensor_squared_error": (observed - peer_target_share) ** 2,
                     "sensor_sample_fraction": q_sensor / p.N,
                     **_coverage(agents, weights, p.controller_target)}
        rounds.append(round_row)
    return EpisodeResult(seed=seed, params={**asdict(p), "budget": p.budget},
                         fact_weights=weights.tolist(), rounds=rounds, micro=micro)


def _agents_from_persistence(persistence, agents, weights):
    return [{"agent_id": item["agent_id"], "vote": item["class_after"][2],
             "active_fact_ids": item["after_fact_ids"], "class": item["class_after"]}
            for item in persistence]


def replay_next_day(agent_records: list[dict], board_records: list[dict], weights: list[int],
                    params, rng_state: dict, day: int) -> tuple[list[dict], list[dict]]:
    """Clone a v3 boundary and replay night + daytime with an explicit RNG state.

    The factual-action branch must reproduce the saved next boundary exactly.
    Forced-action branches can change only the front-page controller messages.
    """
    from .state import AgentState
    from .game import SyntheticGame
    agents = [AgentState(vote=int(a["vote"]), active_facts=set(a["active_fact_ids"])) for a in agent_records]
    board = [Message(int(m["author"]), int(m["vote"]), m["fact_id"],
                     m["source"] == "controller", m["message_id"]) for m in board_records]
    weights_array = np.asarray(weights, dtype=int)
    rng = np.random.default_rng()
    rng.bit_generator.state = rng_state
    game = SyntheticGame(params)
    for agent in agents:
        agent.active_facts = {f for f in sorted(agent.active_facts) if rng.random() < params.rho}
    peer = []
    for slot in range(params.N):
        focal = int(rng.integers(params.N))
        agent = agents[focal]
        sampled, _ = _sample_messages(board, min(params.q, len(board)), rng)
        agent.active_facts.update(m.fact_id for m in sampled if m.fact_id is not None)
        p_truth = sigmoid(params.beta_evidence * game.evidence_signal(agent, weights_array)
                          + params.beta_social * game.social_signal(sampled))
        agent.vote = 1 if rng.random() < p_truth else -1
        eligible = sorted(f for f in agent.active_facts if weights_array[f] == agent.vote)
        fact = int(rng.choice(eligible)) if eligible else None
        peer.append(Message(focal, agent.vote, fact, False, f"r{day}-peer-{slot}"))
    return _agents(agents, weights_array), [_message(m, weights_array) for m in peer]


def _vote_by_class(agents, weights):
    groups = defaultdict(list)
    for agent in agents:
        r, s, vote = _class(agent, weights)
        groups[(r,s)].append(vote)
    return [{"r":r,"s":s,"n":len(votes),"truth_vote_share":sum(v==1 for v in votes)/len(votes)}
            for (r,s),votes in sorted(groups.items())]
