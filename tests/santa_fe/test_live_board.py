import json
from dataclasses import replace

import numpy as np
import pandas as pd

from santa_fe.game import SyntheticGame
from santa_fe.live_board import overload_factor
from santa_fe.live_board_metrics import extra_information
from santa_fe.llm_parallel import adapt_live_board_trajectories
from santa_fe.state import SimulationParameters
from mas_cc.games.hidden_bench.imitation_round_feedback.analysis import round_information_analysis


def parameters(**changes):
    base = SimulationParameters(
        model_version="santa_fe_live_board_v4", board_clock="live_board",
        persistence_clock="round_boundary", peer_posting_mode="vote_aligned_fact",
        controller_message_mode="target_aligned_fact",
        controller_fact_selection="uniform_with_replacement",
        N=24, F=12, F_plus=8, initial_fact_redundancy=2,
        rounds=3, q=24, sensing_fraction=1.0, budget_fraction=1.0,
        controller_target=1, rho=0.4, beta_evidence=1.0, beta_social=1.3,
        policy_beta=4.0, policy_threshold=0.5, save_micro=True,
    )
    return replace(base, **changes)


def test_live_board_mechanics_and_records():
    p = parameters()
    episode = SyntheticGame(p).run_episode(12345)
    assert len(episode.rounds) == 4
    assert len(episode.micro) == 3 * 24
    initial = json.loads(episode.rounds[0]["initial_agent_states_json"])
    facts = [agent["active_fact_ids"][0] for agent in initial]
    assert len(facts) == 24 and sorted(np.bincount(facts)) == [2] * 12
    for t, row in enumerate(episode.rounds[:-1]):
        slots = episode.micro[t * 24:(t + 1) * 24]
        assert row["round"] == t and len(json.loads(row["sensor_agent_ids_json"])) == 24
        assert len(set(json.loads(row["sensor_agent_ids_json"]))) == 24
        assert row["budget_used"] == p.budget * row["controller_U"]
        assert len(json.loads(row["live_board_end_json"])) == row["budget_used"] + 24
        assert all(m["expires_before_round"] == t for m in json.loads(row["expired_board_json"]))
        assert row["next_target_count"] == episode.rounds[t + 1]["target_count"]
        for slot in slots:
            ids = json.loads(slot["sampled_message_ids_json"])
            authors = json.loads(slot["sampled_message_authors_json"])
            assert len(ids) == len(set(ids)) == slot["q_effective"]
            assert all(author != slot["focal"] for author in authors)
            assert all(not i.startswith(f"r{t}-peer-") or
                       int(i.rsplit("-", 1)[1]) < slot["slot"] for i in ids)
            assert set(json.loads(slot["facts_after_json"])) <= set(json.loads(slot["history_after_json"]))
            assert set(json.loads(slot["history_before_json"])) <= set(json.loads(slot["history_after_json"]))
            assert slot["overload_factor"] == overload_factor(
                slot["active_load"], p.overload_threshold, p.overload_alpha)
            assert np.isclose(slot["vote_logit"], slot["overload_factor"] *
                              (p.beta_evidence * slot["evidence_signal"] +
                               p.beta_social * slot["social_signal"]))
    assert overload_factor(7, 7, .2) == 1
    assert np.isclose(overload_factor(12, 7, .2), .5)


def test_zero_budget_target_arms_share_physical_path():
    plus = SyntheticGame(parameters(budget_fraction=0, controller_target=1)).run_episode(678)
    minus = SyntheticGame(parameters(budget_fraction=0, controller_target=-1)).run_episode(678)
    for a, b in zip(plus.rounds, minus.rounds):
        assert a["agent_states_json"] == b["agent_states_json"]
        assert a["target_share"] + b["target_share"] == 1
    assert all(row["budget_used"] == 0 for row in plus.rounds)
    assert all(row["budget_used"] == 0 for row in minus.rounds)


def test_overload_applies_to_social_only_field():
    assert overload_factor(8, 7, .2) * 2 < 2


def test_shared_information_adapter_uses_pre_action_lag_and_virtual_gate():
    p=parameters(rounds=3,budget_fraction=0,controller_target=-1)
    rows=[]
    for seed in range(12):
        episode=SyntheticGame(p).run_episode(seed)
        rows += [{"cell_id":0,"seed":seed,"N":p.N,"model_version":p.model_version,**row}
                 for row in episode.rounds]
    frame=pd.DataFrame(rows)
    events=adapt_live_board_trajectories(frame)
    assert len(events)==12*3
    for event in events:
        row=frame.loc[(frame.seed==int(event.episode_id)) &
                      (frame["round"]==event.round_index)].iloc[0]
        assert event.target_before==row.target_count
        assert event.target_after==row.next_target_count
        assert event.U_k is not None
    shared,_=round_information_analysis(events,statistics=["round_target_actuation_cmi"],
                                        bootstrap_resamples=0,null_permutations=0)
    check=extra_information(frame)
    value=check.loc[(check.metric=="I_U_Knext_given_K_E_entropy_difference") &
                    (check.conditioning=="none"),"estimate_bits"].iloc[0]
    assert np.isclose(shared[0]["estimate"],value)
