"""Synthetic dynamics only; no estimators or output handling."""
from __future__ import annotations
from dataclasses import asdict
from typing import Optional
import math
import numpy as np
from .state import SimulationParameters, Message, AgentState, EpisodeResult

def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def make_fact_weights(
    F: int,
    truth_fact_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Every fact has signed evidence +/-1.
    The complete fact set is guaranteed to favor truth.
    """
    n_truth = max(F // 2 + 1, int(round(F * truth_fact_fraction)))
    n_truth = min(n_truth, F)
    w = np.array([1] * n_truth + [-1] * (F - n_truth), dtype=int)
    rng.shuffle(w)
    return w


class SyntheticGame:
    def __init__(self, params: SimulationParameters):
        self.p = params

    def initialize(
        self,
        rng: np.random.Generator,
        weights: np.ndarray,
    ) -> list[AgentState]:
        agents = [
            AgentState(vote=1 if rng.random() < 0.5 else -1)
            for _ in range(self.p.N)
        ]

        r = min(self.p.initial_fact_redundancy, self.p.N)
        for fact_id in range(self.p.F):
            holders = rng.choice(self.p.N, size=r, replace=False)
            for i in holders:
                agents[int(i)].active_facts.add(int(fact_id))

        # Initial vote depends on initially available evidence.
        for agent in agents:
            agent.vote = self.draw_vote(
                agent=agent,
                sampled_messages=[],
                weights=weights,
                rng=rng,
            )
        return agents

    def initial_board(
        self,
        agents: list[AgentState],
        rng: np.random.Generator,
    ) -> list[Message]:
        board = []
        for i, a in enumerate(agents):
            fact_id = (
                int(rng.choice(tuple(a.active_facts)))
                if a.active_facts else None
            )
            board.append(Message(i, a.vote, fact_id, False))
        return board

    def evidence_signal(
        self,
        agent: AgentState,
        weights: np.ndarray,
    ) -> float:
        if not agent.active_facts:
            return 0.0

        # Normalized to [-1,+1], so beta_evidence has an interpretable scale.
        raw = sum(weights[f] for f in agent.active_facts)
        return float(raw / len(agent.active_facts))

    @staticmethod
    def social_signal(sampled_messages: list[Message]) -> float:
        if not sampled_messages:
            return 0.0
        return float(np.mean([m.vote for m in sampled_messages]))

    def draw_vote(
        self,
        agent: AgentState,
        sampled_messages: list[Message],
        weights: np.ndarray,
        rng: np.random.Generator,
    ) -> int:
        e = self.evidence_signal(agent, weights)
        s = self.social_signal(sampled_messages)

        logit = self.p.beta_evidence * e + self.p.beta_social * s
        p_truth = sigmoid(logit)
        return 1 if rng.random() < p_truth else -1

    def epistemic_summary(
        self,
        agents: list[AgentState],
        weights: np.ndarray,
    ) -> dict:
        fact_counts = np.array([len(a.active_facts) for a in agents], dtype=float)
        union = set().union(*(a.active_facts for a in agents))

        balances = np.array([
            self.evidence_signal(a, weights) for a in agents
        ], dtype=float)

        # Candidate coarse epistemic coordinates kappa.
        return {
            "kappa_mean_coverage": float(fact_counts.mean() / self.p.F),
            "kappa_full_agent_fraction": float(np.mean(fact_counts == self.p.F)),
            "kappa_population_coverage": float(len(union) / self.p.F),
            "kappa_mean_evidence_balance": float(balances.mean()),
            "kappa_evidence_sd": float(balances.std()),
        }

    @staticmethod
    def truth_share(agents: list[AgentState]) -> float:
        return float(np.mean([a.vote == 1 for a in agents]))

    def target_share(self, agents: list[AgentState]) -> float:
        return float(np.mean([
            a.vote == self.p.controller_target for a in agents
        ]))

    def controller_sense(
        self,
        closed_board: list[Message],
        rng: np.random.Generator,
    ) -> tuple[float, int]:
        if not closed_board:
            return 0.5, 0

        n_sensed = max(
            1,
            int(round(self.p.sensing_fraction * len(closed_board))),
        )
        n_sensed = min(n_sensed, len(closed_board))
        idx = rng.choice(len(closed_board), size=n_sensed, replace=False)
        sensed = [closed_board[int(j)] for j in idx]

        observed_target_share = float(np.mean([
            m.vote == self.p.controller_target for m in sensed
        ]))
        return observed_target_share, n_sensed

    def controller_policy(
        self,
        observed_target_share: float,
        rng: np.random.Generator,
    ) -> tuple[int, float]:
        """
        Feedback policy:
        intervene more often when observed support for controller target is low.
        """
        z = self.p.policy_beta * (
            self.p.policy_threshold - observed_target_share
        )
        p_act = sigmoid(z)
        U = int(rng.random() < p_act)
        return U, p_act

    def run_episode(self, seed: int) -> EpisodeResult:
        rng = np.random.default_rng(seed)
        weights = make_fact_weights(
            self.p.F,
            self.p.truth_fact_fraction,
            rng,
        )

        agents = self.initialize(rng, weights)
        board = self.initial_board(agents, rng)

        round_rows = []
        micro_rows = []

        def record_round(t, U, p_act, sensed_share, n_sensed, n_ctrl):
            ts = self.truth_share(agents)
            row = {
                "round": t,
                "truth_share": ts,
                "target_share": self.target_share(agents),
                "vote_entropy_bits": binary_entropy(ts),
                "controller_U": U,
                "controller_effective_U": int(U and self.p.budget > 0),
                "controller_p_act": p_act,
                "controller_observed_target_share": sensed_share,
                "controller_sensed_messages": n_sensed,
                "controller_messages_next_board": n_ctrl,
                "budget_used": n_ctrl,
                **self.epistemic_summary(agents, weights),
            }
            round_rows.append(row)

        record_round(0, 0, 0.0, np.nan, 0, 0)

        for t in range(1, self.p.rounds + 1):
            previous_board = list(board)
            closed_board: list[Message] = []

            # N asynchronous microscopic updates per round.
            for slot in range(self.p.N):
                focal = int(rng.integers(0, self.p.N))
                agent = agents[focal]

                vote_before = agent.vote
                facts_before = set(agent.active_facts)

                # Persistence: each currently active fact survives independently.
                retained = {
                    f for f in agent.active_facts
                    if rng.random() < self.p.rho
                }

                q_eff = min(self.p.q, len(previous_board))
                sampled = []
                if q_eff:
                    idx = rng.choice(
                        len(previous_board),
                        size=q_eff,
                        replace=False,
                    )
                    sampled = [previous_board[int(j)] for j in idx]

                acquired = {
                    m.fact_id for m in sampled
                    if m.fact_id is not None
                }
                agent.active_facts = retained | acquired

                e_signal = self.evidence_signal(agent, weights)
                s_signal = self.social_signal(sampled)

                agent.vote = self.draw_vote(
                    agent,
                    sampled,
                    weights,
                    rng,
                )

                # One public post from the focal update.
                fact_id = (
                    int(rng.choice(tuple(agent.active_facts)))
                    if agent.active_facts else None
                )
                closed_board.append(
                    Message(
                        author=focal,
                        vote=agent.vote,
                        fact_id=fact_id,
                        is_controller=False,
                    )
                )

                if self.p.save_micro:
                    micro_rows.append({
                        "round": t,
                        "slot": slot,
                        "focal": focal,
                        "vote_before": vote_before,
                        "vote_after": agent.vote,
                        "facts_before": len(facts_before),
                        "facts_after": len(agent.active_facts),
                        "facts_acquired": len(agent.active_facts - facts_before),
                        "facts_lost": len(facts_before - agent.active_facts),
                        "q_effective": q_eff,
                        "sampled_controller_messages": int(sum(
                            m.is_controller for m in sampled
                        )),
                        "evidence_signal": e_signal,
                        "social_signal": s_signal,
                    })

            # Day closes. Controller senses closed board.
            observed_target_share, n_sensed = self.controller_sense(
                closed_board,
                rng,
            )
            U, p_act = self.controller_policy(
                observed_target_share,
                rng,
            )

            # Recommendation-only control; controller adds no facts.
            controller_posts = []
            if U and self.p.budget > 0:
                controller_posts = [
                    Message(
                        author=self.p.N,
                        vote=self.p.controller_target,
                        fact_id=None,
                        is_controller=True,
                    )
                    for _ in range(self.p.budget)
                ]

            # Front page for next day.
            board = closed_board + controller_posts

            record_round(
                t,
                U,
                p_act,
                observed_target_share,
                n_sensed,
                len(controller_posts),
            )

        return EpisodeResult(
            seed=seed,
            params={
                **asdict(self.p),
                "budget": self.p.budget,
            },
            fact_weights=weights.tolist(),
            rounds=round_rows,
            micro=micro_rows,
        )


