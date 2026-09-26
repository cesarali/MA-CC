"""Measurements over saved round trajectories."""
from __future__ import annotations

import numpy as np
import pandas as pd

STATISTICS = ("I_X_Y_bits", "Tpi_I_U_Xnext_given_X_bits", "Tpi_I_U_Xnext_given_X_K_bits")


def entropy_discrete(values) -> float:
    arr = np.asarray(values)
    if len(arr) == 0:
        return float("nan")
    _, counts = np.unique(arr, return_counts=True, axis=0 if arr.ndim > 1 else None)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def mutual_information_discrete(x, y) -> float:
    return entropy_discrete(x) + entropy_discrete(y) - entropy_discrete(np.column_stack([x, y]))


def conditional_mutual_information_discrete(x, y, z) -> float:
    z = np.asarray(z)
    if z.ndim == 1:
        z = z[:, None]
    return (entropy_discrete(np.column_stack([x, z]))
            + entropy_discrete(np.column_stack([y, z]))
            - entropy_discrete(z)
            - entropy_discrete(np.column_stack([x, y, z])))


def bin01(values, bins: int) -> np.ndarray:
    return np.digitize(np.clip(np.asarray(values, dtype=float), 0, 1), np.linspace(0, 1, bins + 1)[1:-1]).astype(int)


def transitions(rounds: pd.DataFrame) -> pd.DataFrame:
    d = rounds.sort_values(["cell_id", "seed", "round"]).copy()
    d["next_target_share"] = d.groupby(["cell_id", "seed"], sort=False)["target_share"].shift(-1)
    return d[(d["round"] > 0) & d["next_target_share"].notna() &
             d["controller_observed_target_share"].notna()].copy()


def encoded(d: pd.DataFrame, bins: int, coordinate: str) -> dict[str, np.ndarray]:
    x = bin01(d["target_share"], bins)
    xp = bin01(d["next_target_share"], bins)
    y = bin01(d["controller_observed_target_share"], bins)
    u = d["controller_effective_U"].to_numpy(dtype=int)
    k1 = bin01(d["kappa_mean_coverage"], bins)
    k2 = bin01(d["kappa_population_coverage"], bins)
    k = {"mean_coverage": k1[:, None], "population_coverage": k2[:, None],
         "both": np.column_stack([k1, k2])}[coordinate]
    return {"x": x, "xp": xp, "y": y, "u": u, "k": k, "z": np.column_stack([x, k])}


def statistics(d: pd.DataFrame, bins: int, coordinate: str) -> dict[str, float]:
    if d.empty:
        return {name: float("nan") for name in STATISTICS}
    v = encoded(d, bins, coordinate)
    return {"I_X_Y_bits": mutual_information_discrete(v["x"], v["y"]),
            "Tpi_I_U_Xnext_given_X_bits": conditional_mutual_information_discrete(v["u"], v["xp"], v["x"]),
            "Tpi_I_U_Xnext_given_X_K_bits": conditional_mutual_information_discrete(v["u"], v["xp"], v["z"])}


def information_summary(rounds: pd.DataFrame, bins: int, coordinate: str) -> pd.DataFrame:
    return pd.DataFrame([{"cell_id": cell, "n_transitions": len(d), "epistemic_coordinate": coordinate,
                          **statistics(d, bins, coordinate)}
                         for cell, d in transitions(rounds).groupby("cell_id")])


def susceptibility_strata(d: pd.DataFrame, bins: int, coordinate: str) -> pd.DataFrame:
    if d.empty:
        return pd.DataFrame()
    v = encoded(d, bins, coordinate)
    frame = pd.DataFrame({f"z{i}": v["z"][:, i] for i in range(v["z"].shape[1])})
    frame["u"] = v["u"]
    frame["next"] = d["next_target_share"].to_numpy()
    rows = []
    keys = [c for c in frame if c.startswith("z")]
    for state, group in frame.groupby(keys):
        g0 = group.loc[group.u == 0, "next"]
        g1 = group.loc[group.u == 1, "next"]
        if len(g0) and len(g1):
            if not isinstance(state, tuple):
                state = (state,)
            rows.append({**dict(zip(keys, state)), "chi_empirical": float(g1.mean() - g0.mean()),
                         "n_U0": len(g0), "n_U1": len(g1), "n_supported": len(group)})
    return pd.DataFrame(rows)


def susceptibility(d: pd.DataFrame, bins: int, coordinate: str) -> tuple[float, int]:
    strata = susceptibility_strata(d, bins, coordinate)
    if strata.empty:
        return float("nan"), 0
    return float(np.average(strata.chi_empirical, weights=strata.n_supported)), int(strata.n_supported.sum())


def susceptibility_summary(rounds: pd.DataFrame, bins: int, coordinate: str) -> pd.DataFrame:
    rows = []
    for cell, d in transitions(rounds).groupby("cell_id"):
        value, supported = susceptibility(d, bins, coordinate)
        rows.append({"cell_id": cell, "chi_empirical": value, "n_supported": supported,
                     "epistemic_coordinate": coordinate})
    return pd.DataFrame(rows)


def final_summary(rounds: pd.DataFrame) -> pd.DataFrame:
    final = rounds.loc[rounds.groupby(["cell_id", "seed"])["round"].idxmax()]
    return final.groupby("cell_id", as_index=False).agg(
        beta_evidence=("beta_evidence", "first"), beta_social=("beta_social", "first"),
        beta_regime=("beta_regime", "first"), rho=("rho", "first"),
        budget_fraction=("budget_fraction", "first"), budget=("budget", "first"),
        n_episodes=("seed", "nunique"), final_truth_share=("truth_share", "mean"),
        final_target_share=("target_share", "mean"), final_vote_entropy_bits=("vote_entropy_bits", "mean"),
        final_mean_coverage=("kappa_mean_coverage", "mean"),
        final_population_coverage=("kappa_population_coverage", "mean"))
