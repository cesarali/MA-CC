"""Write the full efficiency report for a reviewing agent: 2026-10-12_report.md (next to this file).

Every number in its tables is read from results/efficiency_estimates/<date>/ (efficiency.csv,
uncertainty.csv, efficiency_phase.csv, downsample.csv, ranking.csv); the prose is fixed below.

Usage, from the repository root (after efficiency.py, downsample.py, phase.py):
    .venv/bin/python analysis/task003_llm_free/efficiency_estimates/make_md_report.py [--date 2026-10-12]
"""
from __future__ import annotations
import argparse, pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
RES = HERE.parent / "results" / "efficiency_estimates"
SW, AV, AB = "swarm_beliefs_k3", "agent_votes", "agent_beliefs"
SPEC = {"sim1": "b3", "sim2": "lc1_gateon"}


def f(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{nd}f}"


def md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(out)


def setting_label(sim, s):
    if sim == "sim1":
        return f"b = {s[1:]}"
    lc, g = s.split("_")
    return f"λc = {lc[2:]}, gate {g[4:]}"


def order(sim, s):
    if sim == "sim1":
        return (0, int(s[1:]))
    lc, g = s.split("_")
    return (0 if g == "gateon" else 1, float(lc[2:]))


def tables(d: pathlib.Path) -> dict[str, str]:
    e = pd.read_csv(d / "efficiency.csv")
    u = pd.read_csv(d / "uncertainty.csv")
    ph = pd.read_csv(d / "efficiency_phase.csv")
    ds = pd.read_csv(d / "downsample.csv")
    rk = pd.read_csv(d / "ranking.csv")
    T = {}
    e30 = e[e.t == 30]
    u30 = u[u.t == 30]
    blocks = []
    for sim in ("sim1", "sim2"):
        for setup in ("task003_symmetric", "task003_nosolution"):
            for rho in (0.75, 1.0):
                x = e30[(e30.sim == sim) & (e30.setup == setup) & (e30.rho == rho)]
                if x.empty:
                    continue
                rows = []
                for s in sorted(x.setting.unique(), key=lambda s: order(sim, s)):
                    a = x[(x.setting == s) & (x.resolution == AV)].iloc[0]
                    w = x[(x.setting == s) & (x.resolution == SW)].iloc[0]
                    eta = a.eta if (np.isfinite(a.C) and a.C >= 0.01) else np.nan
                    eta3 = (2 / 3) * a.I / a.I3 if a.I3 > 1e-6 else np.nan
                    pq = ph[(ph.sim == sim) & (ph.setup == setup) & (ph.rho == rho) & (ph.qc == 12)
                            & (ph.resolution == AV)]
                    xv = float(s[1:]) if sim == "sim1" else float(s.split("_")[0][2:])
                    gate = "on" if sim == "sim1" else s.split("_")[1][4:]
                    pr = pq[(pq.x == xv) & (pq.gate == gate)]
                    pp = pr.S_per_fact_plus.iloc[0] if len(pr) else np.nan
                    pm = pr.S_per_fact_minus.iloc[0] if len(pr) else np.nan
                    uu = u30[(u30.sim == sim) & (u30.setup == setup) & (u30.rho == rho) & (u30.setting == s)]
                    gu = lambda arm, m: uu[(uu.arm == arm) & (uu.measure == m)]["diff"].iloc[0] if len(uu[(uu.arm == arm) & (uu.measure == m)]) else np.nan
                    rows.append({"setting": setting_label(sim, s), "I swarm": f(w.I), "I agent": f(a.I),
                                 "η": f(eta), "η_JS swarm": f(w.eta_JS), "η₃": f(eta3),
                                 "π A0": f(a.progress_plus), "π A2": f(a.progress_minus),
                                 "φ A0": f(a.focus_plus), "φ A2": f(a.focus_minus),
                                 "bits/fact A0": f(pp), "bits/fact A2": f(pm),
                                 "ΔH̄ A0": f(gu("truth", "Hbar")), "ΔH̄ A2": f(gu("false", "Hbar")),
                                 "ΔJSD A0": f(gu("truth", "JSD")), "ΔJSD A2": f(gu("false", "JSD"))})
                name = {"sim1": "Simulation 1", "sim2": "Simulation 2"}[sim]
                blocks.append(f"**{name}, {setup[8:]}, ρ = {rho:g}** (qc = 12)\n\n" + md(pd.DataFrame(rows)))
    T["settings"] = "\n\n".join(blocks)

    blocks = []
    for sim in ("sim1", "sim2"):
        for setup in ("task003_symmetric", "task003_nosolution"):
            x = e[(e.sim == sim) & (e.setup == setup) & (e.rho == 0.75) & (e.setting == SPEC[sim])]
            uu = u[(u.sim == sim) & (u.setup == setup) & (u.rho == 0.75) & (u.setting == SPEC[sim])]
            times = [1, 3, 5, 10, 20, 30] + ([35, 40] if sim == "sim2" else [])
            rows = []
            for t in times:
                a = x[(x.t == t) & (x.resolution == AV)]
                w = x[(x.t == t) & (x.resolution == SW)]
                if a.empty:
                    continue
                a, w = a.iloc[0], w.iloc[0]
                g = lambda arm, m: uu[(uu.t == t) & (uu.arm == arm) & (uu.measure == m)]["diff"]
                rows.append({"t": t, "I swarm": f"{f(w.I)} [{f(w.I_lo)}, {f(w.I_hi)}]",
                             "I agent": f(a.I), "π A0": f"{f(a.progress_plus)} [{f(a.progress_plus_lo)}, {f(a.progress_plus_hi)}]",
                             "π A2": f"{f(a.progress_minus)} [{f(a.progress_minus_lo)}, {f(a.progress_minus_hi)}]",
                             "φ A0": f(a.focus_plus), "φ A2": f(a.focus_minus),
                             "η": f(a.eta if a.C >= 0.01 else np.nan), "η_JS": f(w.eta_JS),
                             "ΔH̄ A0": f(g("truth", "Hbar").iloc[0]) if len(g("truth", "Hbar")) else "–",
                             "ΔJSD A2": f(g("false", "JSD").iloc[0]) if len(g("false", "JSD")) else "–"})
            name = {"sim1": "Simulation 1, ρ 0.75, qc 12, b 3 (t = day)", "sim2": "Simulation 2, ρ 0.75, qc 12, λc 1, gate on"}[sim]
            blocks.append(f"**{name}, {setup[8:]}** (95% intervals in brackets)\n\n" + md(pd.DataFrame(rows)))
    T["time"] = "\n\n".join(blocks)

    rows = []
    for (sim, setup, rho, gate), g in ph.groupby(["sim", "setup", "rho", "gate"]):
        a = g[g.resolution == AV]
        b = g[g.resolution == AB]
        sw = g[g.resolution == SW]
        rng = lambda s: "–" if not np.isfinite(s.replace(np.inf, np.nan)).any() else \
            f"{np.nanmin(s.replace(np.inf, np.nan)):.2f} … {np.nanmax(s.replace(np.inf, np.nan)):.2f}"
        rng3 = lambda s: "–" if not np.isfinite(s.replace(np.inf, np.nan)).any() else \
            f"{np.nanmin(s.replace(np.inf, np.nan)):.3f} … {np.nanmax(s.replace(np.inf, np.nan)):.3f}"
        etab = b.eta.where(b.C >= 0.01)
        rows.append({"simulation": sim, "setup": setup[8:], "ρ": rho, "gate": gate,
                     "η (beliefs)": rng(etab), "η_JS swarm": rng(sw.eta_JS), "π A0": rng(a.progress_plus),
                     "π A2": rng(a.progress_minus), "φ A0": rng(a.focus_plus), "φ A2": rng(a.focus_minus),
                     "bits/fact A0": rng3(a.S_per_fact_plus), "bits/fact A2": rng3(a.S_per_fact_minus)})
    T["phase"] = md(pd.DataFrame(rows))

    x = ds[(ds.t == 30) & ~((ds.setup == "task003_symmetric") & (ds.rho == 1.0))]
    sel = [(SW, "I", "I, swarm (3 bins)"), (AV, "I", "I, one agent (votes)"), (AV, "eta", "η, one agent (votes)"),
           (AB, "eta", "η, one agent (beliefs)"), (SW, "eta_JS", "η_JS, swarm"), (AV, "progress_plus", "π A0 (votes)"),
           (AV, "progress_minus", "π A2 (votes)"), (AV, "focus_plus", "φ A0 (votes)"), (AV, "focus_minus", "φ A2 (votes)"),
           (SW, "C", "C, swarm (KL)")]
    rows = []
    for res, q, lab in sel:
        y = x[(x.resolution == res) & (x.quantity == q)]
        r = {"quantity": lab, "typical value": f(y[np.isfinite(y.full)].full.median())}
        for n in (50, 100, 300):
            z = y[y.n_sub == n]
            r[f"RMSE {n}"] = f(z.rmse.median())
            r[f"undefined {n}"] = f"{z.share_not_finite.mean():.0%}"
        rows.append(r)
    for q, lab in (("dHbar", "ΔH̄"), ("dJSD", "ΔJSD")):
        y = x[x.quantity == q]
        r = {"quantity": lab, "typical value": f(y.full.abs().median())}
        for n in (50, 100, 300):
            r[f"RMSE {n}"] = f(y[y.n_sub == n].rmse.median())
            r[f"undefined {n}"] = "0%"
        rows.append(r)
    T["precision"] = md(pd.DataFrame(rows))

    r = rk[(rk.t == 30) & ~((rk.setup == "task003_symmetric") & (rk.rho == 1.0)) & (rk.n_sub == 100)]
    r = r[r.quantity.isin(["I", "eta", "eta_JS", "progress_plus", "progress_minus"]) & r.resolution.isin([AV, SW])]
    r = r.assign(size=pd.cut(r.full_diff.abs(), [0, 0.01, 0.02, 0.05, 10], labels=["< 0.01", "0.01–0.02", "0.02–0.05", "> 0.05"]))
    t = r.groupby(["resolution", "quantity", "size"], observed=True).share_same_order.mean().unstack("size")
    t = t.map(lambda v: f"{v:.0%}" if np.isfinite(v) else "–").reset_index()
    T["ranking"] = md(t)
    return T


TEMPLATE = """# Control efficiency — full report for review

**12 October 2026.** Measures of how efficiently a controller steers the swarm, for Simulation 1
(days and nights) and Simulation 2 (continuous time), with their precision at 50, 100 and 300
paths. Written for a reviewing agent: every definition, every estimator and every result table is
here, with pointers to the code and data. Tables are generated by `make_md_report.py` from the result
files; do not edit them by hand.

## 0. For the reviewer

**Your task:** check the definitions (§2), their implementation (`efficiency.py`: `kl`,
`kl_binary`, `per_controller`, `measures`, `resolutions`; `phase.py`; `downsample.py`), the numbers
(§3–§4, recompute some from `states.parquet`), and whether the findings (§5) follow from the tables.
Work read-only. Report each problem with file, line or table, and why.

**Files.** Code: `analysis/task003_llm_free/efficiency_estimates/` (README.md lists each script). Data
(not in git): `analysis/task003_llm_free/results/efficiency_estimates/2026-10-12/` — `states.parquet`
(one row per simulation, setting, arm, episode, time), `efficiency.csv` (all measures, all times,
95% bootstrap intervals), `uncertainty.csv`, `efficiency_phase.csv` (whole control plane, t = 30),
`downsample.csv`, `ranking.csv`, `null_I.csv`. Source runs: Simulation 1 `results/simulation_1/2026-10-07_sim1_pm`;
Simulation 2 `results/simulation_2/2026-10-09_sim2_grid` and `2026-10-12_sim2_main_rho1_full`
(identical simulations, full tables). Reports with figures: the four `2026-10-12_report*` PDFs in
`simulation_1/analysis/reports/` and `simulation_2/analysis/reports/`, section "How efficiently does
the controller steer?".

**Rerun:** from the repository root, in order: `build_states.py`, `efficiency.py`, `downsample.py`,
`phase.py`, `make_report_figures.py`, `make_md_report.py` (all in `efficiency_estimates/`, option
`--date 2026-10-12`).

**What deserves the most scrutiny:** (1) the per-controller measures (§2.4): are the signs and the
undefined rules right; (2) the reference distribution (§2.6); (3) whether the swarm-level measures
lose information by dropping A1 (§2.6); (4) the claims in §5 against §3.

## 1. Data and settings

- Probability matching, q = 6 (posts an agent reads). 1,000 paired episodes per arm (silent, target
  A0, target A2; same random draws).
- Simulation 1 (`sim1_pm`): qc ∈ {3, 6, 12, 24} (posts the controller reads), b ∈ {1, 2, 3, 6, 9}
  (facts per night), ρ ∈ {0.75, 1}; measured at the end of each day 1–30.
- Simulation 2 (grid): qc ∈ {3, 6, 12, 24}, controller rate λc ∈ {0.5, 1, 2, 4, 8}, vote gate on/off,
  proof stop on, ρ ∈ {0.75, 1}; measured at t = 0, 1, …, 40 (controller active for t < 30).
- Agents' uncertainty needs per-agent beliefs: Simulation 1 every cell; Simulation 2 the qc = 12 cells.

## 2. Definitions

### 2.1 Distributions

Arms a ∈ {0 (silent), + (target A0), − (target A2)}. At a fixed time t, each episode e contributes a
probability vector y_a(e) over the values x of the state X, and P̂_a = (1/n) Σ_e y_a(e):

- **one agent**: X = the vote of a randomly chosen agent ∈ {A0, A1, A2, not yet voted}; y_a(e) = the
  episode's vote shares (or, for "beliefs", its mean beliefs (m_A0, m_A1, m_A2));
- **swarm**: X = the bin of s = m_A0 − m_A2 (three equal bins on [−1, 1]); y_a(e) marks the bin.

These are distributions **at one time**, across episodes; nothing is pooled over time, and nothing is
a distribution over trajectories.

### 2.2 Common and directional parts

Any pair can be written P̂_± = P̂_0 + μ ± δ with **μ = ½(P̂_+ + P̂_−) − P̂_0** (common part, a vector)
and **δ = ½(P̂_+ − P̂_−)** (directional part, a vector). Q̂ = P̂_0 + μ is the 50/50 mixture of the two
controlled arms.

### 2.3 Readability and pair efficiency

With KL(p‖q) = Σ_x p(x) log₂[p(x)/q(x)]:

- **I** = ½ KL(P̂_+‖Q̂) + ½ KL(P̂_−‖Q̂) = ½ KL(Q̂+δ‖Q̂) + ½ KL(Q̂−δ‖Q̂): mutual information between
  the target Z (fair coin) and X. Depends on δ and on Q̂. At most 1 bit.
- **C** = ½ KL(P̂_+‖P̂_0) + ½ KL(P̂_−‖P̂_0): total change of the distribution relative to silence (a number).
- **R** = KL(Q̂‖P̂_0) = KL(P̂_0+μ‖P̂_0): depends only on μ (given P̂_0).
- **Identity: C = I + R** (split log(P̂_z/P̂_0) = log(P̂_z/Q̂) + log(Q̂/P̂_0) and average over z). Hence
  R = C − I, and **η = I/C = I/(I + R)** ∈ [0, 1].
- Small changes (KL(p+ε‖p) ≈ Σ ε²/(2 ln 2 · p)): I ≈ ‖δ‖²/(2 ln 2), R ≈ ‖μ‖²/(2 ln 2), with
  ‖v‖² = Σ_x v(x)²/P̂_0(x), so **η ≈ 1/(1 + ‖μ‖²/‖δ‖²)**. η depends on the ratio of common to
  directional change, not on its size: η = 1 for mirror-image changes (μ = 0) of any size; it falls
  with a common push (both controllers move X the same way) or a lopsided pair (same direction,
  opposite signs, sizes a ≠ b: ‖μ‖/‖δ‖ = |a − b|/(a + b)).
- **η_JS** = I / (I + JS(Q̂, P̂_0)): JS is the Jensen–Shannon divergence (½ KL to the midpoint each
  way), equal to the information X carries about *whether* a controller is present (50/50 prior).
  Bounded (≤ 1 bit), so finite where C is infinite.
- **η₃** = ⅔ I / I₃, with I₃ the information about W ∈ {+, −, 0} (uniform). Chain rule:
  I₃ = I(presence; X) + ⅔ I, where the presence information uses the prior 2/3 controlled, 1/3
  silent. η₃ is the share of the three-way information that is about direction. Bounded, always finite.
- Rule: η is reported only where C ≥ 0.01 bits.

### 2.4 Each controller against its own target

For target z (A0 for +, A2 for −), one agent, votes:

- **S_z** = KL(Bern(P̂_z(z)) ‖ Bern(P̂_0(z))): the least information any intervention must inject to
  move the target's share from its silent value to the observed one (the cheapest way raises the
  target and scales the other answers in proportion; that costs exactly this binary KL).
- **progress π_z** = S_z / log₂(1/P̂_0(z)): share of the way from silence to unanimity on z, in bits.
  Undefined when log₂(1/P̂_0(z)) < 0.01 bits (silence already at unanimity).
- **focus φ_z** = S_z / KL(P̂_z‖P̂_0): share of the controller's actual change needed for its gain; the
  rest reshuffled the other answers (A1 included). Undefined when KL(P̂_z‖P̂_0) < 0.005 bits or infinite.
- Both are **signed**: negative when P̂_z(z) < P̂_0(z) (the controller moved its target down).
- **price** = S_z / facts posted per episode (bits per fact; Simulation 2: one fact per message).

### 2.5 Agents' own uncertainty

Per episode: H̄ = mean over agents of the entropy of their beliefs (0 to log₂ 3 ≈ 1.58 bits);
JSD = H(mean belief) − H̄ (disagreement; H(m) = H̄ + JSD exactly). Reported as controlled minus
silent in the same episode (ΔH̄, ΔJSD), averaged over episodes.

### 2.6 Choices a reviewer should know about

- **Reference distribution.** C and R use the **silent** arm: "how far did control move the swarm
  from no control". The three-command version (η₃, reference = mixture of all three arms) is in the
  tables for comparison; η_JS is its 50/50 counterpart.
- **A1.** At one-agent level A1 is one of the values of X and enters every KL, focus and progress (in
  progress A1 is part of "not the target"). At swarm level, s = m_A0 − m_A2 is a projection: moves
  into or out of A1 that leave m_A0 − m_A2 unchanged are invisible there. A 2-D swarm state
  (m_A0, m_A2) would keep it but needs more paths.
- **Saturation.** With full memory, swarm-level I is close to 1 bit in nosolution even where effects
  are small (every episode ends on its controller's side): it measures distinguishability, not size.
- **Infinite KL.** Where the controlled swarm reaches a state (or an answer) the silent swarm never
  visits, KL against silence is infinite; such cells are reported as undefined, never smoothed.

### 2.7 Estimation

Point estimates from all 1,000 episodes. 95% intervals: 1,000 bootstrap resamples of episode
numbers, the same resample for all arms (paired). Precision: 200 random subsets of 50, 100, 300
episode numbers (same subsets for every arm and setting), each quantity recomputed and compared with
its 1,000-path value (bias, RMSE, share undefined). Ordering: for neighbouring settings, the share
of subsets that order the two as the 1,000-path values do.

## 3. Results at 1,000 paths

### 3.1 Over time, at one setting

{time}

### 3.2 Against the control setting (qc = 12, t = 30)

{settings}

### 3.3 Across the whole control plane (t = 30; range over qc × b or qc × λc)

{phase}

## 4. Precision

### 4.1 Error at 50, 100, 300 paths (t = 30; median over settings; symmetric ρ = 1 excluded)

{precision}

### 4.2 How often 100 paths order two neighbouring settings correctly (Simulation 2, by size of the 1,000-path difference)

{ranking}

## 5. Findings

1. **η measures the symmetry of the pair, not the size of control.** It is 0.86–0.95 over most of
   Simulation 2 because rate, gate and qc scale both controllers' effects together (§2.3). It falls
   (a) with a common push — Simulation 1 nosolution, b = 9, both controllers raise A0 (η 0.25 at
   ρ 0.75, 0.06 at ρ 1); (b) with a lopsided pair — Simulation 1 symmetric, larger b makes the false
   controller strong while the truth controller is at the ceiling (0.73 → 0.56); Simulation 2
   nosolution with full memory, faster control stalls the false controller (0.93 → 0.67).
2. **Progress shows what η hides.** Simulation 2 symmetric: the truth controller's progress is
   0.10–0.48 across the plane while η is flat; the false controller's 0.01–0.05. Simulation 2
   nosolution, full memory: the false controller's progress falls with the rate (to 0.01–0.08) while
   the truth controller's holds (0.30–0.49).
3. **The large-budget backfire is negative progress and negative focus** (Simulation 1 nosolution,
   b = 9): the false controller moves A2 down; with full memory its focus is −1.
4. **Focus is close to 1 almost everywhere**: controllers rarely waste change on other answers; the
   exception is the backfire.
5. **Price: slow control and one fact per night are cheapest**; in Simulation 2 the gate makes slow
   control up to twice as cheap per fact.
6. **Controllers make agents surer but some split the swarm**: the false controller on symmetric
   raises disagreement (+0.13 in Simulation 2, +0.20 in Simulation 1) without making agents surer
   (Simulation 2: ΔH̄ ≈ 0) or even making them less sure (Simulation 1: ΔH̄ = +0.09 at b = 3).
7. **Precision at 100 paths is adequate** for one-agent I, η, progress and focus and for swarm I and
   η_JS (§4); swarm-level C and η are not estimable (C infinite in most subsets).

## 6. Open points

- η₃ and η_JS are alternatives to η that never become infinite; which to keep as the pair measure is
  open (η is the one in the reports).
- Swarm-level measures drop A1 (§2.6); a 2-D swarm state is part of the coarse-graining plan
  (`../coarse_graining/2026-10-12_plan_simulator_data.md`), which also covers path-level versions.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-10-12")
    a = ap.parse_args()
    T = tables(RES / a.date)
    text = TEMPLATE
    for k, v in T.items():
        text = text.replace("{" + k + "}", v)
    out = HERE / f"{a.date}_report.md"
    out.write_text(text)
    print("written", out)


if __name__ == "__main__":
    main()
