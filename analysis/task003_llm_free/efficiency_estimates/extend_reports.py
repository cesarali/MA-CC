"""Build the extended reports of 12 October 2026 from the dated originals.

Each new report = the original, unchanged, plus (a) one paragraph after the title explaining how
the simulator works and (b) a section on the efficiency of control before the closing caveats.
The original's text is checked to be contained, unchanged, in the new file.

Usage, from the repository root (after make_report_figures.py):
    .venv/bin/python analysis/task003_llm_free/efficiency_estimates/extend_reports.py
"""
from __future__ import annotations
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPORTS = {
    ("sim1", "symmetric"): ("simulation_1", "2026-10-07_report1_symmetric", "2026-10-12_report1_symmetric"),
    ("sim1", "nosolution"): ("simulation_1", "2026-10-07_report2_nosolution", "2026-10-12_report2_nosolution"),
    ("sim2", "symmetric"): ("simulation_2", "2026-10-09_report1_symmetric", "2026-10-12_report1_symmetric"),
    ("sim2", "nosolution"): ("simulation_2", "2026-10-09_report2_nosolution", "2026-10-12_report2_nosolution"),
}

SIMULATOR = {
"sim1": r"""\paragraph{How the simulator works.} The world, \texttt{task\_003}, has three candidate
allocations; A0 is the correct one. Each of 24 agents is given one true fact about it and reasons as
an exact Bayesian: its \emph{belief} is the exact probability of A0, A1 and A2 given the facts it
currently remembers. An episode has 30 rounds, each a day and a night. During the day the agents act
one at a time in a new random order: an agent reads $q$ posts drawn at random from a shared
\emph{board} (never its own), adds their facts to its memory, recomputes its belief, \emph{votes}
(\emph{argmax}: the most probable allocation; \emph{probability matching}: a vote drawn from its
belief) and posts one remembered fact that supports its vote, or nothing if none does. At night a
\emph{controller} with a target, A0 (truth) or A2 (false), reads $q_c$ agent posts; unless at least
75\% of them come from agents voting for its target, or what it read already proves its target, it
posts the $b$ facts from its own pool that best support its target. The board is then cleared,
keeping only the controller's new posts. At each dawn every remembered fact is forgotten with
probability $1-\rho$ and comes back if read again. Every episode is run three times with the same
random draws: without a controller, with the truth controller and with the false controller.
""",
"sim2": r"""\paragraph{How the simulator works.} The world, \texttt{task\_003}, has three candidate
allocations; A0 is the correct one. Each of 24 agents is given one true fact about it and reasons as
an exact Bayesian: its \emph{belief} is the exact probability of A0, A1 and A2 given the facts it
currently remembers. There are no rounds. Each agent acts at random times, on average once per time
unit: it reads $q$ posts drawn at random from the 48 most recent posts on a shared \emph{board}
(never its own), adds their facts to its memory, recomputes its belief, \emph{votes} by probability
matching (a vote drawn from its belief) and posts one remembered fact that supports its vote, or
nothing if none does. Every remembered fact is forgotten after a random lifetime (on average a
fraction $\rho$ survives one time unit; reading it again renews it). A \emph{controller} with a
target, A0 (truth) or A2 (false), acts at random times at rate $\lambda_c$ until $t = 30$: it reads
$q_c$ recent agent posts and, unless the \emph{vote gate} (at least 75\% of them from agents voting
for its target) or the \emph{proof stop} (its target already proved) silences it, posts the one fact
from its pool that best supports its target, at most 30 messages per episode. Agents continue alone
until $t = 40$. Every episode is run three times with the same random draws: without a controller,
with the truth controller and with the false controller.
""",
}

DIAGRAM = r"""
\begin{center}
\begin{tikzpicture}[font=\small, dist/.style={draw, circle, minimum size=1.15cm, align=center, inner sep=1pt}]
  \node[dist, fill=gray!15] (P0) at (0,0) {$P_0$\\[-2pt]{\scriptsize silent}};
  \node[dist, fill=blue!12] (Pp) at (-3.2,2.1) {$P_+$\\[-2pt]{\scriptsize target A0}};
  \node[dist, fill=red!12] (Pm) at (3.2,2.1) {$P_-$\\[-2pt]{\scriptsize target A2}};
  \node[dist, dashed, fill=violet!8] (Q) at (0,2.1) {$Q$\\[-2pt]{\scriptsize mixture}};
  \draw[-{Stealth}, very thick, gray] (P0) -- node[left, font=\scriptsize, pos=0.55]{total change $C$} (Pp);
  \draw[-{Stealth}, very thick, gray] (P0) -- (Pm);
  \draw[-{Stealth}, thick, violet] (P0) -- node[right, font=\scriptsize]{undirected $R$} (Q);
  \draw[-{Stealth}, thick, blue!60!black] (Q) -- node[above, font=\scriptsize]{directed $I$} (Pp);
  \draw[-{Stealth}, thick, red!60!black] (Q) -- node[above, font=\scriptsize]{directed $I$} (Pm);
\end{tikzpicture}
\diagramcaption{Three distributions of the swarm's state at time $t$, one per arm, each over the
episodes of that arm. The controller's total change $C$ splits exactly into a part that depends on
its target ($I$, from the mixture $Q$ to each arm) and a part that does not ($R$, from silence to
$Q$). Efficiency $\eta = I/C$ is the directed share.}
\end{center}
"""

DEFINITIONS = r"""
\section{How efficiently does the controller steer?}\label{sec:eff}

A controller can change the swarm in two ways: in the direction it is asked to, or in ways that
happen whatever its target (for example pulling agents out of A1, or splitting them). We call the
share of the first kind its \emph{efficiency}. The target is chosen by a fair coin, so how much the
swarm's state reveals about it is a well-defined information.
""" + DIAGRAM + r"""
\noindent For each arm $a$ (silent $0$, target A0 $+$, target A2 $-$) and a fixed time $t$, each
episode $e$ contributes a probability vector $y_a(e)$ over the values $x$ of the state $X$, and
\begin{equation}\label{eq:eff}
\hat P_a(x) = \frac{1}{n}\sum_{e=1}^{n} y_a(e)(x),\qquad
I = \tfrac12\,\mathrm{KL}(\hat P_+\Vert \hat Q) + \tfrac12\,\mathrm{KL}(\hat P_-\Vert \hat Q),\qquad
C = \tfrac12\,\mathrm{KL}(\hat P_+\Vert \hat P_0) + \tfrac12\,\mathrm{KL}(\hat P_-\Vert \hat P_0),
\end{equation}
with $\hat Q = \frac12(\hat P_+ + \hat P_-)$ and $\mathrm{KL}(p\Vert q) = \sum_x p(x)\log_2[p(x)/q(x)]$.
$I$ is the mutual information $I(Z; X_t)$ between the target $Z$ (A0 or A2, chosen by a fair coin)
and the state $X_t$; silence does not enter it. The reference $\hat P_0$ in $C$ is the
distribution of the \emph{silent} arm. Exactly $C = I + \mathrm{KL}(\hat Q\Vert\hat P_0)$, and
$\eta = I/C \in [0,1]$. Two resolutions:
\textbf{one agent} ($X$ = the vote of a randomly chosen agent; $y_a(e)$ = that episode's vote shares)
and the \textbf{swarm} ($X$ = $s = m_{\text{A0}} - m_{\text{A2}}$, the difference of the agents' mean
beliefs, in three equal bins on $[-1,1]$; $y_a(e)$ marks the episode's bin). At swarm level the
controlled swarm often reaches states the silent swarm never visits, which makes $C$ infinite; there
we use the bounded version $\eta_{JS} = I/(I + \mathrm{JS}(\hat Q,\hat P_0))$, with JS the
Jensen--Shannon divergence. We also measure how unsure the agents become: $\bar H$, the mean entropy
of the agents' beliefs (0 to 1.58 bits), and the disagreement $\mathrm{JSD} = H(\text{mean belief}) -
\bar H$, each as controlled minus silent in the same episode ($\Delta\bar H$, $\Delta$JSD). All
quantities are taken at one time, across episodes; they are not divergences between whole
trajectories (those would be at least as large). Setting: probability matching, $q = 6$,
$q_c = 12$ (Figure~\ref{fig:effphase}: every $q_c$); 1{,}000 episodes per arm.

\paragraph{What $\eta$ does and does not say.} $\eta = 1$ when the two controllers move the swarm by
equal amounts in opposite directions around silence. It falls in two cases: both controllers push
the swarm the same way (a \emph{common push}), or one moves it much more than the other (a
\emph{lopsided} response, for example when silence is already close to one target). $\eta$ measures
how directed the change is, not how large it is ($I$) or what it costs; information per controller
message ($I$ divided by the messages sent) varies far more across settings.
"""

PRECISION = r"""
\begin{table}[H]
\centering\small
\caption{Precision. Value from 1{,}000 paths per arm, and root-mean-square error of the estimate from
50, 100 or 300 paths (200 random subsets each), at $t = 30$, MAINSETTING. $^\ast$: share of subsets in
which the quantity is infinite or undefined.}
\label{tab:effprec}
\begin{tabular}{lcccc}
\toprule
quantity & 1{,}000 paths & error, 50 & error, 100 & error, 300 \\
\midrule
ROWS
\bottomrule
\end{tabular}
\end{table}
"""

FIGS = r"""
\begin{figure}[!htb]
\centering\includegraphics[width=0.82\linewidth]{figures/fig_eff_settings.pdf}
\caption{Steering $I$ (left), efficiency (middle) and the change in the agents' uncertainty
(right; circles $\Delta\bar H$, triangles $\Delta$JSD) at $t = 30$ against XLABEL. Bars: 95\%
intervals.}
\label{fig:effsettings}
\end{figure}

\begin{figure}[!htb]
\centering\includegraphics[width=PHASEW\linewidth]{figures/fig_eff_phase.pdf}
\caption{Phase diagram of efficiency at $t = 30$ over the controller's reading budget $q_c$ and
XPHASE ($q = 6$): one-agent $\eta$ (beliefs) and swarm $\eta_{JS}$. Dash: undefined (the
controller changes almost nothing).}
\label{fig:effphase}
\end{figure}

\begin{figure}[!htb]
\centering\includegraphics[width=0.76\linewidth]{figures/fig_eff_time.pdf}
\caption{Over time, $\rho = 0.75$, TIMESET. Bands: 95\% intervals.}
\label{fig:efftime}
\end{figure}
"""

FINDINGS = {
("sim1", "symmetric"): r"""
\paragraph{Findings.} (Figures~\ref{fig:effsettings}, \ref{fig:efftime}; probability matching,
$\rho = 0.75$.)
\begin{itemize}\itemsep1pt
\item \textbf{Larger batches make the response lopsided.} From $b = 1$ to $b = 9$ the target becomes
  easier to read from the swarm ($I$ from 0.24 to 0.94 bits; one agent 0.10 to 0.39), because the
  false controller now moves the swarm a lot (A0 votes 85\% $\to$ 27\%) while the truth controller
  cannot move it further (85\% $\to$ 93\%, close to the ceiling). Averaged over the two targets,
  control therefore shifts the swarm toward A2, and one-agent $\eta$ falls from 0.73 to 0.56. This
  is not added noise: it is the asymmetry between what the two controllers can do. Reading more
  posts ($q_c$) makes it slightly worse (Figure~\ref{fig:effphase}).
\item \textbf{The false controller confuses rather than convinces.} The truth controller makes agents
  surer ($\Delta\bar H = -0.21$ to $-0.26$ bits) and more alike ($\Delta$JSD $-0.04$ to $-0.06$). The
  false controller makes them less sure ($+0.04$ to $+0.17$) and splits them ($\Delta$JSD $+0.19$ at
  $b \le 3$).
\item \textbf{With full memory there is nothing to measure}: every agent ends up proving A0, both
  controllers leave the same state, $I = 0$ and efficiency is undefined.
\item \textbf{100 paths are enough for differences of about 0.03 or more} (Table~\ref{tab:effprec});
  efficiency is the least precise quantity here, because the effects on symmetric are small.
\end{itemize}
""",
("sim1", "nosolution"): r"""
\paragraph{Findings.} (Figures~\ref{fig:effsettings}, \ref{fig:efftime}; probability matching.)
\begin{itemize}\itemsep1pt
\item \textbf{The large-budget backfire is a collapse of efficiency.} Up to $b = 6$ the change is
  almost all directed (one agent: $\eta = 0.89$--$0.93$ at $\rho = 0.75$); at $b = 9$ it drops to 0.25
  ($\rho = 0.75$) and 0.06 ($\rho = 1$), and the steering $I$ collapses with it. The extra facts move
  the swarm, but not in a target-dependent way.
\item \textbf{With full memory, efficiency falls step by step with $b$} (0.87 at $b = 1$, 0.69 at
  $b = 6$; at $q_c = 24$ it stays above 0.9, Figure~\ref{fig:effphase}): the more of the pool goes
  out, the more both controllers push the swarm toward the same place (a common push toward A0).
\item \textbf{Controllers make agents surer} ($\Delta\bar H \approx -0.4$ bits at $b \le 3$,
  $\rho = 0.75$); the truth controller also splits them a little ($\Delta$JSD $+0.06$ to $+0.07$). At
  $b = 9$ the gain in certainty shrinks.
\item \textbf{The swarm-level $I$ saturates near 1 bit} with full memory: every episode ends on its
  controller's side, however small the push. $I$ measures how distinguishable the outcomes are, not
  how large the effect is; the gains of Section~\ref{sec:budget} measure size.
\item \textbf{100 paths}: errors of about $\pm0.02$--$0.04$ (Table~\ref{tab:effprec}).
\end{itemize}
""",
("sim2", "symmetric"): r"""
\paragraph{Findings.} (Figures~\ref{fig:effsettings}, \ref{fig:efftime}; $\rho = 0.75$.)
\begin{itemize}\itemsep1pt
\item \textbf{Control is small but balanced.} The target is only weakly readable from the swarm
  ($I$ = 0.22--0.33 bits; one agent 0.04--0.12), but the two controllers move it by similar amounts in
  opposite directions ($\eta$ = 0.89--0.94 over the whole plane, Figure~\ref{fig:effphase}), unlike in
  Simulation 1, where the false controller dominated (0.50--0.81).
\item \textbf{Faster controllers and turning the gate off steer more} without losing efficiency.
\item \textbf{The false controller confuses}: it leaves individual certainty unchanged
  ($|\Delta\bar H| \le 0.02$) but splits the swarm ($\Delta$JSD $+0.10$ to $+0.13$). The truth
  controller makes agents surer ($-0.15$ to $-0.34$ bits), more so with the gate off.
\item \textbf{With full memory there is nothing to measure}: every agent proves A0, $I = 0$.
\item \textbf{100 paths}: see Table~\ref{tab:effprec}; the small effects make efficiency the least
  precise quantity.
\end{itemize}
""",
("sim2", "nosolution"): r"""
\paragraph{Findings.} (Figures~\ref{fig:effsettings}, \ref{fig:efftime}.)
\begin{itemize}\itemsep1pt
\item \textbf{When agents forget, $\eta$ is nearly constant} (0.86--0.95 over the whole plane,
  Figure~\ref{fig:effphase}): the two controllers move the swarm about equally far in opposite
  directions whatever the rate, gate or $q_c$. What changes is the price: about 0.12 bits of $I$ per
  message at $\lambda_c = 0.5$ with the gate on, 0.03 at $\lambda_c = 8$. Fast controllers reach their
  information sooner (Figure~\ref{fig:efftime}).
\item \textbf{With full memory, faster control is less efficient}: one-agent $\eta$ falls from 0.93
  ($\lambda_c = 0.5$, gate on) to 0.67 ($\lambda_c = 8$), least at $q_c = 3$ and $24$
  (Figure~\ref{fig:effphase}). This is the mechanism of Section~3 (more different pool facts in
  circulation pull both targets toward the same point), now measured as a common push. Swarm
  $\eta_{JS}$ moves the other way here, because swarm-level $I$ is saturated at 1 bit and cannot
  register the change; the one-agent $\eta$ is the informative one.
\item \textbf{Controllers make agents surer} ($\Delta\bar H \approx -0.27$ bits at $\rho = 0.75$) but
  split them somewhat ($\Delta$JSD $+0.06$ for target A2, $+0.11$ for target A0).
\item \textbf{The swarm-level $I$ saturates at 1 bit} with full memory: every episode ends on its
  controller's side. $I$ measures distinguishability, the gains of Section~3 measure size.
\item \textbf{100 paths}: errors of $\pm0.02$--$0.03$ for $I$ and $\eta$, $\pm0.006$ for $\Delta$JSD
  (Table~\ref{tab:effprec}).
\end{itemize}
""",
}


def build(key):
    sim, setup = key
    folder, old, new = REPORTS[key]
    base = ROOT / folder / "analysis" / "reports"
    src = (base / old / f"{old}.tex").read_text()
    head, sep, tail = src.partition("\\paragraph{Caveats.}")
    assert sep, "no Caveats paragraph"
    anchor = "\\maketitle\n\\vspace{-1cm}\n"
    assert head.count(anchor) == 1
    head = head.replace(anchor, anchor + "\n" + SIMULATOR[sim])
    head = head.replace("\\author{LLM-free simulator, runs of",
                        "\\author{LLM-free simulator; extended with the efficiency of control, 12 October 2026; runs of")
    main = "$b = 1$, $\\rho = 0.75$" if sim == "sim1" else "$\\lambda_c = 1$, gate on, $\\rho = 0.75$"
    xlabel = ("facts per night $b$ ($q = 6$, $q_c = 12$)" if sim == "sim1"
              else "the controller rate $\\lambda_c$ (solid: gate on; dashed: gate off)")
    timeset = ("one line per $b$, end of each day" if sim == "sim1"
               else "gate on, one line per rate; the controller stops at $t = 30$ (dashed)")
    xphase = "the controller rate $\\lambda_c$" if sim == "sim2" else "the facts per night $b$"
    phasew = {("sim1", "symmetric"): "0.42", ("sim1", "nosolution"): "0.45",
              ("sim2", "symmetric"): "0.72", ("sim2", "nosolution"): "0.78"}[key]
    figs = FIGS
    findings = FINDINGS[key]
    if sim == "sim1":                       # one controller step per night: the time course adds little
        k = figs.index("\\begin{figure}[!htb]\n\\centering\\includegraphics[width=0.76")
        figs = figs[:k]
        findings = findings.replace("Figures~\\ref{fig:effsettings}, \\ref{fig:efftime}", "Figure~\\ref{fig:effsettings}")
    section = (DEFINITIONS + figs.replace("XLABEL", xlabel).replace("TIMESET", timeset)
               .replace("XPHASE", xphase).replace("PHASEW", phasew)
               + PRECISION.replace("MAINSETTING", main).replace(
                   "ROWS", (base / new / "figures" / "tab_eff_precision.tex").read_text().strip()) + findings + "\n")
    out = head + section + sep + tail
    # the original text, minus the two insertions, must be unchanged
    check = out.replace(SIMULATOR[sim], "").replace(section, "").replace(
        "\\author{LLM-free simulator; extended with the efficiency of control, 12 October 2026; runs of",
        "\\author{LLM-free simulator, runs of")
    assert check.replace(anchor + "\n", anchor) == src, "original text changed"
    (base / new / f"{new}.tex").write_text(out)
    print("written", base / new / f"{new}.tex")


if __name__ == "__main__":
    for k in REPORTS:
        build(k)
