"""Build the extended reports of 12 October 2026 from the dated originals.

Each new report = the original, unchanged, plus (a) one paragraph after the title explaining how
the simulator works and (b) a section on the efficiency of control before the closing caveats.
The original's text is checked to be contained, unchanged, in the new file.

Usage, from the repository root (after make_report_figures.py):
    .venv/bin/python analysis/task003_llm_free/shared_analysis/efficiency_estimates/extend_reports.py
"""
from __future__ import annotations
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
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

SECTION_HEAD = r"""
\section{How efficiently does the controller steer?}\label{sec:eff}

A controller can change the swarm in two ways: in the direction it is asked to, or in ways that
happen whatever its target. We measure how readable the command is in the swarm, how far each
controller gets toward its target, how much of its change is directed, what each fact it posts buys,
and how unsure the agents become. Throughout, the target is chosen by a fair coin, and all
quantities are taken at one time $t$, across the 1{,}000 episodes of each arm (not over whole
trajectories). Setting: probability matching, $q = 6$.

\subsection{Measures}
""" + DIAGRAM + r"""
\noindent\textbf{Distributions.} For each arm $a$ (silent $0$, target A0 $+$, target A2 $-$) and a
time $t$, each episode $e$ contributes a probability vector $y_a(e)$ over the values $x$ of the
state $X$: at \emph{one-agent} level its vote shares (the vote of a randomly chosen agent), at
\emph{swarm} level a mark on the bin of $s = m_{\text{A0}} - m_{\text{A2}}$ (difference of mean
beliefs, three equal bins on $[-1,1]$). Then $\hat P_a = \frac1n\sum_e y_a(e)$. Any pair of controllers
can be written as
\begin{equation}\label{eq:split}
\hat P_\pm = \hat P_0 + \mu \pm \delta,\qquad \mu = \tfrac12(\hat P_+ + \hat P_-) - \hat P_0,\quad
\delta = \tfrac12(\hat P_+ - \hat P_-),
\end{equation}
a \emph{common} part $\mu$ (what control does on average, whatever the target) and a
\emph{directional} part $\delta$; both are vectors over the values of $X$ (all answers, A1 included).

\noindent\textbf{Readability and the pair efficiency.} With $\hat Q = \hat P_0 + \mu$ and
$\mathrm{KL}(p\Vert q) = \sum_x p(x)\log_2[p(x)/q(x)]$,
\begin{equation}\label{eq:eta}
I = \tfrac12\mathrm{KL}(\hat P_+\Vert\hat Q) + \tfrac12\mathrm{KL}(\hat P_-\Vert\hat Q),\quad
C = \tfrac12\mathrm{KL}(\hat P_+\Vert\hat P_0) + \tfrac12\mathrm{KL}(\hat P_-\Vert\hat P_0) = I + R,\quad
R = \mathrm{KL}(\hat Q\Vert\hat P_0),\quad \eta = I/C .
\end{equation}
$I$ is the mutual information between the target and the state: how readable the command is
(at most 1 bit). $I = \frac12\mathrm{KL}(\hat Q+\delta\Vert\hat Q) + \frac12\mathrm{KL}(\hat Q-\delta\Vert\hat Q)$
depends on $\delta$ and on where the pair sits ($\hat Q$); $R = \mathrm{KL}(\hat P_0+\mu\Vert\hat P_0)$ depends
only on $\mu$; $C$ (a number) $= I + R$, so $R = C - I$. The reference $\hat P_0$ is the silent arm. For
small changes, $\mathrm{KL}(p+\epsilon\Vert p) \approx \sum_x \epsilon(x)^2/(2\ln 2\,p(x))$ gives
\begin{equation}\label{eq:etasmall}
I \approx \frac{\Vert\delta\Vert^2}{2\ln 2},\quad R \approx \frac{\Vert\mu\Vert^2}{2\ln 2},\quad
\eta \approx \frac{1}{1 + \Vert \mu\Vert^2/\Vert\delta\Vert^2},\qquad \Vert v\Vert^2 = \sum_x \frac{v(x)^2}{\hat P_0(x)} .
\end{equation}
So $\eta = 1$ exactly when the two controllers' changes are mirror images ($\mu = 0$), whatever their
size, and $\eta$ falls only through the \emph{ratio} of common to directional change: when both push
the same way (a common push), or one moves the swarm much more than the other (a lopsided pair:
along one direction with sizes $a$ and $b$, $\Vert \mu\Vert/\Vert\delta\Vert = |a-b|/(a+b)$). It is a
property of the pair, not of either controller, and it stays constant whenever both effects grow
together. At swarm level $C$ is often infinite (the controlled swarm reaches states silence never
visits); there we use $\eta_{JS} = I/(I + \mathrm{JS}(\hat Q,\hat P_0))$ with the bounded
Jensen--Shannon divergence; $\mathrm{JS}(\hat Q,\hat P_0)$ is the information the state carries about
\emph{whether} a controller is present (controlled or silent, 50/50), so $\eta_{JS}$ compares
information about the direction with information about the presence of control.

\noindent\textbf{Each controller against its own target} $z$ (one agent, votes):
\begin{equation}\label{eq:progress}
S_z = \mathrm{KL}\big(\mathrm{Bern}(\hat P_z(z))\,\Vert\,\mathrm{Bern}(\hat P_0(z))\big),\qquad
\pi_z = \frac{S_z}{\log_2[1/\hat P_0(z)]},\qquad
\varphi_z = \frac{S_z}{\mathrm{KL}(\hat P_z\Vert\hat P_0)},\qquad
\text{price} = \frac{S_z}{\text{facts posted}} .
\end{equation}
$S_z$ is the least information any intervention must inject to raise the target's share from its
silent value to the observed one (raise the target, scale the other answers in proportion).
\textbf{Progress} $\pi_z$ is the share of the way from silence to unanimity on the target, in bits;
\textbf{focus} $\varphi_z$ the share of the controller's actual change that was needed for that gain;
both are negative when the controller moved its target \emph{down}. The \textbf{price} is the
information injected per fact the controller posted. Finally, the agents' own uncertainty:
$\Delta\bar H$ (mean entropy of their beliefs) and $\Delta$JSD (disagreement, the entropy of the mean
belief minus $\bar H$), controlled minus silent, same episode.
"""

TIME_FIG = r"""
\subsection{Over time, at one setting}

\begin{figure}[!htb]
\centering\includegraphics[width=0.92\linewidth]{figures/fig_eff_time2.pdf}
\caption{At SPECSET: readability $I(t)$, each controller's progress $\pi_z(t)$, directedness
(focus $\varphi_z$, pair efficiency $\eta$, $\eta_{JS}$), and the agents' uncertainty. Bands: 95\%
intervals over episodes.}
\label{fig:efftime2}
\end{figure}
"""

PHASE_FIGS = r"""
\subsection{Across the control plane}

\begin{figure}[!htb]
\centering\includegraphics[width=PROGW\linewidth]{figures/fig_eff_phase_progress.pdf}
\caption{Progress $\pi_z$ of each controller at $t = 30$ over the reading budget $q_c$ and XPHASE
(one agent, votes). Red: the controller moved its target down. Dash: undefined (silence already at
unanimity).}
\label{fig:effprog}
\end{figure}

\begin{figure}[!htb]
\centering
\includegraphics[width=SMALLW\linewidth]{figures/fig_eff_phase_focus.pdf}\hfill
\includegraphics[width=SMALLW\linewidth]{figures/fig_eff_phase_price.pdf}
\caption{Left: focus $\varphi_z$ (negative: the change went against the target). Right: price,
information injected per fact posted (bits). $t = 30$; GATENOTE Dash: undefined (no change, or the
controlled swarm considers an answer silence rules out completely).}
\label{fig:efffocusprice}
\end{figure}

\begin{figure}[!htb]
\centering\includegraphics[width=PHASEW\linewidth]{figures/fig_eff_phase.pdf}
\caption{Pair efficiency at $t = 30$ over $q_c$ and XPHASE: one-agent $\eta$ (beliefs) and swarm
$\eta_{JS}$. Dash: undefined.}
\label{fig:effphase}
\end{figure}
"""

SETTINGS_FIGS = r"""
\subsection{Readability, pair efficiency and uncertainty against XSHORT}

\begin{figure}[!htb]
\centering\includegraphics[width=0.72\linewidth]{figures/fig_eff_settings.pdf}
\caption{Readability $I$ (left), pair efficiency (middle) and the change in the agents' uncertainty
(right; circles $\Delta\bar H$, triangles $\Delta$JSD) at $t = 30$ against XLABEL. Bars: 95\% intervals.}
\label{fig:effsettings}
\end{figure}
"""

TIME_RATES = r"""
\begin{figure}[!htb]
\centering\includegraphics[width=0.66\linewidth]{figures/fig_eff_time.pdf}
\caption{Readability and pair efficiency over time, $\rho = 0.75$, $q_c = 12$, gate on, one line per
rate; the controller stops at $t = 30$ (dashed). Bands: 95\% intervals.}
\label{fig:efftime}
\end{figure}
"""

PRECISION = r"""
\subsection{Precision at 50, 100 and 300 paths}

\begin{table}[H]
\centering\small
\caption{Value from 1{,}000 paths per arm, and root-mean-square error of the estimate from 50, 100
or 300 paths (200 random subsets each), at $t = 30$, MAINSETTING. $^\ast$: share of subsets in which
the quantity is infinite or undefined.}
\label{tab:effprec}
\begin{tabular}{lcccc}
\toprule
quantity & 1{,}000 paths & error, 50 & error, 100 & error, 300 \\
\midrule
ROWS
\bottomrule
\end{tabular}
\end{table}

\noindent Across all settings, at 100 paths: readability $\pm0.01$ (one agent) to $\pm0.03$ (swarm),
progress $\pm0.02$, focus $\pm0.01$--$0.02$, $\eta$ $\pm0.02$--$0.03$, $\Delta\bar H$ $\pm0.02$, $\Delta$JSD
$\pm0.006$. Two settings that differ by at least 0.02 are ordered correctly in 90--100\% of subsets.
"""

TEXT = {
("sim1", "symmetric"): dict(
time=r"""
\begin{itemize}\itemsep1pt
\item \textbf{Control is reached within a few days and then stays.} Both controllers get 90\% of
  their day-30 progress by day 4--5: $\pi_{\text{A0}} = 0.18$, $\pi_{\text{A2}} = 0.15$ at day 30.
\item \textbf{Both controllers are focused} ($\varphi \ge 0.97$): almost all of what each changes moves
  its own target. The pair efficiency is nevertheless only 0.65 ($\eta_{JS}$ 0.77): the pair is
  lopsided (next subsections).
\item \textbf{The false controller confuses}: it makes agents less sure ($\Delta\bar H = +0.09$ bits)
  and splits them ($\Delta$JSD $= +0.20$); the truth controller makes them surer ($-0.24$) and more
  alike ($-0.05$).
\end{itemize}
""",
plane=r"""
\begin{itemize}\itemsep1pt
\item \textbf{The false controller's progress grows with $b$}, from 0.02--0.08 at $b = 1$ to 0.43--0.49
  at $b = 9$; the truth controller's is set by $q_c$ (0.25--0.42 at $q_c = 3$, 0.03--0.06 at $q_c = 24$):
  the silent swarm already sits at 85\% A0, so its vote gate (at least 75\% of the posts read) keeps it
  silent unless it reads few posts.
\item \textbf{Focus stays at 0.93 or more everywhere}: neither controller wastes its change.
\item \textbf{Each extra fact buys less}: the price (target A0) falls from up to 0.020 bits per fact at
  $b = 1$ to 0.002--0.005 at $b = 9$.
\item \textbf{With full memory} every agent proves A0 and nothing is measurable.
\end{itemize}
""",
pair=r"""
\begin{itemize}\itemsep1pt
\item \textbf{$\eta$ falls with $b$ (0.73 to 0.56) because the pair becomes lopsided}, not because
  either controller wastes effort: the false controller moves the A0 share from 85\% to 27\% at
  $b = 9$, the truth controller only to 93\% (near the ceiling). On average control then shifts the
  swarm toward A2; that shift is the same whatever the target, so $\eta$ counts it as undirected.
\item Readability rises with $b$ (swarm 0.24 to 0.94 bits).
\end{itemize}
"""),
("sim1", "nosolution"): dict(
time=r"""
\begin{itemize}\itemsep1pt
\item \textbf{Control is reached fast}: 90\% of the day-30 progress by day 9 (target A0) and day 4
  (target A2); $\pi_{\text{A0}} = 0.37$, $\pi_{\text{A2}} = 0.21$ at day 30, with focus 0.93 and 0.99.
\item \textbf{Both controllers make agents surer} ($\Delta\bar H \approx -0.39$ bits); the truth
  controller also splits them a little ($\Delta$JSD $+0.06$).
\end{itemize}
""",
plane=r"""
\begin{itemize}\itemsep1pt
\item \textbf{The large-budget backfire is negative progress.} At $b = 9$ the false controller moves
  A2 \emph{down} ($\pi_{\text{A2}}$ down to $-0.03$ at $\rho = 0.75$ and $-0.18$ at $\rho = 1$) and its focus
  turns negative ($-0.19$ to $-0.45$ at $q_c \le 12$; $-1$ with full memory): its change goes against
  its own target.
  The truth controller keeps progressing (0.19--0.51).
\item \textbf{The cheapest control is one fact per night}: the price (target A0) falls from
  0.03--0.13 bits per fact at $b = 1$ to 0.002--0.013 at $b = 9$; with full memory, reading more posts
  makes each fact worth more (0.03 at $q_c = 3$, 0.13 at $q_c = 24$, $b = 1$).
\end{itemize}
""",
pair=r"""
\begin{itemize}\itemsep1pt
\item \textbf{$\eta$ collapses at $b = 9$} (0.25 at $\rho = 0.75$, 0.06 at $\rho = 1$) through a
  \emph{common push}: both controllers raise A0. Up to $b = 6$, at $\rho = 0.75$, the pair is balanced
  ($\eta \approx 0.9$).
\item The swarm-level readability saturates near 1 bit with full memory (every episode ends on its
  controller's side), so it measures distinguishability, not size; progress measures size.
\end{itemize}
"""),
("sim2", "symmetric"): dict(
time=r"""
\begin{itemize}\itemsep1pt
\item \textbf{Control builds slowly and stays}: 90\% of the $t = 30$ progress by $t \approx 15$ (target
  A0) and 19 (target A2); it is still there at $t = 40$. The truth controller gets 18\% of the way to
  unanimity, the false controller 2\% (A2 starts at 12\%, far from unanimity).
\item \textbf{Both are focused} ($\varphi \ge 0.99$) and the pair is balanced ($\eta = 0.91$).
\item \textbf{The false controller splits the swarm} ($\Delta$JSD $+0.13$) without changing individual
  certainty; the truth controller makes agents surer ($-0.18$ bits).
\end{itemize}
""",
plane=r"""
\begin{itemize}\itemsep1pt
\item \textbf{The truth controller's progress varies fourfold where $\eta$ is flat}: 0.10--0.40 with
  the gate on, 0.22--0.48 off; it grows with the rate and falls with $q_c$ (reading more posts lets the
  gate keep it silent). The false controller stays at 0.01--0.05.
\item \textbf{Focus is 0.95 or more everywhere.}
\item \textbf{Slow control is cheapest}: 0.011--0.021 bits per fact at $\lambda_c = 0.5$, 0.003--0.011
  at $\lambda_c = 8$ (target A0, gate on).
\end{itemize}
""",
pair=r"""
\begin{itemize}\itemsep1pt
\item \textbf{$\eta$ is almost constant (0.88--0.94) because every knob scales both controllers alike.}
  From $\lambda_c = 0.5$ (gate on) to $\lambda_c = 8$ (gate off) the A0 share moves $+0.07 / -0.09$ and then
  $+0.11 / -0.17$: both changes grow 1.7--2 times, $I$ and $R$ grow together, and by
  Eq.~(\ref{eq:etasmall}) their ratio, hence $\eta$, stays. In Simulation 1, larger batches made the
  false controller much stronger while the truth controller hit the ceiling, so $\eta$ fell.
\item With full memory every agent proves A0; nothing is measurable.
\end{itemize}
"""),
("sim2", "nosolution"): dict(
time=r"""
\begin{itemize}\itemsep1pt
\item \textbf{Control builds within 10--15 time units and persists}: $\pi_{\text{A0}} = 0.37$,
  $\pi_{\text{A2}} = 0.20$ at $t = 30$, almost unchanged at $t = 40$; focus 0.94 and 0.99, $\eta = 0.90$.
\item \textbf{Agents become surer} ($\Delta\bar H = -0.28$ bits) \textbf{but disagree more}
  ($\Delta$JSD $+0.11$ for target A0, $+0.06$ for target A2).
\end{itemize}
""",
plane=r"""
\begin{itemize}\itemsep1pt
\item \textbf{When agents forget, progress hardly depends on the rate, gate or $q_c$}
  ($\pi_{\text{A0}}$ 0.28--0.39, $\pi_{\text{A2}}$ 0.14--0.23), but the price does: up to 0.067 bits per fact
  at $\lambda_c = 0.5$ with the gate on, 0.016 at $\lambda_c = 8$; the gate makes slow control up to twice
  as cheap.
\item \textbf{With full memory the false controller stalls as the rate rises}: $\pi_{\text{A2}}$ falls from
  0.11--0.18 at $\lambda_c = 0.5$ to 0.01--0.08 at $\lambda_c = 8$ (gate on) while $\pi_{\text{A0}}$ stays at
  0.30--0.49. Its focus stays near 1: it still
  pushes toward A2, it just gets almost nowhere (Section~3: more different pool facts pull both
  targets toward the full-pool belief).
\end{itemize}
""",
pair=r"""
\begin{itemize}\itemsep1pt
\item \textbf{When agents forget, $\eta$ is flat (0.86--0.95)}: the two controllers' effects grow together.
\item \textbf{With full memory $\eta$ falls with the rate} (0.93 to 0.67) because the pair turns
  lopsided: the false controller's progress collapses, the truth controller's does not. Swarm
  $\eta_{JS}$ moves the other way, because swarm readability is saturated at 1 bit; the one-agent
  $\eta$ is the informative one here.
\end{itemize}
"""),
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
    main = "$b = 3$, $q_c = 12$, $\\rho = 0.75$" if sim == "sim1" else "$\\lambda_c = 1$, $q_c = 12$, gate on, $\\rho = 0.75$"
    specset = ("$\\rho = 0.75$, $q_c = 12$, $b = 3$ (end of each day)" if sim == "sim1"
               else "$\\rho = 0.75$, $q_c = 12$, $\\lambda_c = 1$, gate on (the controller stops at $t = 30$, dashed)")
    xlabel = ("facts per night $b$ ($q_c = 12$)" if sim == "sim1"
              else "the controller rate $\\lambda_c$ ($q_c = 12$; solid: gate on; dashed: gate off)")
    xphase = "the controller rate $\\lambda_c$" if sim == "sim2" else "the facts per night $b$"
    xshort = "the facts per night" if sim == "sim1" else "the controller rate"
    w = {("sim1", "symmetric"): ("0.42", "0.34", "0.32"), ("sim1", "nosolution"): ("0.44", "0.38", "0.36"),
         ("sim2", "symmetric"): ("0.66", "0.36", "0.55"), ("sim2", "nosolution"): ("0.7", "0.4", "0.62")}[key]
    gatenote = "gate on. " if sim == "sim2" else ""
    T = TEXT[key]
    rows = (base / new / "figures" / "tab_eff_precision.tex").read_text().strip()
    section = (SECTION_HEAD
               + TIME_FIG.replace("SPECSET", specset) + T["time"]
               + PHASE_FIGS.replace("XPHASE", xphase).replace("PROGW", w[0]).replace("SMALLW", w[1])
                 .replace("PHASEW", w[2]).replace("GATENOTE", gatenote) + T["plane"]
               + SETTINGS_FIGS.replace("XLABEL", xlabel).replace("XSHORT", xshort)
               + (TIME_RATES if sim == "sim2" else "") + T["pair"]
               + PRECISION.replace("MAINSETTING", main).replace("ROWS", rows) + "\n")
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
