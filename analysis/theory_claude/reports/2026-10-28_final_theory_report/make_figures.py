"""Figures for final_theory_report.tex: copied from the two 28-October reports (no new runs).
Usage, from the repository root:
    .venv/bin/python analysis/theory_claude/reports/2026-10-28_final_theory_report/make_figures.py"""
import pathlib, shutil

HERE = pathlib.Path(__file__).resolve().parent
R = HERE.parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)
for name in ("fig2_mf_reversal", "fig3_dose_sensitivity", "fig4_threshold", "fig5_finite_reversal", "fig6_diversity",
             "fig7_scaling", "fig8_counterevidence", "fig9_gating", "fig10_susceptibility"):
    shutil.copy(R / "2026-10-28_session_report" / "figures" / f"{name}.pdf", OUT / f"{name}.pdf")
for name in ("fig_kernels", "fig_selection"):
    shutil.copy(R / "2026-10-28_session_report_detailed" / "figures" / f"{name}.pdf", OUT / f"{name}.pdf")
print("->", OUT)
