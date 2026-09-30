"""Offline estimators for blackboard control efficiency bounds.

Implements Part II of ``CONTROL_EFFICIENCY_DERIVATION_AND_ESTIMATORS.md``:

* endpoint target information ``I(Z; Y_h)`` by cross-fitted probabilistic
  classification (constant / linear / quadratic logistic / small MLP), plus
  frequency and smoothed plug-in comparisons;
* trajectory cost ``C = sum_z w_z KL(p_z || p_base)`` by controlled-versus-silent
  density-ratio classifiers (summary logistic, flattened-path MLP, GRU) scored
  with held-out NWJ / DV variational objectives;
* parent-grouped cross-fitting, one-standard-error model selection, paired
  parent bootstrap of saved contributions, whole-parent label swaps;
* synthetic known-law benchmarks with exact KL / MI by enumeration;
* a LaTeX/PDF report (``report_tex.py``; requires ``latexmk``).

Everything runs offline from the two immutable archives.  No provider calls.
"""

__version__ = "0.1.0"
