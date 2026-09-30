"""Observables and efficiencies for the 21-09-2026-full-vs-report-v1 study.

Implements BLACKBOARD_OBSERVABLES_AND_EFFICIENCIES.md against the
new_rnd_init_experiment archive.  The archive is read-only; nothing in this
package writes to it.

Vocabulary used throughout (see the reference document for the full definitions):

* init      one physical initialization, identified by ``physical_initial_state_hash``.
            The 60 initializations are reused across settings and are the only
            independent unit.  A "parent" in the paired-experiment code.
* setting   a (communication profile, evidence persistence rho) pair.
* arm       ``silent`` (no controller), ``t0`` (truth-target) or ``t2`` (false-target).
* comparison  a setting plus a posting budget b; it holds the initializations that
            have all three arms complete.
* Y_t       the three-count vote vector after t rounds, Y_0 the initialization state.
"""

__version__ = "0.1.0"
