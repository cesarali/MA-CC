"""Provider-backed diagnostic probes that run outside any game's population loop.

A *probe* asks one question about how a model behaves on a single, fully
specified decision.  It reuses a game's prompt machinery verbatim but never
runs that game's dynamics: no rounds, no sensing, no controller policy, no
trajectory.  The unit of work is one prompt and one provider call.
"""

from __future__ import annotations

__all__: list[str] = []
