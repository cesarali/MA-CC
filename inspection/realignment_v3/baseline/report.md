# Phase 3–6 realignment baseline

- Git revision: `73c2bd27a329c43beb873f85f57969c8b4bc5a6c`
- Captured: 2026-08-02 before Version 3 prompt-kernel edits
- Authoritative environment: `MA-CC` / Python 3.11.15
- Baseline suite: 210 tests passed.
- The ambient Python 3.10 runner failed during collection in five unchanged
  `src/naming_game` tests because `datetime.UTC` requires Python 3.11.
- Phase 3, 4, 5, and 6 baseline inspections all passed and were written to
  separate subdirectories. They are not overwritten by Version 3 inspections.
- `src/naming_game` is outside the realignment scope and remains unchanged.

The Phase 6 selected traces freeze exact empty, representative, and maximum
bounded-memory wire messages. They include both `Q, M` and `M, Q` presented
orders. The Phase 5 interactions and game-call plan freeze the toy messages and
planning estimates.
