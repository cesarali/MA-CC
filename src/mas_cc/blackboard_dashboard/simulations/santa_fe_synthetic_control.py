"""Compatibility entry point for the Santa Fe synthetic control package.

Run ``python -m santa_fe.cli --config configs/santa_fe/exploratory.yaml``.
"""
from santa_fe.state import SimulationParameters, Message, AgentState, EpisodeResult
from santa_fe.game import SyntheticGame
from santa_fe.runner import run
from santa_fe.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
