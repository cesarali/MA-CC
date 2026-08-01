#!/usr/bin/env python3
"""Check the University of Potsdam LLM proxy through MA-CC's shared client.

Run from the repository root with the project conda environment:

    conda run --live-stream -n MA-CC python scripts/Potsdam/check_university_api.py

The script reads ``POTSDAM_API_KEY``, ``BASE_POTSDAM_LLM_URL``, and the optional
``POTSDAM_MODEL`` directly from the repository-root ``.env``. Add ``--chat``
only for a minimal, billable completion.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path

import requests
from dotenv import dotenv_values

from naming_game.api_client import AsyncLLMClient, LLMAPIError
from naming_game.models import ConfigurationError


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = REPOSITORY_ROOT / ".env"
DEFAULT_MODEL = "gwdg/qwen3-30b-a3b-instruct-2507"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check University of Potsdam LLM proxy connectivity."
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Concrete chat model to verify. Overrides POTSDAM_MODEL; if neither "
            f"is set, use {DEFAULT_MODEL}."
        ),
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Send one minimal, billable chat completion after model discovery.",
    )
    parser.add_argument(
        "--direct",
        action="store_true",
        help=(
            "Under WSL, bypass the automatic Windows VPN bridge and test the "
            "raw WSL network path directly."
        ),
    )
    parser.add_argument(
        "--windows",
        action="store_true",
        help="Under WSL, run the complete diagnostic in Windows PowerShell.",
    )
    return parser.parse_args()


def load_config(model_override: str | None) -> tuple[str, str, str]:
    """Read the Potsdam configuration from .env without exposing credentials."""
    if not ENV_PATH.is_file():
        raise ConfigurationError(f"Repository .env not found at {ENV_PATH}.")

    values = dotenv_values(ENV_PATH)
    api_key = (values.get("POTSDAM_API_KEY") or "").strip()
    base_url = (values.get("BASE_POTSDAM_LLM_URL") or "").strip()
    env_model = (values.get("POTSDAM_MODEL") or "").strip()
    model = (model_override or env_model or DEFAULT_MODEL).strip()

    if not api_key:
        raise ConfigurationError("POTSDAM_API_KEY is missing or empty in .env.")
    if not base_url:
        raise ConfigurationError(
            "BASE_POTSDAM_LLM_URL is missing or empty in .env."
        )
    if not model:
        raise ConfigurationError("The requested model is empty.")

    return api_key, base_url, model


def is_wsl() -> bool:
    return "microsoft" in os.uname().release.lower()


def run_via_windows(args: argparse.Namespace) -> int:
    """Use the Windows network stack so a host VPN is available from WSL."""
    companion = Path(__file__).resolve().with_suffix(".ps1")
    try:
        converted = subprocess.run(
            ["wslpath", "-w", str(companion)],
            check=True,
            capture_output=True,
            text=True,
        )
        command = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            converted.stdout.strip(),
            "-TimeoutSec",
            str(args.timeout),
        ]
        if args.model:
            command.extend(["-Model", args.model])
        if args.chat:
            command.append("-Chat")
        completed = subprocess.run(command, check=False)
        return completed.returncode
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(
            f"API check failed: could not launch the Windows helper ({exc}).",
            file=sys.stderr,
        )
        return 2


async def check_api(args: argparse.Namespace) -> None:
    api_key, base_url, model = load_config(args.model)
    client = AsyncLLMClient(
        model=model,
        timeout_seconds=args.timeout,
        max_retries=0,
        api_key=api_key,
        base_url=base_url,
        allow_windows_proxy=not args.direct,
    )
    try:
        if is_wsl() and not args.direct:
            print("Using the restricted Windows VPN bridge for this WSL process.")
        print(f"Configuration loaded from {ENV_PATH}; checking model {model!r}.")
        models = await client.validate_model()
        print(f"University proxy reachable; {len(models)} models listed.")
        print(f"Requested model available: True ({model})")

        if not args.chat:
            print("Connectivity/model check passed; use --chat for a completion test.")
            return

        print(f"Starting minimal chat test with {model} …")
        response = await client.complete(
            [{"role": "user", "content": "Reply exactly: API works."}],
            temperature=0,
            max_tokens=16,
        )
        print(f"Chat result received: {response.content.strip()!r}")
        print(f"Usage: {response.usage}")
    finally:
        client.close()


def main() -> int:
    args = parse_args()
    if is_wsl() and args.windows:
        return run_via_windows(args)
    asyncio.run(check_api(args))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except requests.Timeout as exc:
        print(
            "API check failed: the University proxy timed out. Connect to the "
            "University of Potsdam network/VPN and try again.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    except requests.ConnectionError as exc:
        print(
            "API check failed: could not connect to the University proxy. Check "
            "the Potsdam VPN/network connection and BASE_POTSDAM_LLM_URL.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc
    except (ConfigurationError, LLMAPIError, OSError, ValueError) as exc:
        print(f"API check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
