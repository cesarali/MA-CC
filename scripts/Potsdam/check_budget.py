#!/usr/bin/env python3
"""Show the current University of Potsdam LLM proxy budget."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv


RELEVANT_FIELDS = (
    "max_budget",
    "budget",
    "spend",
    "soft_budget",
    "budget_duration",
    "budget_reset_at",
    "rpm_limit",
    "tpm_limit",
)


def find_repo_env(start: Path) -> Path:
    """Find the closest parent .env without assuming where the script is run."""
    for directory in (start, *start.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not find a .env file in this directory or its parents.")


def find_value(data: Any, field: str) -> Any | None:
    """Find a named field in a potentially nested API response."""
    if isinstance(data, Mapping):
        if field in data and data[field] is not None:
            return data[field]
        for value in data.values():
            found = find_value(value, field)
            if found is not None:
                return found
    elif isinstance(data, list):
        for value in data:
            found = find_value(value, field)
            if found is not None:
                return found
    return None


def as_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def format_number(value: Any) -> str:
    number = as_number(value)
    if number is None:
        return str(value)
    return f"{number:,.6f}".rstrip("0").rstrip(".")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query your University of Potsdam LLM proxy budget."
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds (default: 30).",
    )
    return parser.parse_args()


def is_wsl() -> bool:
    return "microsoft" in os.uname().release.lower()


def run_via_windows(timeout: float) -> int:
    """Use the Windows network stack so a host VPN is available from WSL."""
    companion = Path(__file__).resolve().with_suffix(".ps1")
    try:
        converted = subprocess.run(
            ["wslpath", "-w", str(companion)],
            check=True,
            capture_output=True,
            text=True,
        )
        windows_path = converted.stdout.strip()
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                windows_path,
                "-TimeoutSec",
                str(timeout),
            ],
            check=False,
        )
        return completed.returncode
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"Error: could not launch the Windows helper ({exc}).", file=sys.stderr)
        return 1


def main() -> int:
    args = parse_args()

    if is_wsl():
        return run_via_windows(args.timeout)

    try:
        env_path = find_repo_env(Path.cwd().resolve())
    except FileNotFoundError:
        # Also support invoking the script from outside the repository.
        try:
            env_path = find_repo_env(Path(__file__).resolve().parent)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2

    load_dotenv(env_path)
    api_key = os.getenv("POTSDAM_API_KEY")
    base_url = os.getenv("BASE_POTSDAM_LLM_URL")
    if not api_key or not base_url:
        print(
            "Error: POTSDAM_API_KEY and BASE_POTSDAM_LLM_URL must be set in .env.",
            file=sys.stderr,
        )
        return 2

    endpoint = f"{base_url.rstrip('/')}/user/info"
    try:
        response = requests.get(
            endpoint,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=args.timeout,
        )
        response.raise_for_status()
        data = response.json()
    except requests.Timeout:
        print(
            "Error: the request timed out. Connect to the University of Potsdam "
            "network/VPN and try again.",
            file=sys.stderr,
        )
        return 1
    except requests.RequestException as exc:
        status = exc.response.status_code if exc.response is not None else None
        detail = f"HTTP {status}" if status is not None else exc.__class__.__name__
        print(f"Error: budget request failed ({detail}).", file=sys.stderr)
        return 1
    except ValueError:
        print("Error: the endpoint returned invalid JSON.", file=sys.stderr)
        return 1

    values = {field: find_value(data, field) for field in RELEVANT_FIELDS}
    budget = values["max_budget"]
    if budget is None:
        budget = values["budget"]
    spend = values["spend"]
    budget_number = as_number(budget)
    spend_number = as_number(spend)

    print("University of Potsdam LLM account")
    if budget is not None:
        print(f"Budget:   {format_number(budget)}")
    if spend is not None:
        print(f"Spent:    {format_number(spend)}")
    if budget_number is not None and spend_number is not None:
        remaining = budget_number - spend_number
        print(f"Remaining:{format_number(remaining):>12}")
        if budget_number > 0:
            print(f"Used:     {(spend_number / budget_number) * 100:.2f}%")

    labels = {
        "soft_budget": "Soft budget",
        "budget_duration": "Budget period",
        "budget_reset_at": "Budget reset",
        "rpm_limit": "RPM limit",
        "tpm_limit": "TPM limit",
    }
    for field, label in labels.items():
        if values[field] is not None:
            print(f"{label}: {format_number(values[field])}")

    if budget is None and spend is None:
        print(
            "The request succeeded, but no recognized budget or spend fields were returned."
        )
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
