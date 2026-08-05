"""Restricted WSL-to-Windows bridge for the Potsdam VPN endpoint.

This ports the proven legacy networking behavior into the provider layer so the
new adapter remains operational without depending on ``naming_game``.
"""

from __future__ import annotations

import atexit
import os
import socket
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit


DEFAULT_BRIDGE_PORT = 18765
POTSDAM_PROXY_HOST = "llm.ki.k8s.rz.uni-potsdam.de"
_BRIDGE_LOCK = threading.Lock()
_OWNED_PROCESS: subprocess.Popen[bytes] | None = None


class PotsdamNetworkError(OSError):
    """Raised when the Windows VPN bridge cannot be prepared safely."""


def _is_wsl() -> bool:
    return "microsoft" in os.uname().release.lower()


def _bridge_is_ready(port: int) -> bool:
    request = (
        b"GET /__ma_cc_health__ HTTP/1.1\r\n"
        b"Host: localhost\r\nConnection: close\r\n\r\n"
    )
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5) as stream:
            stream.sendall(request)
            response = stream.recv(512)
    except OSError:
        return False
    return b"204 No Content" in response and b"MA-CC-Potsdam-Bridge" in response


def _find_bridge_script() -> Path:
    starts = [Path(__file__).resolve(), Path.cwd().resolve()]
    checked: set[Path] = set()
    for start in starts:
        directory = start if start.is_dir() else start.parent
        for root in (directory, *directory.parents):
            if root in checked:
                continue
            checked.add(root)
            candidate = root / "scripts" / "Potsdam" / "windows_connect_proxy.ps1"
            if candidate.is_file():
                return candidate
    raise PotsdamNetworkError("Could not find scripts/Potsdam/windows_connect_proxy.ps1.")


def _stop_owned_process() -> None:
    global _OWNED_PROCESS
    process = _OWNED_PROCESS
    _OWNED_PROCESS = None
    if process is not None and process.poll() is None:
        process.terminate()


def ensure_windows_vpn_bridge(base_url: str, *, port: int = DEFAULT_BRIDGE_PORT) -> str | None:
    """Return a local HTTPS proxy only for the known Potsdam host under WSL."""

    hostname = (urlsplit(base_url).hostname or "").lower()
    if not _is_wsl() or hostname != POTSDAM_PROXY_HOST:
        return None
    if not 1024 <= port <= 65535:
        raise PotsdamNetworkError("Potsdam Windows bridge port is invalid.")
    with _BRIDGE_LOCK:
        if _bridge_is_ready(port):
            return f"http://127.0.0.1:{port}"
        script = _find_bridge_script()
        try:
            converted = subprocess.run(
                ["wslpath", "-w", str(script)],
                check=True,
                capture_output=True,
                text=True,
            )
            process = subprocess.Popen(
                [
                    "powershell.exe",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    converted.stdout.strip(),
                    "-ListenPort",
                    str(port),
                    "-AllowedHost",
                    hostname,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise PotsdamNetworkError(
                "Could not launch the Windows VPN bridge from WSL."
            ) from exc
        global _OWNED_PROCESS
        _OWNED_PROCESS = process
        atexit.register(_stop_owned_process)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if _bridge_is_ready(port):
                return f"http://127.0.0.1:{port}"
            if process.poll() is not None:
                break
            time.sleep(0.1)
        _stop_owned_process()
        raise PotsdamNetworkError(
            "The Windows VPN bridge did not start. Check PowerShell policy and port 18765."
        )
