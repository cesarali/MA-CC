"""WSL networking bridge for the University of Potsdam LLM proxy."""

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


def _find_bridge_script(repository_root: Path | None) -> Path:
    starts = [Path(__file__).resolve(), Path.cwd().resolve()]
    if repository_root is not None:
        starts.insert(0, repository_root.resolve())
    checked: set[Path] = set()
    for start in starts:
        directory = start if start.is_dir() else start.parent
        for candidate_root in (directory, *directory.parents):
            if candidate_root in checked:
                continue
            checked.add(candidate_root)
            candidate = (
                candidate_root / "scripts" / "Potsdam" / "windows_connect_proxy.ps1"
            )
            if candidate.is_file():
                return candidate
    raise PotsdamNetworkError(
        "Could not find scripts/Potsdam/windows_connect_proxy.ps1."
    )


def _stop_owned_process() -> None:
    global _OWNED_PROCESS
    process = _OWNED_PROCESS
    _OWNED_PROCESS = None
    if process is not None and process.poll() is None:
        process.terminate()


def ensure_windows_vpn_bridge(
    base_url: str,
    *,
    repository_root: Path | None = None,
    port: int = DEFAULT_BRIDGE_PORT,
) -> str | None:
    """Return a local HTTPS proxy URL when Potsdam traffic needs Windows.

    The bridge is used only for the known Potsdam proxy under WSL. TLS remains
    end-to-end between the Python HTTP client and the University endpoint; the
    Windows process only copies bytes after validating the CONNECT destination.
    """
    hostname = (urlsplit(base_url).hostname or "").lower()
    if not _is_wsl() or hostname != POTSDAM_PROXY_HOST:
        return None
    if not 1024 <= port <= 65535:
        raise PotsdamNetworkError("POTSDAM Windows bridge port is invalid.")

    with _BRIDGE_LOCK:
        if _bridge_is_ready(port):
            return f"http://127.0.0.1:{port}"

        script = _find_bridge_script(repository_root)
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
            "The Windows VPN bridge did not start. Check PowerShell execution "
            "policy and whether local port 18765 is already in use."
        )
