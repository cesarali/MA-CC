"""Start a restricted Windows CONNECT bridge when called from WSL.

The companion PowerShell resource is package-local so the integration remains
usable from an installed wheel instead of depending on a repository layout.
"""

from __future__ import annotations

import atexit
import platform
import socket
import subprocess
import sys
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
    return sys.platform == "linux" and "microsoft" in platform.release().lower()


def _bridge_is_ready(port: int) -> bool:
    request = (
        b"GET /__potsdam_bridge_health__ HTTP/1.1\r\n"
        b"Host: localhost\r\nConnection: close\r\n\r\n"
    )
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5) as stream:
            stream.sendall(request)
            response = stream.recv(512)
    except OSError:
        return False
    return b"204 No Content" in response and b"Potsdam-WSL-Bridge" in response


def _bridge_script() -> Path:
    script = Path(__file__).with_name("windows_connect_proxy.ps1")
    if script.is_file():
        return script
    raise PotsdamNetworkError(
        "The packaged windows_connect_proxy.ps1 resource is missing. "
        "Include it as package data when building the project."
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
    port: int = DEFAULT_BRIDGE_PORT,
) -> str | None:
    """Return a local HTTPS-proxy URL only for the Potsdam host under WSL."""

    hostname = (urlsplit(base_url).hostname or "").lower()
    if not _is_wsl() or hostname != POTSDAM_PROXY_HOST:
        return None
    if not 1024 <= port <= 65535:
        raise PotsdamNetworkError("The Potsdam Windows bridge port is invalid.")

    with _BRIDGE_LOCK:
        if _bridge_is_ready(port):
            return f"http://127.0.0.1:{port}"

        script = _bridge_script()
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
        except (OSError, subprocess.CalledProcessError) as exc:
            raise PotsdamNetworkError(
                "Could not launch the Windows VPN bridge from WSL. "
                "Check that powershell.exe and wslpath are available."
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
            f"The Windows VPN bridge did not start on local port {port}. "
            "Check the PowerShell policy and whether the port is already occupied."
        )
