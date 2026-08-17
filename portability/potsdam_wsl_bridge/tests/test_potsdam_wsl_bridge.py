from __future__ import annotations

from types import SimpleNamespace

import pytest

# After copying, replace this with the receiving project's full package path.
from potsdam_wsl_bridge import bridge


POTSDAM_URL = f"https://{bridge.POTSDAM_PROXY_HOST}/v1"


def test_packaged_powershell_resource_exists():
    assert bridge._bridge_script().is_file()


def test_noop_outside_wsl(monkeypatch):
    monkeypatch.setattr(bridge, "_is_wsl", lambda: False)
    assert bridge.ensure_windows_vpn_bridge(POTSDAM_URL) is None


def test_noop_for_any_other_host(monkeypatch):
    monkeypatch.setattr(bridge, "_is_wsl", lambda: True)
    assert bridge.ensure_windows_vpn_bridge("https://example.com/v1") is None


def test_reuses_an_existing_healthy_bridge_before_loading_resource(monkeypatch):
    monkeypatch.setattr(bridge, "_is_wsl", lambda: True)
    monkeypatch.setattr(bridge, "_bridge_is_ready", lambda port: True)
    monkeypatch.setattr(
        bridge,
        "_bridge_script",
        lambda: pytest.fail("resource should not be loaded when the bridge is healthy"),
    )

    assert bridge.ensure_windows_vpn_bridge(POTSDAM_URL) == "http://127.0.0.1:18765"


def test_rejects_invalid_port_under_wsl(monkeypatch):
    monkeypatch.setattr(bridge, "_is_wsl", lambda: True)
    with pytest.raises(bridge.PotsdamNetworkError, match="port is invalid"):
        bridge.ensure_windows_vpn_bridge(POTSDAM_URL, port=443)


def test_launch_command_is_restricted_and_requires_no_network(monkeypatch):
    monkeypatch.setattr(bridge, "_is_wsl", lambda: True)
    readiness = iter((False, True))
    monkeypatch.setattr(bridge, "_bridge_is_ready", lambda port: next(readiness))
    monkeypatch.setattr(
        bridge.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout=r"C:\bridge\windows_connect_proxy.ps1\n"),
    )

    launched = {}

    class FakeProcess:
        def __init__(self, command, **kwargs):
            launched["command"] = command

        def poll(self):
            return None

        def terminate(self):
            pass

    monkeypatch.setattr(bridge.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(bridge.atexit, "register", lambda callback: None)

    assert bridge.ensure_windows_vpn_bridge(POTSDAM_URL) == "http://127.0.0.1:18765"
    assert launched["command"][-4:] == [
        "-ListenPort",
        "18765",
        "-AllowedHost",
        bridge.POTSDAM_PROXY_HOST,
    ]
