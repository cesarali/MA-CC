# Agent handoff: install the Potsdam WSL bridge

Use the accompanying `potsdam_wsl_bridge/` directory as the canonical copy.

1. Copy that directory under the receiving project's importable Python
   package; do not leave the PowerShell script in a repository-root `scripts/`
   directory.
2. Update `UniversityProvider` to call `ensure_windows_vpn_bridge(base_url)`
   once when its `requests.Session` is first obtained, and assign a returned
   URL to `session.proxies["https"]`.
3. Add the `.ps1` file to the build backend's package-data configuration.
4. Copy and adapt `tests/test_potsdam_wsl_bridge.py`. Run it without secrets or
   network access.
5. Build a wheel and inspect its contents to prove the `.ps1` resource was
   included. Testing only an editable install is insufficient.
6. Do not perform a paid completion unless the repository owner explicitly
   authorizes it. Bridge startup and the local health endpoint can be tested
   without an API request.

Preserve these security properties when making changes: WSL-only activation,
exact hostname matching, loopback-only listening, port-443-only `CONNECT`, no
TLS termination, and no credential handling in PowerShell.

If an older root-level `scripts/Potsdam/windows_connect_proxy.ps1` was added,
remove that dependency only after the package-local integration and wheel test
pass. Avoid maintaining two canonical copies.
