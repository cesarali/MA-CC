# Portable Potsdam WSL bridge

This folder is a copyable integration bundle for Python clients that must reach
the University of Potsdam LLM endpoint from WSL while the VPN is connected on
Windows.

The bridge is deliberately narrow:

- it runs only under WSL;
- it activates only for `llm.ki.k8s.rz.uni-potsdam.de`;
- the Windows listener binds to loopback;
- it permits HTTP `CONNECT` only to that host on port 443; and
- TLS and credentials remain end to end between the Python client and the API.

## Copy into another repository

Copy the `potsdam_wsl_bridge/` package into the receiving project's Python
package. For example:

```text
src/your_project/
├── llm_runtime/
│   └── providers/
└── potsdam_wsl_bridge/       # copy this directory here
    ├── __init__.py
    ├── bridge.py
    └── windows_connect_proxy.ps1
```

Also copy `tests/test_potsdam_wsl_bridge.py`, then replace its import
`from potsdam_wsl_bridge ...` with the receiving package's full import path,
such as `from your_project.potsdam_wsl_bridge ...`.

Ensure the PowerShell file is included in built wheels. With setuptools:

```toml
[tool.setuptools.package-data]
"your_project.potsdam_wsl_bridge" = ["*.ps1"]
```

Editable source checkouts may work without this declaration, so verify the
built wheel as well:

```bash
python -m build
unzip -l dist/*.whl | grep windows_connect_proxy.ps1
```

## Connect a `requests.Session`

Call the helper lazily, immediately after creating the session:

```python
from your_project.potsdam_wsl_bridge import ensure_windows_vpn_bridge

session = requests.Session()
proxy_url = ensure_windows_vpn_bridge(base_url)
if proxy_url is not None:
    session.proxies["https"] = proxy_url
```

For the vendored `UniversityProvider` used by `mas_cc` and `llm-pharmpy`, the
complete adapter method is:

```python
def _get_session(self):
    session = super()._get_session()
    if not self._windows_proxy_configured:
        from your_project.potsdam_wsl_bridge import ensure_windows_vpn_bridge

        proxy = ensure_windows_vpn_bridge(self._base_url)
        if proxy is not None:
            session.proxies["https"] = proxy
        self._windows_proxy_configured = True
    return session
```

Initialize `self._windows_proxy_configured = False` before calling the base
provider constructor.

## Runtime requirements

- Run under WSL with `powershell.exe` and `wslpath` available.
- Connect the university VPN on Windows.
- Use the exact Potsdam API hostname in the configured base URL.
- Prefer mirrored WSL networking on Windows 11 (`networkingMode=mirrored`,
  `dnsTunneling=true`, and `autoProxy=true`). Restart WSL after changing
  `%UserProfile%\.wslconfig`.
- Keep local TCP port 18765 free. An already-running healthy copy of this
  bridge is reused.

The unit tests do not contact the API and do not incur model usage. A real
completion should be a separate, explicitly authorized end-to-end test.

See `AGENT_HANDOFF.md` for a ready-to-send implementation request.
