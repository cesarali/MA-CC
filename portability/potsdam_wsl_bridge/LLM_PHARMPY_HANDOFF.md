# Exact handoff for `llm-pharmpy`

Give the receiving agent this bundle and the following task:

> Replace the repository-root Potsdam bridge dependency with the package-local
> portable bundle. Copy `potsdam_wsl_bridge/` to
> `src/llm_pharmpy/harness/llm_runtime/potsdam_wsl_bridge/`. In
> `src/llm_pharmpy/harness/llm_runtime/providers/adapters/university.py`, import
> `ensure_windows_vpn_bridge` from
> `llm_pharmpy.harness.llm_runtime.potsdam_wsl_bridge`; retain the existing
> one-time session configuration and `session.proxies["https"]` assignment.
> Add the PowerShell resource to setuptools package data, adapt and install the
> supplied tests, build a wheel, and verify the wheel contains
> `windows_connect_proxy.ps1`. Do not make a paid LLM request.

The setuptools declaration for that repository is:

```toml
[tool.setuptools.package-data]
"llm_pharmpy.harness.llm_runtime.potsdam_wsl_bridge" = ["*.ps1"]
```

The adapter import should become:

```python
from llm_pharmpy.harness.llm_runtime.potsdam_wsl_bridge import (
    ensure_windows_vpn_bridge,
)
```

Run the copied tests and existing provider tests. Then build and inspect the
artifact:

```bash
uv run pytest tests/test_potsdam_wsl_bridge.py
uv build
unzip -l dist/*.whl | grep windows_connect_proxy.ps1
```

Once those checks pass, `scripts/Potsdam/windows_connect_proxy.ps1` is no
longer a runtime dependency. It may be removed, or retained only as clearly
marked historical documentation. Update `VENDORED_FROM.md` so future syncs
copy the package-local directory and its package-data rule.

Do not claim that the old manifest's “plus one script under scripts/” referred
to this bridge: that sentence described a Python caller whose imports were
rewritten. The actual source manifest incorrectly said the runtime directory
needed nothing else; this package-local bundle repairs that portability gap.
