"""Optional client for TypeSafe System One (Jev) typed judgments.

System One is not a text generator. A request carries a ``state`` block plus a
map of typed questions and returns calibrated typed answers:

* ``noul``   - probability that a proposition holds, ``answer["noul"]`` in [0, 1];
* ``choice`` - one of N options with per-option ``probabilities`` and ``confidence``;
* ``score``  - a position on an ordered level scale with ``probabilities``.

Nothing in MA-CC calls this module unless a caller opts in explicitly (see
``semantic_attribution``). It exists so research instruments can ask bounded,
atomic questions about text the runs already saved without adding provider calls
to the numeric finalizer, which stays at ``provider_calls: 0``.

Endpoint and credentials come from the environment, never from configs:

* ``MA_CC_SYSTEMONE_URL``      - explicit endpoint; otherwise ``$LLM_BASE`` +
  ``/typesafe/v1/systemone`` (the in-cluster gateway the Slurm launchers export),
  otherwise the public ``https://llm.unsigned.gg`` pass-through. Spend is
  attributed to the caller's key either way;
* ``MA_CC_SYSTEMONE_KEY_FILE`` - file holding the bearer token (preferred), else
  ``LLM_KEY`` in the environment (the Slurm launchers already export it);
* ``MA_CC_SYSTEMONE_MODEL``    - default ``jev-latest``.

Every answered request is cached on disk under a content hash of
(model, state, questions), so re-running an instrument over the same saved text
is free and reproducible; the cache directory is the instrument's provenance.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PUBLIC_URL = "https://llm.unsigned.gg/typesafe/v1/systemone"
PASSTHROUGH_PATH = "/typesafe/v1/systemone"
DEFAULT_MODEL = "jev-latest"
USER_AGENT = "mas-cc-systemone/1 (+https://github.com/cesarali/MA-CC)"


def default_url() -> str:
    """Explicit override, else the in-cluster gateway the launchers export, else the public host.

    Slurm jobs on Cygnus run inside the cluster: ``LLM_BASE`` (set by every
    launcher) points at the gateway Service, which is both faster and not behind
    the public edge's browser checks (Cloudflare error 1010 for library agents).
    """
    explicit = os.environ.get("MA_CC_SYSTEMONE_URL")
    if explicit:
        return explicit
    base = os.environ.get("LLM_BASE", "").rstrip("/")
    if base:
        return base + PASSTHROUGH_PATH
    return PUBLIC_URL


DEFAULT_URL = PUBLIC_URL
RETRY_STATUSES = {429, 529, 502, 503, 504}


class SystemOneError(RuntimeError):
    """A System One request failed after retries or returned an invalid body."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def request_id(model: str, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> str:
    """Stable content hash of one request; the cache key and provenance id."""
    return hashlib.sha256(_canonical({"model": model, "state": state, "questions": questions}).encode()).hexdigest()


def noul(instructions: Any, *, true: str | None = None, false: str | None = None) -> dict[str, Any]:
    question: dict[str, Any] = {"type": "noul", "instructions": instructions}
    if true is not None or false is not None:
        question["criteria"] = {"true": true, "false": false}
    return question


def choice(instructions: Any, criteria: Mapping[str, str | None]) -> dict[str, Any]:
    if not criteria:
        raise ValueError("a choice question needs at least one option")
    return {"type": "choice", "instructions": instructions, "criteria": dict(criteria)}


def score(instructions: Any, levels: Sequence[str]) -> dict[str, Any]:
    if len(levels) < 2:
        raise ValueError("a score question needs at least two ordered levels")
    return {"type": "score", "instructions": instructions, "criteria": list(levels)}


@dataclass
class Usage:
    requests: int = 0
    cached: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return dict(vars(self))


@dataclass
class SystemOneClient:
    """Small, dependency-free HTTP client with retry and a content-hash cache."""

    url: str = field(default_factory=default_url)
    model: str = field(default_factory=lambda: os.environ.get("MA_CC_SYSTEMONE_MODEL", DEFAULT_MODEL))
    cache_dir: Path | None = None
    timeout: float = 60.0
    max_attempts: int = 5
    key: str | None = None
    usage: Usage = field(default_factory=Usage)
    transport: Any = None  # test seam: callable(payload: dict) -> dict

    def __post_init__(self) -> None:
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- credentials -------------------------------------------------------
    def _bearer(self) -> str:
        if self.key:
            return self.key
        key_file = os.environ.get("MA_CC_SYSTEMONE_KEY_FILE")
        if key_file:
            return Path(key_file).read_text(encoding="utf-8").strip()
        key = os.environ.get("LLM_KEY", "").strip()
        if not key:
            raise SystemOneError("no System One credential: set MA_CC_SYSTEMONE_KEY_FILE or LLM_KEY")
        return key

    # -- cache -------------------------------------------------------------
    def _cache_path(self, rid: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / rid[:2] / f"{rid}.json"

    def cached(self, rid: str) -> dict[str, Any] | None:
        path = self._cache_path(rid)
        if path is not None and path.is_file():
            with open(path, encoding="utf-8") as handle:
                return json.load(handle)
        return None

    def _store(self, rid: str, record: Mapping[str, Any]) -> None:
        path = self._cache_path(rid)
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, sort_keys=True, indent=1)
        os.replace(tmp, path)

    # -- transport ---------------------------------------------------------
    def _post(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        if self.transport is not None:
            return self.transport(dict(payload))
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=body, method="POST",
            headers={"content-type": "application/json", "authorization": f"Bearer {self._bearer()}",
                     "user-agent": USER_AGENT, "accept": "application/json"},
        )
        last: str = ""
        for attempt in range(1, self.max_attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                last = f"HTTP {exc.code}: {detail}"
                if exc.code not in RETRY_STATUSES or attempt == self.max_attempts:
                    raise SystemOneError(last) from None
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last = f"{type(exc).__name__}: {exc}"
                if attempt == self.max_attempts:
                    raise SystemOneError(last) from None
            self.usage.retries += 1
            time.sleep(min(30.0, 1.5 * 2 ** (attempt - 1)))
        raise SystemOneError(last or "unreachable")

    # -- public ------------------------------------------------------------
    def ask(self, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """Evaluate one state against several questions; returns the cached record.

        Record shape: ``{"request_id", "model", "state", "questions", "answers",
        "usage", "requested_at", "cached"}``. ``answers`` is the API's map keyed
        by the caller's question ids.
        """
        if not questions:
            raise ValueError("at least one question is required")
        for name, question in questions.items():
            if question.get("type") not in {"noul", "choice", "score"}:
                raise ValueError(f"question {name!r} has unsupported type {question.get('type')!r}")
        rid = request_id(self.model, state, questions)
        hit = self.cached(rid)
        if hit is not None:
            self.usage.cached += 1
            return {**hit, "cached": True}
        started = time.monotonic()
        response = self._post({"model": self.model, "state": state, "questions": dict(questions)})
        elapsed = time.monotonic() - started
        answers = response.get("answers")
        if not isinstance(answers, Mapping) or set(answers) != set(questions):
            raise SystemOneError(f"malformed System One response for {rid}: {str(response)[:300]}")
        usage = response.get("usage") or {}
        self.usage.requests += 1
        self.usage.input_tokens += int(usage.get("input_tokens") or 0)
        self.usage.output_tokens += int(usage.get("output_tokens") or 0)
        self.usage.seconds += elapsed
        record = {
            "request_id": rid, "model": response.get("model", self.model), "requested_model": self.model,
            "state": state, "questions": dict(questions), "answers": dict(answers),
            "usage": dict(usage), "seconds": elapsed,
            "requested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._store(rid, record)
        return {**record, "cached": False}


def flatten_answer(name: str, answer: Mapping[str, Any]) -> dict[str, Any]:
    """One flat row per answer, for tables: type, point value, confidence, probabilities JSON."""
    kind = answer.get("type")
    row: dict[str, Any] = {"question": name, "answer_type": kind}
    if kind == "noul":
        row.update(value=float(answer["noul"]), label=None, confidence=None, probabilities_json=None)
    elif kind == "choice":
        row.update(value=None, label=str(answer["choice"]), confidence=float(answer.get("confidence", float("nan"))),
                   probabilities_json=_canonical(answer.get("probabilities", {})))
    elif kind == "score":
        row.update(value=float(answer["score"]), label=None, confidence=float(answer.get("confidence", float("nan"))),
                   probabilities_json=_canonical(answer.get("probabilities", {})))
    else:
        raise SystemOneError(f"unknown answer type {kind!r} for {name}")
    return row


__all__ = ["SystemOneClient", "SystemOneError", "Usage", "choice", "default_url", "flatten_answer", "noul", "request_id", "score"]
