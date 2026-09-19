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
* ``MA_CC_SYSTEMONE_MODEL``    - default ``jev-1.13.0``. The version is pinned on
  purpose: ``jev-latest`` resolves to the same model today, but a re-run months
  later must be judged by the same model or the tables are not comparable;
* ``MA_CC_SYSTEMONE_CACHE``    - shared cache root; identical requests across
  users and runs are then served from disk instead of re-paid;
* ``MA_CC_SYSTEMONE_WORKERS``  - default concurrency for ``ask_many`` (8).

Every answered request is cached on disk under a content hash of
(model, state, questions), so re-running an instrument over the same saved text
is free and reproducible; the cache directory is the instrument's provenance.

Batching (``ask_many(batch_size>1)``) folds several states into one request.
It is NOT a free speedup: on 300 b9/b15 messages, four-per-request agreed with
single-message judgments on the stance label 82.7 % of the time and doubled the
mean pressure score (0.45 -> 0.98), the "distraction by irrelevant state"
failure mode the vendor documents. Concurrency alone (``workers``) changes
nothing about the answers and gave 6.8x (58 s -> 8.5 s at 8 workers, no errors
up to 32 in flight, ~50 requests/s). Default is ``batch_size=1``.

Costs: TypeSafe lists $42 per 1e9 input tokens; output tokens are assumed to
be priced the same (they are ~10 % of input for these questions). The
:class:`Budget` guard refuses to send a request whose projected total would
cross ``max_input_tokens`` or ``max_usd``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PUBLIC_URL = "https://llm.unsigned.gg/typesafe/v1/systemone"
PASSTHROUGH_PATH = "/typesafe/v1/systemone"
DEFAULT_MODEL = "jev-1.13.0"
USER_AGENT = "mas-cc-systemone/2 (+https://github.com/cesarali/MA-CC)"
USD_PER_INPUT_TOKEN = 42e-9
USD_PER_OUTPUT_TOKEN = 42e-9  # assumption; TypeSafe publishes the input price only
RETRY_STATUSES = {429, 529, 502, 503, 504}
MAX_BATCH = 8


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


def default_workers() -> int:
    return max(1, int(os.environ.get("MA_CC_SYSTEMONE_WORKERS", "8")))


DEFAULT_URL = PUBLIC_URL


class SystemOneError(RuntimeError):
    """A System One request failed after retries or returned an invalid body."""


class BudgetExceeded(SystemOneError):
    """Sending the next request would cross the configured budget."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def request_id(model: str, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> str:
    """Stable content hash of one request; the cache key and provenance id."""
    return hashlib.sha256(_canonical({"model": model, "state": state, "questions": questions}).encode()).hexdigest()


def estimate_tokens(payload: Any) -> int:
    """Cheap pre-send token estimate (measured ~3.6 bytes per input token on this workload)."""
    return max(1, len(_canonical(payload).encode("utf-8")) // 4)


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
class Budget:
    """Ceilings for one client's lifetime; ``None`` means unlimited."""

    max_input_tokens: int | None = None
    max_usd: float | None = None


@dataclass
class Usage:
    requests: int = 0
    batches: int = 0
    cached: int = 0
    deduplicated: int = 0  # identical items within one ask_many call, served by one request
    input_tokens: int = 0
    output_tokens: int = 0
    retries: int = 0
    seconds: float = 0.0

    @property
    def usd(self) -> float:
        return self.input_tokens * USD_PER_INPUT_TOKEN + self.output_tokens * USD_PER_OUTPUT_TOKEN

    def as_dict(self) -> dict[str, Any]:
        return {**vars(self), "usd": round(self.usd, 6)}


# -- batching helpers -----------------------------------------------------------

def _rebase_text(text: str, prefix: str, keys: Iterable[str]) -> str:
    """Prefix backticked state paths (`key…`) with ``prefix.`` so a question written
    for a single state still points at the right member of a batched state."""
    for key in sorted(keys, key=len, reverse=True):
        text = re.sub(rf"`{re.escape(key)}(?=[`.\[])", f"`{prefix}.{key}", text)
    return text


def _rebase(value: Any, prefix: str, keys: Iterable[str]) -> Any:
    if isinstance(value, str):
        return _rebase_text(value, prefix, keys)
    if isinstance(value, Mapping):
        return {k: _rebase(v, prefix, keys) for k, v in value.items()}
    if isinstance(value, list):
        return [_rebase(v, prefix, keys) for v in value]
    return value


def batch_payload(items: Sequence[tuple[Any, Mapping[str, Mapping[str, Any]]]]) -> tuple[dict[str, Any], dict[str, dict[str, Any]], list[list[str]]]:
    """Fold several (state, questions) pairs into one request.

    Returns ``(state, questions, question_names_per_item)``. Item ``i`` lives under
    state key ``m{i}``; its question ``q`` becomes ``m{i}__q`` with every backticked
    path rebased to ``m{i}.``. Only mapping states can be batched (a string state
    has no paths to rebase); callers fall back to single requests otherwise.
    """
    state: dict[str, Any] = {}
    questions: dict[str, dict[str, Any]] = {}
    names: list[list[str]] = []
    for index, (item_state, item_questions) in enumerate(items):
        if not isinstance(item_state, Mapping):
            raise ValueError("only mapping states can be batched")
        prefix = f"m{index}"
        state[prefix] = item_state
        keys = list(item_state.keys())
        names.append([])
        for name, question in item_questions.items():
            rebased = dict(_rebase(dict(question), prefix, keys))
            rebased["instructions"] = _rebase(question["instructions"], prefix, keys)
            if isinstance(rebased["instructions"], str):
                rebased["instructions"] = f"About `{prefix}` only: " + rebased["instructions"]
            questions[f"{prefix}__{name}"] = rebased
            names[-1].append(name)
    return state, questions, names


@dataclass
class SystemOneClient:
    """Small, dependency-free HTTP client with retry, a content-hash cache, a budget and a pool."""

    url: str = field(default_factory=default_url)
    model: str = field(default_factory=lambda: os.environ.get("MA_CC_SYSTEMONE_MODEL", DEFAULT_MODEL))
    cache_dir: Path | None = None
    timeout: float = 60.0
    max_attempts: int = 5
    key: str | None = None
    budget: Budget | None = None
    workers: int = field(default_factory=default_workers)
    usage: Usage = field(default_factory=Usage)
    transport: Any = None  # test seam: callable(payload: dict) -> dict

    def __post_init__(self) -> None:
        if self.cache_dir is None and os.environ.get("MA_CC_SYSTEMONE_CACHE"):
            self.cache_dir = Path(os.environ["MA_CC_SYSTEMONE_CACHE"])
        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

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
        tmp = path.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, sort_keys=True, indent=1)
        os.replace(tmp, path)

    # -- budget ------------------------------------------------------------
    def _reserve(self, payload: Mapping[str, Any]) -> None:
        """Refuse before sending if the projected spend would cross the budget."""
        if self.budget is None:
            return
        projected = estimate_tokens(payload)
        with self._lock:
            tokens = self.usage.input_tokens + projected
            usd = self.usage.usd + projected * USD_PER_INPUT_TOKEN
        if self.budget.max_input_tokens is not None and tokens > self.budget.max_input_tokens:
            raise BudgetExceeded(f"projected {tokens} input tokens exceeds max_input_tokens={self.budget.max_input_tokens}")
        if self.budget.max_usd is not None and usd > self.budget.max_usd:
            raise BudgetExceeded(f"projected ${usd:.4f} exceeds max_usd={self.budget.max_usd}")

    # -- transport ---------------------------------------------------------
    def _post(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        self._reserve(payload)
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
            with self._lock:
                self.usage.retries += 1
            time.sleep(min(30.0, 1.5 * 2 ** (attempt - 1)))
        raise SystemOneError(last or "unreachable")

    def _account(self, usage: Mapping[str, Any], elapsed: float, *, requests: int = 1, batches: int = 0) -> None:
        with self._lock:
            self.usage.requests += requests
            self.usage.batches += batches
            self.usage.input_tokens += int(usage.get("input_tokens") or 0)
            self.usage.output_tokens += int(usage.get("output_tokens") or 0)
            self.usage.seconds += elapsed

    @staticmethod
    def _validate(questions: Mapping[str, Mapping[str, Any]]) -> None:
        if not questions:
            raise ValueError("at least one question is required")
        for name, question in questions.items():
            if question.get("type") not in {"noul", "choice", "score"}:
                raise ValueError(f"question {name!r} has unsupported type {question.get('type')!r}")

    # -- public ------------------------------------------------------------
    def ask(self, state: Any, questions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
        """Evaluate one state against several questions; returns the cached record.

        Record shape: ``{"request_id", "model", "requested_model", "state",
        "questions", "answers", "usage", "seconds", "requested_at", "cached"}``.
        ``answers`` is the API's map keyed by the caller's question ids.
        """
        self._validate(questions)
        rid = request_id(self.model, state, questions)
        hit = self.cached(rid)
        if hit is not None:
            with self._lock:
                self.usage.cached += 1
            return {**hit, "cached": True}
        started = time.monotonic()
        response = self._post({"model": self.model, "state": state, "questions": dict(questions)})
        elapsed = time.monotonic() - started
        answers = response.get("answers")
        if not isinstance(answers, Mapping) or set(answers) != set(questions):
            raise SystemOneError(f"malformed System One response for {rid}: {str(response)[:300]}")
        usage = response.get("usage") or {}
        self._account(usage, elapsed)
        record = {
            "request_id": rid, "model": response.get("model", self.model), "requested_model": self.model,
            "state": state, "questions": dict(questions), "answers": dict(answers),
            "usage": dict(usage), "seconds": elapsed, "batch_size": 1,
            "requested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self._store(rid, record)
        return {**record, "cached": False}

    def _ask_batch(self, items: Sequence[tuple[str, Any, Mapping[str, Mapping[str, Any]]]]) -> list[dict[str, Any]]:
        """One HTTP request for several items; per-item records cached under their own ids."""
        state, questions, names = batch_payload([(s, q) for _, s, q in items])
        started = time.monotonic()
        response = self._post({"model": self.model, "state": state, "questions": questions})
        elapsed = time.monotonic() - started
        answers = response.get("answers")
        if not isinstance(answers, Mapping) or set(answers) != set(questions):
            raise SystemOneError(f"malformed System One batch response: {str(response)[:300]}")
        usage = response.get("usage") or {}
        self._account(usage, elapsed, requests=1, batches=1)
        share = {k: (int(v) / len(items) if isinstance(v, (int, float)) else v) for k, v in usage.items()}
        batch_id = hashlib.sha256(_canonical({"model": self.model, "state": state, "questions": questions}).encode()).hexdigest()
        records = []
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for index, (rid, item_state, item_questions) in enumerate(items):
            record = {
                "request_id": rid, "model": response.get("model", self.model), "requested_model": self.model,
                "state": item_state, "questions": dict(item_questions),
                "answers": {name: answers[f"m{index}__{name}"] for name in names[index]},
                "usage": share, "seconds": elapsed / len(items), "batch_size": len(items), "batch_id": batch_id,
                "requested_at": stamp,
            }
            self._store(rid, record)
            records.append({**record, "cached": False})
        return records

    def ask_many(self, items: Sequence[tuple[Any, Mapping[str, Mapping[str, Any]]]], *,
                 batch_size: int = 1, workers: int | None = None) -> list[dict[str, Any]]:
        """Evaluate many (state, questions) pairs concurrently; results in input order.

        Cache hits never leave the process. Misses are grouped ``batch_size`` at a
        time (mapping states only; ``batch_size=1`` sends one request per item) and
        the batches run on a thread pool of ``workers``. A ``BudgetExceeded`` from
        any request stops the pool and propagates.
        """
        if not 1 <= batch_size <= MAX_BATCH:
            raise ValueError(f"batch_size must be in 1..{MAX_BATCH}")
        pool_size = max(1, workers if workers is not None else self.workers)
        results: list[dict[str, Any] | None] = [None] * len(items)
        misses: list[tuple[int, str, Any, Mapping[str, Mapping[str, Any]]]] = []
        for index, (state, questions) in enumerate(items):
            self._validate(questions)
            rid = request_id(self.model, state, questions)
            hit = self.cached(rid)
            if hit is not None:
                with self._lock:
                    self.usage.cached += 1
                results[index] = {**hit, "cached": True}
            else:
                misses.append((index, rid, state, questions))
        # Identical misses within one call share a request.
        by_rid: dict[str, list[int]] = {}
        unique: list[tuple[str, Any, Mapping[str, Mapping[str, Any]]]] = []
        for index, rid, state, questions in misses:
            if rid not in by_rid:
                by_rid[rid] = []
                unique.append((rid, state, questions))
            else:
                with self._lock:
                    self.usage.deduplicated += 1
            by_rid[rid].append(index)
        batchable = batch_size > 1
        groups: list[list[tuple[str, Any, Mapping[str, Mapping[str, Any]]]]] = []
        singles: list[tuple[str, Any, Mapping[str, Mapping[str, Any]]]] = []
        for item in unique:
            if batchable and isinstance(item[1], Mapping):
                if not groups or len(groups[-1]) >= batch_size:
                    groups.append([])
                groups[-1].append(item)
            else:
                singles.append(item)

        def run_group(group: Sequence[tuple[str, Any, Mapping[str, Mapping[str, Any]]]]) -> list[dict[str, Any]]:
            if len(group) == 1:
                rid, state, questions = group[0]
                return [self.ask(state, questions)]
            return self._ask_batch(group)

        work: list[Sequence[tuple[str, Any, Mapping[str, Mapping[str, Any]]]]] = [*groups, *[[s] for s in singles]]
        if work:
            with ThreadPoolExecutor(max_workers=min(pool_size, len(work))) as pool:
                for records in pool.map(run_group, work):
                    for record in records:
                        for index in by_rid[record["request_id"]]:
                            results[index] = record
        return [r for r in results if r is not None] if all(r is not None for r in results) else [r or {} for r in results]


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


__all__ = ["Budget", "BudgetExceeded", "MAX_BATCH", "SystemOneClient", "SystemOneError", "Usage", "batch_payload",
           "choice", "default_url", "estimate_tokens", "flatten_answer", "noul", "request_id", "score"]
