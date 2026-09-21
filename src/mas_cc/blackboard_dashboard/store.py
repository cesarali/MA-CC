"""Read-only access to a published dashboard bundle, on disk or in an object store.

A bundle is a small tree of immutable objects described by ``index.json`` (relative path,
bytes and sha256 of every other object).

``open_store`` returns a ``BundleStore`` over one of two backends:

* a directory that holds ``index.json``;
* ``s3://bucket/prefix`` or ``r2://bucket/prefix`` through boto3, which is an optional
  dependency (``pip install mas-cc[r2]``). Endpoint and credentials come from the
  environment only: ``MA_CC_R2_ENDPOINT`` or ``AWS_ENDPOINT_URL`` and the standard ``AWS_*``
  variables. Nothing is ever written to the store.

Objects are materialised into a byte-bounded cache directory (``MA_CC_DASHBOARD_CACHE``,
default under the system temp dir) and validated against the index hash before use, so a
truncated or swapped object is an error, never a silently different dashboard. Objects whose
name ends in ``.gz`` are stored compressed and appear decompressed in the cache.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlparse

INDEX_NAME = "index.json"
BUNDLE_SCHEMA_VERSION = 1
DEFAULT_CACHE_BYTES = 5 * 1024**3
OBJECT_SCHEMES = frozenset({"s3", "r2"})


class StoreError(ValueError):
    """The bundle is missing, malformed, or an object failed validation."""


def safe_relative(relative: str) -> str:
    """Normalise a bundle-relative path; refuse anything that could leave the bundle."""

    text = str(relative)
    path = PurePosixPath(text)
    if (path.is_absolute() or not path.parts or "//" in text or text.endswith("/")
            or any(part in {"", ".", ".."} for part in text.split("/"))):
        raise StoreError(f"unsafe bundle path: {relative!r}")
    return path.as_posix()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Transport(Protocol):
    """The two operations a store needs from its backend."""

    def get(self, relative: str) -> bytes: ...

    def describe(self) -> str: ...


class _DirectoryTransport:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).expanduser().resolve()

    def get(self, relative: str) -> bytes:
        path = self.root / safe_relative(relative)
        if not path.is_file():
            raise StoreError(f"bundle object is missing: {relative}")
        return path.read_bytes()

    def describe(self) -> str:
        return self.root.name


class _ObjectTransport:
    def __init__(self, bucket: str, prefix: str, client: Any | None = None) -> None:
        self.bucket, self.prefix = bucket, prefix.strip("/")
        self._client = client

    def _s3(self) -> Any:
        if self._client is None:
            try:
                import boto3  # optional: only object-store dashboards need it
            except ImportError as error:  # pragma: no cover - exercised by message only
                raise StoreError("reading s3:// or r2:// bundles needs boto3 (pip install 'mas-cc[r2]')") from error
            endpoint = os.environ.get("MA_CC_R2_ENDPOINT") or os.environ.get("AWS_ENDPOINT_URL")
            self._client = boto3.client("s3", endpoint_url=endpoint or None,
                                        region_name=os.environ.get("AWS_DEFAULT_REGION", "auto"))
        return self._client

    def get(self, relative: str) -> bytes:
        key = "/".join(part for part in (self.prefix, safe_relative(relative)) if part)
        client = self._s3()  # outside the try: "boto3 is not installed" must reach the user as itself
        try:
            return client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except Exception as error:  # botocore raises its own hierarchy; keep keys out of messages
            code = getattr(error, "response", {}).get("Error", {}).get("Code", type(error).__name__)
            raise StoreError(f"bundle object could not be read ({code}): {relative}") from None

    def describe(self) -> str:
        return f"{self.bucket}/{self.prefix}".rstrip("/")


class BundleStore:
    """Index-validated, cached, read-only view of one published bundle."""

    def __init__(self, transport: Transport, *, cache_dir: str | Path | None = None,
                 cache_limit_bytes: int | None = None) -> None:
        self._transport = transport
        self._lock = threading.RLock()
        base = Path(cache_dir or os.environ.get("MA_CC_DASHBOARD_CACHE")
                    or Path(tempfile.gettempdir()) / "mas-cc-dashboard-cache")
        identity = hashlib.sha256(transport.describe().encode()).hexdigest()[:16]
        self.cache_dir = base.expanduser() / identity
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_limit_bytes = int(cache_limit_bytes if cache_limit_bytes is not None
                                     else os.environ.get("MA_CC_DASHBOARD_CACHE_BYTES", DEFAULT_CACHE_BYTES))
        self._cached: OrderedDict[str, int] = OrderedDict()  # relative -> bytes on disk, LRU order
        self.requests = 0  # backend GETs, for tests and the access log
        raw = self._get(INDEX_NAME)
        try:
            self.index: dict[str, Any] = json.loads(raw)
        except ValueError as error:
            raise StoreError("bundle index is not valid JSON") from error
        if int(self.index.get("schema_version", 0)) != BUNDLE_SCHEMA_VERSION:
            raise StoreError(f"unsupported bundle schema version: {self.index.get('schema_version')}")
        objects = self.index.get("objects")
        if not isinstance(objects, Mapping):
            raise StoreError("bundle index lists no objects")
        self.objects: dict[str, Mapping[str, Any]] = {safe_relative(k): v for k, v in objects.items()}

    def describe(self) -> str:
        return self._transport.describe()

    def _get(self, relative: str) -> bytes:
        with self._lock:
            self.requests += 1
        return self._transport.get(relative)

    def _validated(self, relative: str) -> bytes:
        relative = safe_relative(relative)
        entry = self.objects.get(relative)
        if entry is None:
            raise StoreError(f"not part of this bundle: {relative}")
        data = self._get(relative)
        if len(data) != int(entry["bytes"]) or sha256_bytes(data) != entry["sha256"]:
            raise StoreError(f"bundle object failed validation: {relative}")
        return data

    def has(self, relative: str) -> bool:
        return safe_relative(relative) in self.objects

    def read_json(self, relative: str) -> Any:
        return json.loads(self.fetch_file(relative).read_bytes())

    def fetch_file(self, relative: str, *, keep_prefix: str | None = None) -> Path:
        """Materialise one object; ``name.gz`` objects land decompressed as ``name``.

        ``keep_prefix`` protects everything under that prefix from eviction while a directory
        is being opened, however small the cache limit is.
        """

        relative = safe_relative(relative)
        target_relative = relative[:-3] if relative.endswith(".gz") else relative
        target = self.cache_dir / target_relative
        with self._lock:
            if relative in self._cached and target.is_file():
                self._cached.move_to_end(relative)
                return target
        data = self._validated(relative)
        if relative.endswith(".gz"):
            data = gzip.decompress(data)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + f".partial-{threading.get_ident()}")
        partial.write_bytes(data)
        partial.replace(target)
        with self._lock:
            self._cached[relative] = len(data)
            self._cached.move_to_end(relative)
            self._evict(keep=relative, keep_prefix=keep_prefix)
        return target

    def fetch_dir(self, prefix: str) -> Path:
        """Materialise every object under ``prefix/``; returns the local directory."""

        prefix = safe_relative(prefix)
        members = [name for name in self.objects if name.startswith(prefix + "/")]
        if not members:
            raise StoreError(f"bundle has nothing under: {prefix}")
        for name in members:
            self.fetch_file(name, keep_prefix=prefix + "/")
        return self.cache_dir / prefix

    def _evict(self, keep: str, keep_prefix: str | None = None) -> None:
        total = sum(self._cached.values())
        for name in list(self._cached):  # least recently used first
            if total <= self.cache_limit_bytes:
                break
            if name == keep or (keep_prefix and name.startswith(keep_prefix)):
                continue
            size = self._cached.pop(name)
            total -= size
            victim = self.cache_dir / (name[:-3] if name.endswith(".gz") else name)
            victim.unlink(missing_ok=True)


def is_store_url(value: str | os.PathLike[str]) -> bool:
    return urlparse(str(value)).scheme in OBJECT_SCHEMES


def open_store(location: str | os.PathLike[str], *, client: Any | None = None, **options: Any) -> BundleStore:
    """``s3://bucket/prefix`` / ``r2://bucket/prefix`` or a directory holding ``index.json``."""

    parsed = urlparse(str(location))
    if parsed.scheme in OBJECT_SCHEMES:
        if not parsed.netloc:
            raise StoreError(f"object-store URL needs a bucket: {location}")
        return BundleStore(_ObjectTransport(parsed.netloc, parsed.path, client), **options)
    return BundleStore(_DirectoryTransport(Path(str(location))), **options)


def is_bundle_dir(path: str | os.PathLike[str]) -> bool:
    return (Path(str(path)).expanduser() / INDEX_NAME).is_file()


def write_index(destination: Path, *, study_id: str, extra: Mapping[str, Any] | None = None,
                clock: Callable[[], str] | None = None) -> dict[str, Any]:
    """Hash every file under ``destination`` and write ``index.json`` last."""

    from datetime import datetime, timezone

    destination = Path(destination)
    objects = {}
    for path in sorted(p for p in destination.rglob("*") if p.is_file() and p.name != INDEX_NAME):
        data = path.read_bytes()
        objects[path.relative_to(destination).as_posix()] = {"bytes": len(data), "sha256": sha256_bytes(data)}
    index = {"schema_version": BUNDLE_SCHEMA_VERSION, "study_id": study_id,
             "generated_at": (clock or (lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")))(),
             **dict(extra or {}), "objects": objects}
    (destination / INDEX_NAME).write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return index


def copy_gzipped(source: Path, target: Path) -> None:
    """Deterministic gzip (mtime 0) so republishing unchanged data yields identical objects."""

    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as raw, target.open("wb") as sink:
        with gzip.GzipFile(filename="", mode="wb", fileobj=sink, mtime=0, compresslevel=6) as packed:
            shutil.copyfileobj(raw, packed, 1024 * 1024)


__all__ = ["BUNDLE_SCHEMA_VERSION", "BundleStore", "INDEX_NAME", "StoreError", "copy_gzipped", "is_bundle_dir",
           "is_store_url", "open_store", "safe_relative", "sha256_bytes", "write_index"]
