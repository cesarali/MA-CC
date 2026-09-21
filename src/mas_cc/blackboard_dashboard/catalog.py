"""Several studies behind one server: published bundles listed by ``catalog.json`` and live study roots.

A source is either

* a published prefix - a directory or ``s3://`` / ``r2://`` URL holding the ``catalog.json`` that
  ``mas-cc study publish --with-dashboard`` maintains; each row names a bundle beside it;
* a directory whose children are live study roots (standardized studies or direct grids) on a
  filesystem the server can read.

Readers are created on first use and kept in a small LRU; the listing is refreshed at most every
``refresh_seconds``. Locations never leave the server: the API exposes keys, ids and counts only.
"""
from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

from .published import PublishedStudyReader
from .store import StoreError, child_location, is_store_url, open_store, read_raw
from .study_data import BlackboardStudyReader, is_direct_grid_root, is_study_root

CATALOG_NAME = "catalog.json"


class StudyCatalog:
    def __init__(self, sources: Iterable[str], *, reader_limit: int = 4, refresh_seconds: float = 30.0,
                 scheduler: bool = True, client: Any | None = None) -> None:
        self.sources = [str(source) for source in sources]
        if not self.sources:
            raise ValueError("a catalog needs at least one source")
        self._limit, self._refresh_seconds, self._scheduler, self._client = reader_limit, refresh_seconds, scheduler, client
        self._lock = threading.RLock()
        self._entries: dict[str, dict[str, Any]] = {}
        self._listed_at = 0.0
        self._readers: OrderedDict[str, Any] = OrderedDict()

    # -- listing ---------------------------------------------------------------------------
    def _published(self, source: str) -> list[dict[str, Any]]:
        raw = read_raw(source, CATALOG_NAME, client=self._client)
        if raw is None:
            return []
        try:
            catalog = json.loads(raw)
        except ValueError:
            raise StoreError("catalog.json is not valid JSON") from None
        rows = []
        for row in catalog.get("studies", []):
            if not isinstance(row, dict) or not row.get("study") or not row.get("dashboard"):
                continue
            rows.append({"key": str(row["study"]), "study_id": str(row.get("study_id") or row["study"]),
                         "source": "published", "cells": row.get("cells"), "episodes": row.get("episodes_with_detail"),
                         "published_at": row.get("published_at"), "_location": child_location(source, row["dashboard"])})
        return rows

    @staticmethod
    def _live(source: str) -> list[dict[str, Any]]:
        root = Path(source).expanduser()
        if not root.is_dir():
            return []
        rows = []
        for child in sorted(path for path in root.iterdir() if path.is_dir()):
            if is_study_root(child) or is_direct_grid_root(child):
                rows.append({"key": child.name, "study_id": child.name, "source": "live", "cells": None,
                             "episodes": None, "published_at": None, "_location": str(child.resolve())})
        return rows

    def _list(self) -> dict[str, dict[str, Any]]:
        entries: dict[str, dict[str, Any]] = {}
        for source in self.sources:
            rows = self._published(source) if is_store_url(source) or (Path(source).expanduser() / CATALOG_NAME).is_file() \
                else self._live(source)
            for row in rows:
                key = row["key"] if row["key"] not in entries else f"{row['key']}~{row['source']}"
                entries[key] = {**row, "key": key}
        return entries

    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            if not self._entries or time.monotonic() - self._listed_at > self._refresh_seconds:
                self._entries = self._list()
                self._listed_at = time.monotonic()
            return [{k: v for k, v in entry.items() if not k.startswith("_")} for entry in self._entries.values()]

    def payload(self) -> dict[str, Any]:
        return {"schema_version": 1, "studies": self.entries()}

    # -- readers ---------------------------------------------------------------------------
    def reader(self, key: str) -> Any:
        with self._lock:
            self.entries()
            entry = self._entries.get(key)
            if entry is None:
                raise ValueError("unknown study")
            reader = self._readers.get(key)
            if reader is None:
                location = entry["_location"]
                reader = (PublishedStudyReader(open_store(location, client=self._client)) if entry["source"] == "published"
                          else BlackboardStudyReader(location, scheduler=self._scheduler))
                self._readers[key] = reader
            self._readers.move_to_end(key)
            while len(self._readers) > self._limit:
                self._readers.popitem(last=False)
            return reader


__all__ = ["CATALOG_NAME", "StudyCatalog"]
