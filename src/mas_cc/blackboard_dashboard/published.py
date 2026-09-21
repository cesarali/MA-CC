"""A published study: the dashboard payloads computed once, read from a bundle.

``write_dashboard_bundle`` runs the ordinary ``BlackboardStudyReader`` over a finished study
and stores exactly what its API returns, so there is one implementation of every payload:

    index.json                              object table (bytes, sha256), written last
    study.json                              BlackboardStudyReader.study()
    cells/<qualified cell id>.json          .cell(id)
    cells/<qualified cell id>.prompts.json  .prompt_examples(id)
    analysis.json                           .analysis_catalog()
    analysis/<artifact id>                  the catalogued analysis files
    episodes/<qualified episode id>/...     the episode's run files, gzip-compressed, laid out
                                            so BlackboardRunReader opens the directory unchanged

``PublishedStudyReader`` serves the same public surface from a ``BundleStore``: opening the
study is two small reads, a cell one more, and an episode one download that stays cached.
Published studies are complete and immutable; live studies keep using the filesystem reader.
"""
from __future__ import annotations

import json
import shutil
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

from .data import BlackboardRunReader
from .store import BundleStore, StoreError, copy_gzipped, safe_relative, write_index
from .study_data import BlackboardStudyReader

EPISODE_READER_LIMIT = 8


def _cell_object(qualified_id: str, suffix: str = "") -> str:
    return safe_relative(f"cells/{qualified_id}{suffix}.json")


def write_dashboard_bundle(reader: BlackboardStudyReader, destination: str | Path, *,
                           include_episodes: bool = True, include_analysis_files: bool = True) -> dict[str, Any]:
    """Write the bundle for ``reader``'s study into ``destination`` (must be empty or absent)."""

    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"bundle destination is not empty: {destination.name}")
    destination.mkdir(parents=True, exist_ok=True)

    def dump(relative: str, payload: Any) -> None:
        path = destination / safe_relative(relative)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False, default=str),
                        encoding="utf-8")

    from .server import _finite  # the same NaN -> null rule the HTTP layer applies

    study = _finite(reader.study())
    dump("study.json", study)
    episodes = 0
    for cell in study["cells"]:
        qualified = cell["qualified_id"]
        payload = _finite(reader.cell(qualified))
        dump(_cell_object(qualified), payload)
        dump(_cell_object(qualified, ".prompts"), _finite(reader.prompt_examples(qualified)))
        if not include_episodes:
            continue
        try:
            paths = reader.resolved_paths(qualified)
        except ValueError:
            continue  # an expected cell that never started has no artifacts to publish
        for episode in payload["episodes"]:
            if not episode.get("detail_available"):
                continue
            target = destination / "episodes" / episode["qualified_id"]
            copied = 0
            for root in (paths.full_episodes_root, paths.round_records_root):
                source = root / episode["episode_id"]
                if not source.is_dir():
                    continue
                for path in sorted(p for p in source.rglob("*") if p.is_file()):
                    relative = path.relative_to(paths.run_root)
                    copy_gzipped(path, target / relative.parent / (relative.name + ".gz"))
                    copied += 1
            manifest = paths.run_root / "manifest.json"
            if copied and manifest.is_file():
                copy_gzipped(manifest, target / "manifest.json.gz")
            episodes += bool(copied)
    catalog = _finite(reader.analysis_catalog())
    dump("analysis.json", catalog)
    if include_analysis_files and catalog.get("available"):
        for artifact in catalog.get("artifacts", ()):
            source = reader.analysis_file(artifact["id"])
            target = destination / "analysis" / safe_relative(artifact["id"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return write_index(destination, study_id=str(study.get("study_id") or reader.study_dir.name),
                       extra={"kind": "blackboard_dashboard_bundle", "cells": len(study["cells"]),
                              "episodes_with_detail": episodes})


class PublishedStudyReader:
    """The study-reader surface the HTTP server uses, answered from a bundle."""

    def __init__(self, store: BundleStore) -> None:
        self.store = store
        if store.index.get("kind") != "blackboard_dashboard_bundle":
            raise StoreError("not a blackboard dashboard bundle")
        self.study_id = str(store.index.get("study_id"))
        self._lock = threading.RLock()
        self._episode_readers: OrderedDict[str, BlackboardRunReader] = OrderedDict()

    # -- study and cells -------------------------------------------------------------------
    def study(self) -> dict[str, Any]:
        return self.store.read_json("study.json")

    def cells(self) -> dict[str, Any]:
        return self.study()

    def cell(self, qualified_id: str) -> dict[str, Any]:
        name = self._cell_name(qualified_id)
        return self.store.read_json(name)

    def votes(self, qualified_id: str) -> dict[str, Any]:
        payload = self.cell(qualified_id)
        return {"schema_version": 1, "cell_id": qualified_id, "vote_series": payload["vote_series"],
                "descriptive_mean": payload["descriptive_mean"]}

    def prompt_examples(self, qualified_id: str) -> dict[str, Any]:
        self._cell_name(qualified_id)
        return self.store.read_json(_cell_object(qualified_id, ".prompts"))

    def _cell_name(self, qualified_id: str) -> str:
        try:
            name = _cell_object(qualified_id)
        except StoreError:
            raise ValueError("unknown qualified cell identifier") from None
        if not self.store.has(name):
            raise ValueError("unknown qualified cell identifier")
        return name

    # -- analysis --------------------------------------------------------------------------
    def analysis_catalog(self) -> dict[str, Any]:
        return self.store.read_json("analysis.json")

    def analysis_file(self, identifier: str) -> Path:
        allowed = {artifact["id"] for artifact in self.analysis_catalog().get("artifacts", ())}
        if identifier not in allowed:
            raise ValueError("unknown analysis artifact")
        return self.store.fetch_file(f"analysis/{identifier}")

    # -- episodes --------------------------------------------------------------------------
    def episode_status(self, qualified_id: str) -> dict[str, Any]:
        cell_id = qualified_id.rsplit("~episode-", 1)[0]
        try:
            payload = self.cell(cell_id)
        except ValueError:
            raise ValueError("unknown qualified episode identifier") from None
        for episode in payload["episodes"]:
            if episode["qualified_id"] == qualified_id:
                return {"schema_version": 1, **episode, "scheduler": payload.get("scheduler")}
        raise ValueError("unknown qualified episode identifier")

    def episode_reader(self, qualified_id: str) -> BlackboardRunReader:
        with self._lock:
            reader = self._episode_readers.get(qualified_id)
            if reader is not None:
                self._episode_readers.move_to_end(qualified_id)
                return reader
            status = self.episode_status(qualified_id)
            if not status["detail_available"]:
                raise ValueError(status["detail_reason"])
            try:
                run_root = self.store.fetch_dir(f"episodes/{qualified_id}")
            except StoreError:
                raise ValueError("episode detail was not published for this study") from None
            reader = BlackboardRunReader(run_root, status["episode_id"])
            self._episode_readers[qualified_id] = reader
            while len(self._episode_readers) > EPISODE_READER_LIMIT:
                self._episode_readers.popitem(last=False)
            return reader


__all__ = ["PublishedStudyReader", "write_dashboard_bundle"]
