"""Score System One's semantic attribution against human labels.

Inputs: the judged table (``semantic_attribution.parquet``: one row per message
and question) and one or two label sheets from ``semantic_label_sheet.py``.
Output: one JSON with, per question, agreement between model and humans and,
with two sheets, between the humans; plus a CSV of the per-message comparison.

    python scripts/Cygnus/analysis/semantic_agreement.py --judged semantic_attribution.parquet \
        --sheet labels-A.csv [--sheet labels-B.csv] --output agreement.json [--rows rows.csv]

Per question:
* noul (``cites_controller``, ``provides_evidence``): the model gives a value
  in [0, 1]; thresholds 0.05 … 0.95 are swept and accuracy, Cohen's kappa,
  precision and recall are reported for each, with the kappa-maximising one
  flagged. The human answer is yes/no.
* choice (``stance``): the model's label vs the human label; accuracy, kappa,
  and the confusion table.
* score (``pressure_toward_target``): the model's most probable level (index
  into PRESSURE_LEVELS) vs the human 0–3; exact accuracy, within-one accuracy,
  linear-weighted kappa and the confusion table.

Where two sheets are given, the human-human kappa is the ceiling to read the
model-human kappa against. No network calls; nothing here talks to System One.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from mas_cc.analysis.semantic_attribution import PRESSURE_LEVELS  # noqa: E402

KEY = ["episode_id", "message_id"]
THRESHOLDS = tuple(round(0.05 * k, 2) for k in range(1, 20))
YES = {"yes", "y", "true", "1"}
NO = {"no", "n", "false", "0"}


def cohen_kappa(a: list[Any], b: list[Any], *, weights: str | None = None, levels: list[Any] | None = None) -> float:
    """Cohen's kappa; ``weights='linear'`` for ordinal levels (given in order)."""
    if not a:
        return math.nan
    levels = levels or sorted(set(a) | set(b), key=str)
    index = {level: i for i, level in enumerate(levels)}
    k = len(levels)
    observed = np.zeros((k, k))
    for x, y in zip(a, b, strict=True):
        observed[index[x], index[y]] += 1
    n = observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / n
    if weights == "linear":
        w = np.abs(np.arange(k)[:, None] - np.arange(k)[None, :]) / max(k - 1, 1)
    else:
        w = 1.0 - np.eye(k)
    denominator = float((w * expected).sum())
    if denominator == 0:
        return math.nan
    return 1.0 - float((w * observed).sum()) / denominator


def _yes_no(value: Any) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip().lower()
    if text in YES:
        return True
    if text in NO:
        return False
    return None


def _level(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if text.isdigit() and 0 <= int(text) < len(PRESSURE_LEVELS):
        return int(text)
    for i, level in enumerate(PRESSURE_LEVELS):
        if text.lower() == level.split(":", 1)[0].strip().lower() or text.lower() == level.lower():
            return i
    return None


def _model_level(probabilities_json: Any, value: Any) -> int | None:
    if isinstance(probabilities_json, str) and probabilities_json:
        probabilities = json.loads(probabilities_json)
        if probabilities:
            best = max(probabilities, key=lambda key: probabilities[key])
            for i, level in enumerate(PRESSURE_LEVELS):
                if best == level or best == level.split(":", 1)[0].strip():
                    return i
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return int(round(float(value) * (len(PRESSURE_LEVELS) - 1)))


def load_sheets(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for index, path in enumerate(paths):
        sheet = pd.read_csv(path, dtype=str, keep_default_na=False)
        sheet["labeller_index"] = index
        frames.append(sheet)
    return pd.concat(frames, ignore_index=True)


def binary_report(model_value: pd.Series, human: pd.Series) -> dict[str, Any]:
    mask = human.notna() & model_value.notna()
    truth = human[mask].astype(bool).tolist()
    values = model_value[mask].astype(float).to_numpy()
    if not truth:
        return {"n": 0}
    sweep = []
    for threshold in THRESHOLDS:
        predicted = (values >= threshold).tolist()
        tp = sum(p and t for p, t in zip(predicted, truth, strict=True))
        fp = sum(p and not t for p, t in zip(predicted, truth, strict=True))
        fn = sum((not p) and t for p, t in zip(predicted, truth, strict=True))
        sweep.append({
            "threshold": threshold,
            "accuracy": float(np.mean([p == t for p, t in zip(predicted, truth, strict=True)])),
            "kappa": cohen_kappa(predicted, truth, levels=[False, True]),
            "precision": tp / (tp + fp) if tp + fp else math.nan,
            "recall": tp / (tp + fn) if tp + fn else math.nan,
        })
    best = max(sweep, key=lambda row: (-1.0 if math.isnan(row["kappa"]) else row["kappa"]))
    return {"n": len(truth), "human_positive_rate": float(np.mean(truth)), "model_mean_value": float(values.mean()),
            "best_threshold": best["threshold"], "best": best, "sweep": sweep}


def nominal_report(model_label: pd.Series, human: pd.Series, *, ordinal: bool = False,
                   levels: list[Any] | None = None) -> dict[str, Any]:
    mask = human.notna() & model_label.notna()
    a, b = model_label[mask].tolist(), human[mask].tolist()
    if not a:
        return {"n": 0}
    confusion = Counter(zip(b, a, strict=True))
    report = {
        "n": len(a),
        "accuracy": float(np.mean([x == y for x, y in zip(a, b, strict=True)])),
        "kappa": cohen_kappa(a, b, levels=levels),
        "confusion": [{"human": str(h), "model": str(m), "count": c} for (h, m), c in sorted(confusion.items(), key=str)],
    }
    if ordinal:
        report["within_one_accuracy"] = float(np.mean([abs(int(x) - int(y)) <= 1 for x, y in zip(a, b, strict=True)]))
        report["linear_weighted_kappa"] = cohen_kappa(a, b, weights="linear", levels=levels)
    return report


def agreement(judged: pd.DataFrame, sheets: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    def pivot(question: str, column: str) -> pd.Series:
        frame = judged[judged["question"] == question].drop_duplicates(KEY).set_index(KEY)[column]
        return frame

    sheets = sheets.copy()
    sheets["h_cites"] = sheets["human_cites_controller"].map(_yes_no)
    sheets["h_evidence"] = sheets["human_provides_evidence"].map(_yes_no)
    sheets["h_stance"] = sheets["human_stance"].map(lambda v: None if str(v).strip() == "" else str(v).strip())
    sheets["h_pressure"] = sheets["human_pressure"].map(_level)
    indexed = sheets.set_index(KEY)
    rows = indexed.copy()
    rows["m_cites"] = pivot("cites_controller", "value").reindex(indexed.index).to_numpy()
    rows["m_evidence"] = pivot("provides_evidence", "value").reindex(indexed.index).to_numpy()
    rows["m_stance"] = pivot("stance", "label").reindex(indexed.index).to_numpy()
    pressure = judged[judged["question"] == "pressure_toward_target"].drop_duplicates(KEY).set_index(KEY)
    rows["m_pressure"] = [
        _model_level(pressure["probabilities_json"].get(key), pressure["value"].get(key)) if key in pressure.index else None
        for key in indexed.index
    ]
    report: dict[str, Any] = {"messages_labelled": int(indexed.index.nunique()), "labellers": int(sheets["labeller_index"].nunique())}
    report["model_vs_human"] = {
        "cites_controller": binary_report(rows["m_cites"], rows["h_cites"]),
        "provides_evidence": binary_report(rows["m_evidence"], rows["h_evidence"]),
        "stance": nominal_report(rows["m_stance"], rows["h_stance"]),
        "pressure_toward_target": nominal_report(rows["m_pressure"], rows["h_pressure"], ordinal=True,
                                                 levels=list(range(len(PRESSURE_LEVELS)))),
    }
    if report["labellers"] >= 2:
        first = sheets[sheets["labeller_index"] == 0].set_index(KEY)
        second = sheets[sheets["labeller_index"] == 1].set_index(KEY)
        shared = first.index.intersection(second.index)
        a, b = first.loc[shared], second.loc[shared]
        report["human_vs_human"] = {
            "n": int(len(shared)),
            "cites_controller_kappa": cohen_kappa(a["h_cites"].tolist(), b["h_cites"].tolist()),
            "provides_evidence_kappa": cohen_kappa(a["h_evidence"].tolist(), b["h_evidence"].tolist()),
            "stance_kappa": cohen_kappa(a["h_stance"].tolist(), b["h_stance"].tolist()),
            "pressure_linear_weighted_kappa": cohen_kappa(a["h_pressure"].tolist(), b["h_pressure"].tolist(),
                                                          weights="linear", levels=list(range(len(PRESSURE_LEVELS)))),
        }
    return report, rows.reset_index()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judged", type=Path, required=True)
    parser.add_argument("--sheet", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=Path, default=None)
    args = parser.parse_args(argv)
    judged = pd.read_parquet(args.judged) if args.judged.suffix == ".parquet" else pd.read_csv(args.judged)
    report, rows = agreement(judged, load_sheets(args.sheet))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str) + "\n")
    if args.rows:
        rows.to_csv(args.rows, index=False)
    summary = {q: {k: v for k, v in r.items() if k in ("n", "accuracy", "kappa", "best_threshold", "linear_weighted_kappa")}
               for q, r in report["model_vs_human"].items()}
    print(json.dumps({"messages_labelled": report["messages_labelled"], "labellers": report["labellers"], **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
