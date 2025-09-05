#!/usr/bin/env python3
"""
Build per-layer (L5–L10) master results table from all_results.csv and per-run JSONs.

Policy:
- No layer averaging. One row per (language, probe, layer).
- Adds is_headline_layer (True for L7).
- Adds n_dev_sent and n_test_sent discovered from run artifacts.
- Leaves other covariates for subsequent steps.

Outputs:
- outputs/analysis/master_results_per_layer.csv
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from glob import glob
from typing import Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
ALL_RESULTS_PATH = os.path.join(REPO_ROOT, "outputs", "training_logs", "all_results.csv")
OUTPUT_DIR = os.path.join(REPO_ROOT, "outputs", "analysis")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "master_results_per_layer.csv")


@dataclass
class RunCounts:
    n_dev_sent: Optional[int]
    n_test_sent: Optional[int]


def safe_int(x: object) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def discover_run_dir(language_slug: str, probe: str, layer: str) -> Optional[str]:
    """Return a run directory path containing dev/test detailed metrics if possible."""
    base = os.path.join(
        REPO_ROOT,
        "outputs",
        "baselines_auto",
        language_slug,
        "bert-base-multilingual-cased",
        probe,
        layer,
        "runs",
    )
    if not os.path.isdir(base):
        return None
    # Prefer 'latest' symlink/dir
    latest = os.path.join(base, "latest")
    if os.path.exists(latest):
        return latest
    # Otherwise pick any run dir with test metrics (final or non-final) or dev metrics
    for rd in sorted(glob(os.path.join(base, "*"))):
        bn = os.path.basename(rd)
        if bn in {"latest", ".DS_Store"}:
            continue
        if not os.path.isdir(rd):
            continue
        test_json_final = os.path.join(rd, "test_detailed_metrics_final.json")
        test_json = os.path.join(rd, "test_detailed_metrics.json")
        dev_json = os.path.join(rd, "dev_detailed_metrics.json")
        if os.path.exists(test_json_final) or os.path.exists(test_json) or os.path.exists(dev_json):
            return rd
    return None


def read_counts_for_run(run_dir: str, probe: str) -> RunCounts:
    """Read n_dev_sent and n_test_sent from detailed metrics jsons in run_dir."""
    dev_json = os.path.join(run_dir, "dev_detailed_metrics.json")
    # Prefer final if present; otherwise use non-final
    test_json_final = os.path.join(run_dir, "test_detailed_metrics_final.json")
    test_json_nonfinal = os.path.join(run_dir, "test_detailed_metrics.json")
    test_json = test_json_final if os.path.exists(test_json_final) else test_json_nonfinal

    def load_json(path: str) -> Optional[dict]:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None

    def count_from_payload(payload: Optional[dict]) -> Optional[int]:
        if not payload:
            return None
        # Prefer explicit per-sentence metric arrays
        if probe == "dist":
            arr = payload.get("uuas_per_sentence")
        else:
            arr = payload.get("root_acc_per_sentence")
        if isinstance(arr, list):
            return safe_int(len(arr))
        # Fallback: some payloads may embed counts in debug sections
        dbg = payload.get("spearmanr_content_only_debug") if isinstance(payload, dict) else None
        if isinstance(dbg, dict):
            # Try a few likely keys
            for key in [
                "content_lengths",
                "content_lengths_for_sentences",
                "uuas_per_sentence",
                "root_acc_per_sentence",
            ]:
                maybe = dbg.get(key)
                if isinstance(maybe, list):
                    return safe_int(len(maybe))
        return None

    dev_payload = load_json(dev_json)
    test_payload = load_json(test_json)
    return RunCounts(
        n_dev_sent=count_from_payload(dev_payload),
        n_test_sent=count_from_payload(test_payload),
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    if not os.path.exists(ALL_RESULTS_PATH):
        raise FileNotFoundError(f"Missing all_results.csv at {ALL_RESULTS_PATH}")

    ensure_dir(OUTPUT_DIR)

    with open(ALL_RESULTS_PATH, newline="") as f:
        rdr = csv.DictReader(f)
        rows = list(rdr)

    # Normalize header names expected in all_results.csv
    def g(row: dict, key: str) -> str:
        return (row.get(key) or "").strip()

    out_rows: list[dict] = []
    for row in rows:
        language_slug = g(row, "Language")
        probe = g(row, "Probe")
        layer = g(row, "Layer")
        loss = g(row, "Loss")
        spearman_hm = g(row, "Spearman_HM")
        spearman_content = g(row, "Spearman_Content")
        uuas = g(row, "UUAS")
        root_acc = g(row, "Root_Acc")

        is_headline_layer = str(layer == "L7").lower()

        # Discover counts once per (language, probe, layer)
        n_dev_sent: Optional[int] = None
        n_test_sent: Optional[int] = None
        run_dir = discover_run_dir(language_slug, probe, layer)
        if run_dir:
            counts = read_counts_for_run(run_dir, probe)
            n_dev_sent = counts.n_dev_sent
            n_test_sent = counts.n_test_sent

        out_rows.append(
            {
                "language_slug": language_slug,
                "probe": probe,
                "layer": layer,
                "is_headline_layer": is_headline_layer,
                "loss": loss,
                "spearman_hm": spearman_hm,
                "spearman_content": spearman_content,
                "uuas": uuas,
                "root_acc": root_acc,
                "n_dev_sent": "" if n_dev_sent is None else str(n_dev_sent),
                "n_test_sent": "" if n_test_sent is None else str(n_test_sent),
            }
        )

    fieldnames = [
        "language_slug",
        "probe",
        "layer",
        "is_headline_layer",
        "loss",
        "spearman_hm",
        "spearman_content",
        "uuas",
        "root_acc",
        "n_dev_sent",
        "n_test_sent",
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    # Basic confirmation on stdout
    print(f"Wrote {len(out_rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()


