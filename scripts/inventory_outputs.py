#!/usr/bin/env python3
"""
Inventory the current outputs:
- Verify aggregated CSV and count rows
- Summarize per-language logs in outputs/training_logs
- For baselines_auto (mBERT): for each probe and layer, report the dataset name found in train_probe.log,
  counts of dev/test detailed JSONs, and checkpoint counts. Infer likely overwriting.
"""

from __future__ import annotations

import csv
import json
import os
import re
from glob import glob
from typing import Dict, List


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUT_ROOT = os.path.join(REPO_ROOT, 'outputs')
TRAINING_LOGS = os.path.join(OUT_ROOT, 'training_logs')
ALL_RESULTS = os.path.join(TRAINING_LOGS, 'all_results.csv')
BASE_AUTO = os.path.join(OUT_ROOT, 'baselines_auto', 'bert-base-multilingual-cased')


def read_languages_from_csv(path: str) -> List[str]:
    langs = []
    if not os.path.exists(path):
        return langs
    with open(path, newline='') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            langs.append(row['Language'])
    return sorted(set(langs))


def parse_dataset_from_log(log_path: str) -> str | None:
    try:
        with open(log_path, 'r') as f:
            txt = f.read()
        m = re.search(r"dataset:\s*\n\s*name:\s*([^\n]+)", txt, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        m = re.search(r"Name\s*:\s*([^\n]+)", txt)
        if m:
            return m.group(1).strip()
    except Exception:
        return None
    return None


def inventory_baselines_auto() -> Dict:
    report: Dict = {
        'dist': {},
        'depth': {},
    }
    for probe in ('dist', 'depth'):
        probe_dir = os.path.join(BASE_AUTO, probe)
        for layer in ('L5', 'L6', 'L7', 'L8', 'L9', 'L10'):
            layer_dir = os.path.join(probe_dir, layer)
            if not os.path.isdir(layer_dir):
                continue
            log_path = os.path.join(layer_dir, 'train_probe.log')
            dataset = parse_dataset_from_log(log_path) if os.path.exists(log_path) else None
            dev_details = sorted(glob(os.path.join(layer_dir, 'dev_detailed_metrics_epoch*.json')))
            test_details = sorted(glob(os.path.join(layer_dir, 'test_detailed_metrics_final.json')))
            ckpts = sorted(glob(os.path.join(layer_dir, 'checkpoints', '*.pt')))
            report[probe][layer] = {
                'dataset_in_log': dataset,
                'num_dev_detailed_json': len(dev_details),
                'num_test_detailed_json': len(test_details),
                'num_checkpoints': len(ckpts),
                'layer_dir': layer_dir,
            }
    return report


def main() -> None:
    print('=== Aggregated results ===')
    if os.path.exists(ALL_RESULTS):
        with open(ALL_RESULTS, 'r') as f:
            num_lines = sum(1 for _ in f)
        langs = read_languages_from_csv(ALL_RESULTS)
        print(f"all_results.csv: present ({num_lines} lines). Unique languages: {len(langs)}")
        print(f"Languages: {', '.join(langs)}")
    else:
        print('all_results.csv: MISSING')

    print('\n=== Training logs (outputs/training_logs) ===')
    logs = sorted(glob(os.path.join(TRAINING_LOGS, '*_L[5-9]_ep*.log'))) + \
           sorted(glob(os.path.join(TRAINING_LOGS, '*_L10_ep*.log')))
    print(f"Per-language logs found: {len(logs)} (examples)\n  - {logs[:3]}\n  - ...\n  - {logs[-3:]}" if logs else 'No per-language logs found')

    print('\n=== baselines_auto (mBERT) current contents per probe/layer ===')
    ba = inventory_baselines_auto()
    for probe in ('dist', 'depth'):
        print(f"\n[{probe}]")
        for layer in ('L5', 'L6', 'L7', 'L8', 'L9', 'L10'):
            info = ba[probe].get(layer)
            if not info:
                print(f"  {layer}: (missing)")
                continue
            ds = info['dataset_in_log'] or 'UNKNOWN'
            print(
                f"  {layer}: dataset_in_log={ds}; dev_json={info['num_dev_detailed_json']}; "
                f"test_json={info['num_test_detailed_json']}; ckpts={info['num_checkpoints']}"
            )

    # Overwrite inference: since each layer dir holds at most one dataset, all other languages' detailed jsons/checkpoints
    # that previously wrote to the same dir would have been overwritten.
    print('\n=== Likely overwritten (inferred) ===')
    print('- Dev/test detailed JSONs and best checkpoints for languages that wrote earlier to the same probe/layer directories.')
    print('- Reason: hydra.run.dir did not include language slug, so later runs replaced prior artifacts in those shared layer folders.')


if __name__ == '__main__':
    main()


