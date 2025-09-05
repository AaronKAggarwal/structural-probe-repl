#!/usr/bin/env python3
"""
Summarize structural probe results across languages.

Reads `outputs/training_logs/all_results.csv` with columns:
- Language, Probe (dist|depth), Layer (e.g., L6), Loss, Spearman_HM, Spearman_Content, UUAS, Root_Acc

Outputs a TSV summary to stdout and writes to
`outputs/training_logs/summary.tsv` with, per language:
- best_UUAS (distance probe) and layer of the peak
- band mean UUAS over L6–L9
- band mean Spearman_HM over L6–L9 for distance probe
- best Root_Acc (depth probe) and layer of the peak
- band mean Root_Acc over L6–L9
- band mean Spearman_HM over L6–L9 for depth probe
Also prints overall averages for best_UUAS and band mean UUAS.
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from statistics import mean


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_INPUT = os.path.join(REPO_ROOT, 'outputs', 'training_logs', 'all_results.csv')
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, 'outputs', 'training_logs', 'summary.tsv')
SELECTION_JSON = os.path.join(REPO_ROOT, 'outputs', 'training_logs', 'syntax_band_selection.json')


def parse_float(value: str) -> float:
    if value is None or value == '':
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def load_rows(csv_path: str) -> list[dict]:
    rows: list[dict] = []
    with open(csv_path, newline='') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            row['UUAS'] = parse_float(row.get('UUAS'))
            row['Spearman_HM'] = parse_float(row.get('Spearman_HM'))
            row['Root_Acc'] = parse_float(row.get('Root_Acc'))
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> tuple[list[dict], dict]:
    dist_by_lang: dict[str, list[dict]] = defaultdict(list)
    depth_by_lang: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        probe = row.get('Probe', '').strip()
        lang = row.get('Language', '').strip()
        if probe == 'dist':
            dist_by_lang[lang].append(row)
        elif probe == 'depth':
            depth_by_lang[lang].append(row)

    languages = sorted(set(dist_by_lang.keys()) | set(depth_by_lang.keys()))
    # Default band; may be overridden if selection file exists
    band_layers_selected = {'L6', 'L7', 'L8', 'L9'}
    if os.path.exists(SELECTION_JSON):
        try:
            import json as _json
            with open(SELECTION_JSON, 'r') as f:
                sel = _json.load(f)
            selected = sel.get('selected_band', [])
            if selected and all(isinstance(L, str) for L in selected):
                band_layers_selected = set(selected)
        except Exception:
            pass

    summary_rows: list[dict] = []
    for lang in languages:
        drows = dist_by_lang.get(lang, [])
        prows = depth_by_lang.get(lang, [])

        best_dist = max(drows, key=lambda r: r['UUAS']) if drows else None
        dist_band = [r for r in drows if r.get('Layer') in band_layers_selected]
        band_mean_uuas = mean(r['UUAS'] for r in dist_band) if dist_band else math.nan
        band_mean_dist_spear = mean(r['Spearman_HM'] for r in dist_band) if dist_band else math.nan

        best_depth = max(prows, key=lambda r: r['Root_Acc']) if prows else None
        depth_band = [r for r in prows if r.get('Layer') in band_layers_selected]
        band_mean_root = mean(r['Root_Acc'] for r in depth_band) if depth_band else math.nan
        band_mean_depth_spear = mean(r['Spearman_HM'] for r in depth_band) if depth_band else math.nan

        summary_rows.append({
            'Language': lang,
            'best_UUAS': f"{best_dist['UUAS']:.4f}" if best_dist else '',
            'best_UUAS_layer': best_dist['Layer'] if best_dist else '',
            'bandUUAS_L6_9': f"{band_mean_uuas:.4f}" if not math.isnan(band_mean_uuas) else '',
            'bandDist_SpearmanHM': f"{band_mean_dist_spear:.4f}" if not math.isnan(band_mean_dist_spear) else '',
            'best_RootAcc': f"{best_depth['Root_Acc']:.4f}" if best_depth else '',
            'best_RootAcc_layer': best_depth['Layer'] if best_depth else '',
            'bandRootAcc_L6_9': f"{band_mean_root:.4f}" if not math.isnan(band_mean_root) else '',
            'bandDepth_SpearmanHM': f"{band_mean_depth_spear:.4f}" if not math.isnan(band_mean_depth_spear) else '',
        })

    # Sort by best_UUAS desc where available
    def sort_key(row: dict):
        try:
            return float(row['best_UUAS']) if row.get('best_UUAS') else -1.0
        except ValueError:
            return -1.0

    summary_rows.sort(key=sort_key, reverse=True)

    overall = {
        'mean_best_UUAS': f"{mean(float(r['best_UUAS']) for r in summary_rows if r.get('best_UUAS')):.4f}" if any(r.get('best_UUAS') for r in summary_rows) else '',
        'mean_bandUUAS_L6_9': f"{mean(float(r['bandUUAS_L6_9']) for r in summary_rows if r.get('bandUUAS_L6_9')):.4f}" if any(r.get('bandUUAS_L6_9') for r in summary_rows) else '',
    }

    return summary_rows, overall


def write_tsv(rows: list[dict], overall: dict, out_path: str) -> None:
    fieldnames = [
        'Language',
        'best_UUAS', 'best_UUAS_layer',
        'bandUUAS_L6_9', 'bandDist_SpearmanHM',
        'best_RootAcc', 'best_RootAcc_layer',
        'bandRootAcc_L6_9', 'bandDepth_SpearmanHM',
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        f.write('\t'.join(fieldnames) + '\n')
        for row in rows:
            f.write('\t'.join(row.get(k, '') for k in fieldnames) + '\n')
        # overall lines
        f.write('\n')
        f.write('\t'.join(['OVERALL', overall.get('mean_best_UUAS',''), '', overall.get('mean_bandUUAS_L6_9',''), '', '', '', '', '']) + '\n')


def main() -> None:
    rows = load_rows(DEFAULT_INPUT)
    summary_rows, overall = summarize(rows)

    # Print to stdout (TSV header + rows + overall)
    header = (
        'Language\tbest_UUAS\tlayer\tbandUUAS(L6-9)\tbandDistSpearmanHM\t'
        'bestRootAcc\trootLayer\tbandRootAcc(L6-9)\tbandDepthSpearmanHM'
    )
    print(header)
    for r in summary_rows:
        print('\t'.join([
            r['Language'],
            r['best_UUAS'], r['best_UUAS_layer'],
            r['bandUUAS_L6_9'], r['bandDist_SpearmanHM'],
            r['best_RootAcc'], r['best_RootAcc_layer'],
            r['bandRootAcc_L6_9'], r['bandDepth_SpearmanHM'],
        ]))
    print('\nOVERALL\tmean_best_UUAS\t{}\nOVERALL\tmean_band_UUAS(L6-9)\t{}'.format(
        overall.get('mean_best_UUAS',''), overall.get('mean_bandUUAS_L6_9','')
    ))

    # Write to file
    write_tsv(summary_rows, overall, DEFAULT_OUTPUT)


if __name__ == '__main__':
    main()


