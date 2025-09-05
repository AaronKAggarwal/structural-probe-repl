#!/usr/bin/env python3
"""
Recover per-sentence detailed JSONs for selected W&B runs by downloading the best checkpoint
and running the eval-only script.

Requirements:
  pip install wandb

Usage:
  python scripts/recover_jsons_from_wandb.py --project structural-probes-modern --entity aggarwal-k-aaron \
      --group stage3-full-probe-test --limit 10 --csv outputs/training_logs/all_results.csv

This will:
  - list matching runs in W&B (filtered by project/entity/group)
  - (optional) restrict to triples from the CSV (Language, Probe, Layer)
  - for each triple, pick the newest matching run with a best-model artifact
  - download the 'best_model' artifact (checkpoint)
  - call scripts/evaluate_from_checkpoint.py to regenerate JSONs into per-language directories
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Optional, Set, Tuple
import time
import json
from datetime import datetime

import wandb

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = REPO_ROOT / 'scripts' / 'evaluate_from_checkpoint.py'

Triple = Tuple[str, str, str]  # (dataset, probe, layer)

def read_desired_triples(csv_path: Path) -> Set[Triple]:
    desired: Set[Triple] = set()
    if not csv_path.exists():
        return desired
    import csv
    with open(csv_path, newline='') as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            dataset = row.get('Language')
            probe = row.get('Probe')
            layer = row.get('Layer')
            if not dataset or not probe or not layer:
                continue
            desired.add((dataset.strip(), probe.strip(), layer.strip()))
    return desired


def infer_from_run(run: wandb.apis.public.Run) -> Optional[dict]:
    cfg = run.config or {}
    # Try to infer dataset slug, probe type, layer, model name
    dataset = cfg.get('dataset', {}).get('name') or cfg.get('dataset.name') or cfg.get('dataset_name')
    probe_type = (cfg.get('probe', {}).get('type') or cfg.get('probe.type') or cfg.get('probe_type'))
    layer_index = cfg.get('embeddings', {}).get('layer_index') or cfg.get('embeddings.layer_index')
    model_slug = cfg.get('embeddings', {}).get('source_model_name') or cfg.get('embeddings.source_model_name') or 'bert-base-multilingual-cased'

    # Normalize probe to 'dist'/'depth'
    if probe_type in ('distance', 'dist'):
        probe = 'dist'
    elif probe_type in ('depth',):
        probe = 'depth'
    else:
        return None

    if dataset is None or layer_index is None:
        return None

    layer = f"L{int(layer_index)}"
    return {
        'dataset': dataset,
        'probe': probe,
        'layer': layer,
        'layer_index': int(layer_index),
        'model': model_slug,
    }


def find_best_model_artifact(run: wandb.apis.public.Run) -> Optional[wandb.apis.public.Artifact]:
    # Look for an artifact with type 'model' and name containing 'best_model'
    # Prefer logged_artifacts (created by this run), then used_artifacts
    for coll in (run.logged_artifacts(), run.used_artifacts()):
        for art in coll:
            try:
                if art.type == 'model' and 'best_model' in art.name:
                    return art
            except Exception:
                continue
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True)
    ap.add_argument('--entity', required=False)
    ap.add_argument('--group', required=False)
    ap.add_argument('--limit', type=int, default=0, help='process at most N runs (0 = no limit)')
    ap.add_argument('--csv', type=str, default=str(REPO_ROOT / 'outputs' / 'training_logs' / 'all_results.csv'), help='Optional whitelist of (Language,Probe,Layer)')
    ap.add_argument('--progress_log', type=str, default=str(REPO_ROOT / 'outputs' / 'recovery_progress.jsonl'), help='Write JSON-lines progress here')
    args = ap.parse_args()

    desired_triples = read_desired_triples(Path(args.csv))
    if not desired_triples:
        print(f"No desired triples found in {args.csv}. Proceeding without CSV filtering.")

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}") if args.entity else api.runs(args.project)

    processed = 0
    seen: Set[Triple] = set()
    skipped_no_artifact = 0
    skipped_state = 0
    skipped_not_in_csv = 0
    errors = 0
    # Newest first
    runs = sorted(runs, key=lambda r: r.created_at or r.updated_at, reverse=True)
    total_candidates = len(runs)
    t0 = time.time()
    progress_path = Path(args.progress_log)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    def log_event(event: dict) -> None:
        try:
            with open(progress_path, 'a') as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass
    for run in runs:
        if args.group and run.group != args.group:
            continue
        if getattr(run, 'state', None) and str(run.state).lower() != 'finished':
            skipped_state += 1
            continue
        meta = infer_from_run(run)
        if not meta:
            continue
        triple: Triple = (meta['dataset'], meta['probe'], meta['layer'])
        if desired_triples and triple not in desired_triples:
            skipped_not_in_csv += 1
            continue
        if triple in seen:
            continue

        art = find_best_model_artifact(run)
        if not art:
            skipped_no_artifact += 1
            continue

        # Download artifact to a local dir
        dest_dir = REPO_ROOT / 'artifacts' / run.id
        dest_dir.mkdir(parents=True, exist_ok=True)
        start_download = time.time()
        local_path = art.download(root=str(dest_dir))
        dl_dur = time.time() - start_download
        # Heuristic: checkpoint file inside the artifact
        ckpt_path = None
        for p in Path(local_path).rglob('*.pt'):
            ckpt_path = p
            break
        if ckpt_path is None:
            continue

        # Call evaluator
        cmd = [
            'poetry', 'run', 'python', str(EVAL_SCRIPT),
            '--dataset', meta['dataset'],
            '--model', meta['model'],
            '--probe', meta['probe'],
            '--layer', meta['layer'],
            '--layer_index', str(meta['layer_index']),
            '--checkpoint', str(ckpt_path),
            '--run_id', run.id,
        ]
        idx = processed + 1
        print(f"[{idx}] {meta['dataset']} {meta['probe']} {meta['layer']} → run {run.id} | artifact {art.name} | dl {dl_dur:.2f}s", flush=True)
        print('Running:', ' '.join(cmd), flush=True)
        start_eval = time.time()
        try:
            subprocess.run(cmd, check=True)
            eval_dur = time.time() - start_eval
            elapsed = time.time() - t0
            seen.add(triple)
            processed += 1
            eta = None
            target = None
            if args.limit:
                target = args.limit
            elif desired_triples:
                target = len(desired_triples)
            if target:
                remaining = max(target - processed, 0)
                rate = processed / elapsed if elapsed > 0 else 0.0
                eta = (remaining / rate) if rate > 0 else None
            print(f"[{processed}] DONE {meta['dataset']} {meta['probe']} {meta['layer']} in {eval_dur:.2f}s | elapsed {elapsed/60:.1f}m" + (f" | ETA {eta/60:.1f}m" if eta else ''), flush=True)
            log_event({
                'ts': datetime.utcnow().isoformat(),
                'event': 'recovered',
                'run_id': run.id,
                'dataset': meta['dataset'],
                'probe': meta['probe'],
                'layer': meta['layer'],
                'artifact': getattr(art, 'name', None),
                'download_seconds': round(dl_dur, 3),
                'evaluate_seconds': round(eval_dur, 3),
                'processed': processed,
            })
        except subprocess.CalledProcessError as e:
            errors += 1
            print(f"[ERR] Evaluation failed for {meta['dataset']} {meta['probe']} {meta['layer']} run {run.id}: {e}", flush=True)
            log_event({
                'ts': datetime.utcnow().isoformat(),
                'event': 'error',
                'run_id': run.id,
                'dataset': meta['dataset'],
                'probe': meta['probe'],
                'layer': meta['layer'],
                'artifact': getattr(art, 'name', None),
                'error': str(e),
            })
        if args.limit and processed >= args.limit:
            break

    total_elapsed = time.time() - t0
    summary = {
        'ts': datetime.utcnow().isoformat(),
        'event': 'summary',
        'processed': processed,
        'skipped_no_artifact': skipped_no_artifact,
        'skipped_state': skipped_state,
        'skipped_not_in_csv': skipped_not_in_csv,
        'errors': errors,
        'elapsed_seconds': round(total_elapsed, 2),
        'total_candidates': total_candidates,
    }
    print(f"Summary: {summary}")
    try:
        with open(progress_path, 'a') as f:
            f.write(json.dumps(summary) + "\n")
    except Exception:
        pass

if __name__ == '__main__':
    main()
