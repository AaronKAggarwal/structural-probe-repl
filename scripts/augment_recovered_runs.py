#!/usr/bin/env python3
"""
Augment recovered run folders with checkpoint copy, W&B metadata, training history,
curves, and provenance manifests. Mirrors the selection/dedup logic of
`scripts/recover_jsons_from_wandb.py` so it targets the same canonical runs.

Outputs per run under:
  outputs/baselines_auto/{dataset}/{model}/{probe}/{layer}/runs/{run_id}/
Adds:
  - checkpoint_best.pt (copied from best_model artifact)
  - wandb_config.json, wandb_summary.json, wandb_history.csv
  - loss_vs_epoch.png, dev_loss_vs_epoch.png (if matplotlib available)
  - RUN_INFO.json (provenance), MANIFEST.json (file listing with sha256)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple
import time
from datetime import datetime

import wandb

REPO_ROOT = Path(__file__).resolve().parent.parent


Triple = Tuple[str, str, str]  # (dataset, probe, layer)


def _json_safe(obj):
    try:
        import numpy as _np  # type: ignore
    except Exception:
        _np = None  # type: ignore

    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    # numpy scalars
    if _np is not None and isinstance(obj, (_np.generic,)):
        try:
            return obj.item()
        except Exception:
            return float(obj) if hasattr(obj, "__float__") else str(obj)
    # mappings
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    # sequences
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    # W&B Summary/Config like objects
    if hasattr(obj, "to_dict"):
        try:
            return _json_safe(obj.to_dict())
        except Exception:
            pass
    # Fallback
    return str(obj)


def read_desired_triples(csv_path: Path) -> Set[Triple]:
    desired: Set[Triple] = set()
    if not csv_path.exists():
        return desired
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
    dataset = cfg.get('dataset', {}).get('name') or cfg.get('dataset.name') or cfg.get('dataset_name')
    probe_type = (cfg.get('probe', {}).get('type') or cfg.get('probe.type') or cfg.get('probe_type'))
    layer_index = cfg.get('embeddings', {}).get('layer_index') or cfg.get('embeddings.layer_index')
    model_slug = (
        cfg.get('embeddings', {}).get('source_model_name')
        or cfg.get('embeddings.source_model_name')
        or 'bert-base-multilingual-cased'
    )

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
    for coll in (run.logged_artifacts(), run.used_artifacts()):
        for art in coll:
            try:
                if art.type == 'model' and 'best_model' in art.name:
                    return art
            except Exception:
                continue
    return None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_manifest(root: Path) -> None:
    entries = []
    for p in sorted(root.rglob('*')):
        if p.is_file():
            try:
                entries.append({
                    'path': str(p.relative_to(root)),
                    'size': p.stat().st_size,
                    'sha256': sha256_file(p),
                })
            except Exception:
                continue
    (root / 'MANIFEST.json').write_text(json.dumps({'files': entries}, indent=2))


def write_run_info(root: Path, run: wandb.apis.public.Run, meta: Dict[str, object], art: Optional[wandb.apis.public.Artifact], ckpt_rel: Optional[str]) -> None:
    info = {
        'run_id': run.id,
        'run_name': getattr(run, 'name', None),
        'project': run.project,  # type: ignore[attr-defined]
        'entity': run.entity,    # type: ignore[attr-defined]
        'group': run.group,
        'state': str(getattr(run, 'state', '')),
        'created_at': str(getattr(run, 'created_at', '')),
        'updated_at': str(getattr(run, 'updated_at', '')),
        'job_type': getattr(run, 'jobType', None),
        'commit': getattr(run, 'commit', None),
        'dataset': meta.get('dataset'),
        'probe': meta.get('probe'),
        'layer': meta.get('layer'),
        'layer_index': meta.get('layer_index'),
        'model': meta.get('model'),
        'checkpoint_relpath': ckpt_rel,
        'artifact': {
            'name': getattr(art, 'name', None) if art else None,
            'type': getattr(art, 'type', None) if art else None,
            'version': getattr(art, 'version', None) if art else None,
            'digest': getattr(art, 'digest', None) if art else None,
        } if art else None,
    }
    (root / 'RUN_INFO.json').write_text(json.dumps(info, indent=2))


def export_wandb_metadata(root: Path, run: wandb.apis.public.Run) -> None:
    # Config and summary
    (root / 'wandb_config.json').write_text(json.dumps(_json_safe(run.config or {}), indent=2))
    try:
        # W&B summary is a special object; convert to plain dict then json-safe
        summ = _json_safe(dict(run.summary or {}))
    except Exception:
        summ = {}
    (root / 'wandb_summary.json').write_text(json.dumps(summ, indent=2))

    # History
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None  # type: ignore
    rows: Iterable[dict]
    try:
        rows = run.history(samples=100000, pandas=False)
    except Exception:
        rows = []
    fieldnames: Set[str] = set()
    buffered = []
    for r in rows:
        if isinstance(r, dict):
            buffered.append(r)
            fieldnames.update(r.keys())
    if buffered:
        # Write CSV
        fn = root / 'wandb_history.csv'
        with open(fn, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
            writer.writeheader()
            writer.writerows(buffered)
        # Optional plots if matplotlib available
        try:
            import matplotlib.pyplot as plt  # type: ignore
            if pd is None:
                # Minimal plotting using aggregated rows
                x = [r.get('_step') for r in buffered if '_step' in r]
                def plot_one(key: str, out: Path) -> None:
                    y = [r.get(key) for r in buffered if key in r and isinstance(r.get(key), (int, float))]
                    if x and y and len(y) == len(x):
                        plt.figure(figsize=(6, 3))
                        plt.plot(x, y)
                        plt.xlabel('step')
                        plt.ylabel(key)
                        plt.tight_layout()
                        plt.savefig(out)
                        plt.close()
                plot_one('train/loss', root / 'loss_vs_epoch.png')
                plot_one('dev/loss', root / 'dev_loss_vs_epoch.png')
            else:
                df = pd.DataFrame(buffered)
                def plot_df(col: str, out: Path) -> None:
                    if col in df.columns and '_step' in df.columns:
                        plt.figure(figsize=(6, 3))
                        plt.plot(df['_step'], df[col])
                        plt.xlabel('step')
                        plt.ylabel(col)
                        plt.tight_layout()
                        plt.savefig(out)
                        plt.close()
                for col, out in (
                    ('train/loss', root / 'loss_vs_epoch.png'),
                    ('dev/loss', root / 'dev_loss_vs_epoch.png'),
                ):
                    plot_df(col, out)
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True)
    ap.add_argument('--entity', required=False)
    ap.add_argument('--group', required=False)
    ap.add_argument('--csv', type=str, default=str(REPO_ROOT / 'outputs' / 'training_logs' / 'all_results.csv'))
    ap.add_argument('--runs_root', type=str, default=str(REPO_ROOT / 'outputs' / 'baselines_auto'))
    ap.add_argument('--copy_checkpoint', action='store_true', default=True)
    ap.add_argument('--no-copy_checkpoint', dest='copy_checkpoint', action='store_false')
    ap.add_argument('--export_wandb', action='store_true', default=True)
    ap.add_argument('--no-export_wandb', dest='export_wandb', action='store_false')
    ap.add_argument('--only_missing', action='store_true', default=True, help='If set, skip downloading/exporting items that already exist locally')
    ap.add_argument('--overwrite', dest='only_missing', action='store_false', help='If set, overwrite existing files')
    ap.add_argument('--progress_log', type=str, default=str(REPO_ROOT / 'outputs' / 'augment_progress.jsonl'))
    args = ap.parse_args()

    desired_triples = read_desired_triples(Path(args.csv))
    if not desired_triples:
        print(f"No desired triples found in {args.csv}. Proceeding without CSV filtering.")

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}") if args.entity else api.runs(args.project)
    runs = sorted(runs, key=lambda r: r.created_at or r.updated_at, reverse=True)

    seen: Set[Triple] = set()
    root = Path(args.runs_root)

    progress_path = Path(args.progress_log)
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    processed = 0
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
            continue
        meta = infer_from_run(run)
        if not meta:
            continue
        triple: Triple = (meta['dataset'], meta['probe'], meta['layer'])
        if desired_triples and triple not in desired_triples:
            continue
        if triple in seen:
            continue

        # Determine run directory per our recovery layout
        run_dir = root / meta['dataset'] / meta['model'] / meta['probe'] / meta['layer'] / 'runs' / run.id
        run_dir.mkdir(parents=True, exist_ok=True)

        ckpt_rel = None
        art = None
        dl_dur = None
        did_checkpoint = False
        skipped_checkpoint_exists = False
        ckpt_dst = run_dir / 'checkpoint_best.pt'
        need_checkpoint = args.copy_checkpoint and (not (args.only_missing and ckpt_dst.exists()))
        if args.copy_checkpoint and ckpt_dst.exists() and args.only_missing:
            # Already present; record relative path
            ckpt_rel = ckpt_dst.name
            skipped_checkpoint_exists = True
        if need_checkpoint:
            art = find_best_model_artifact(run)
            if art:
                dest_dir = REPO_ROOT / 'artifacts' / run.id
                dest_dir.mkdir(parents=True, exist_ok=True)
                t_dl = time.time()
                local_path = art.download(root=str(dest_dir))
                dl_dur = time.time() - t_dl
                # Copy first *.pt into run dir as checkpoint_best.pt
                ckpt_src = None
                for p in Path(local_path).rglob('*.pt'):
                    ckpt_src = p
                    break
                if ckpt_src:
                    try:
                        data = ckpt_src.read_bytes()
                        ckpt_dst.write_bytes(data)
                        ckpt_rel = ckpt_dst.name
                        did_checkpoint = True
                    except Exception:
                        pass

        # Metadata/plots export
        t_meta = None
        did_wandb_export = False
        skipped_metadata_exists = False
        meta_targets = [
            run_dir / 'wandb_config.json',
            run_dir / 'wandb_summary.json',
            run_dir / 'wandb_history.csv',
            run_dir / 'loss_vs_epoch.png',
            run_dir / 'dev_loss_vs_epoch.png',
        ]
        need_metadata = args.export_wandb and (not (args.only_missing and all(p.exists() for p in meta_targets)))
        if args.export_wandb and args.only_missing and all(p.exists() for p in meta_targets):
            skipped_metadata_exists = True
        if need_metadata:
            t_meta = time.time()
            export_wandb_metadata(run_dir, run)
            t_meta = time.time() - t_meta
            did_wandb_export = True

        write_run_info(run_dir, run, meta, art, ckpt_rel)
        write_manifest(run_dir)

        processed += 1
        seen.add(triple)
        log_event({
            'ts': datetime.utcnow().isoformat(),
            'event': 'augmented',
            'run_id': run.id,
            'dataset': meta['dataset'],
            'probe': meta['probe'],
            'layer': meta['layer'],
            'download_seconds': round(dl_dur, 3) if dl_dur is not None else None,
            'wandb_export_seconds': round(t_meta, 3) if t_meta is not None else None,
            'processed': processed,
            'did_checkpoint': did_checkpoint,
            'did_wandb_export': did_wandb_export,
            'skipped_checkpoint_exists': skipped_checkpoint_exists,
            'skipped_metadata_exists': skipped_metadata_exists,
        })


if __name__ == '__main__':
    main()


