#!/usr/bin/env python3
"""
Evaluate a saved probe checkpoint on dev and test to regenerate detailed per-sentence JSONs.

Usage example:
  python scripts/evaluate_from_checkpoint.py \
    --dataset UD_English-EWT \
    --model bert-base-multilingual-cased \
    --probe dist \
    --layer L7 \
    --checkpoint /path/to/distance_probe_rank128_best.pt \
    --run_id wandb_run_ABC123

It will write outputs to:
  outputs/baselines_auto/{dataset}/{model}/{probe}/{layer}/runs/{run_id}/
    - metrics_summary.json
    - dev_detailed_metrics.json
    - test_detailed_metrics.json

Notes:
  - Requires embeddings HDF5 and CoNLL paths from configs/dataset/{dataset}/{dataset}.yaml
  - Uses content-only evaluation and UPOS punctuation filtering to match main runs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
import sys, os

# Ensure repository root is on sys.path for `src.*` imports when running as a script
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from src.torch_probe.dataset import ProbeDataset, collate_probe_batch
from src.torch_probe.evaluate import evaluate_probe
from src.torch_probe.loss_functions import depth_l1_loss, distance_l1_loss
from src.torch_probe.probe_models import DepthProbe, DistanceProbe
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parent.parent


def load_dataset_paths(dataset_slug: str) -> Dict[str, str]:
    cfg_path = REPO_ROOT / 'configs' / 'dataset' / dataset_slug / f'{dataset_slug}.yaml'
    if not cfg_path.exists():
        raise FileNotFoundError(f'Dataset config not found: {cfg_path}')
    cfg = OmegaConf.load(str(cfg_path))
    paths = cfg.get('paths', {})
    return {
        'train': str((REPO_ROOT / paths['conllu_train']).resolve()),
        'dev': str((REPO_ROOT / paths['conllu_dev']).resolve()),
        'test': str((REPO_ROOT / paths['conllu_test']).resolve()),
    }


def make_output_dir(dataset_slug: str, model_slug: str, probe: str, layer: str, run_id: str) -> Path:
    base = REPO_ROOT / 'outputs' / 'baselines_auto' / dataset_slug / model_slug / probe / layer / 'runs' / run_id
    base.mkdir(parents=True, exist_ok=True)
    # Write sentinel
    sentinel = base.parent.parent.parent.parent / '.dataset_slug'
    if not sentinel.exists():
        sentinel.write_text(dataset_slug)
    # Update latest symlink
    latest = base.parent / 'latest'
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
        latest.symlink_to(base.name)
    except Exception:
        pass
    return base


def load_probe_from_checkpoint(checkpoint_path: str, embedding_dim: int, probe: str, rank: int = 128, device: torch.device | None = None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    if probe == 'dist':
        model = DistanceProbe(embedding_dim=embedding_dim, probe_rank=rank)
    else:
        model = DepthProbe(embedding_dim=embedding_dim, probe_rank=rank)
    state = torch.load(checkpoint_path, map_location=device)
    # Handle checkpoints saved via save_checkpoint(...)
    if isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.to(device)
    return model, device


def run_eval_split(probe_model, loss_fn, device, probe_type: str, conllu_path: str, hdf5_path: str, layer_index: int) -> Dict[str, Any]:
    ds = ProbeDataset(
        conllu_filepath=conllu_path,
        hdf5_filepath=hdf5_path,
        embedding_layer_index=layer_index,
        probe_task_type='distance' if probe_type == 'dist' else 'depth',
        preload=False,
        collapse_punct=False,
    )
    loader = DataLoader(ds, batch_size=20, collate_fn=collate_probe_batch, shuffle=False)
    metrics = evaluate_probe(
        probe_model=probe_model,
        dataloader=loader,
        loss_fn=loss_fn,
        device=device,
        probe_type='distance' if probe_type == 'dist' else 'depth',
        filter_by_non_punct_len=True,
        punctuation_strategy='upos',
        spearman_min_len=5,
        spearman_max_len=50,
        use_content_only_spearman=True,
    )
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, help='UD dataset slug, e.g., UD_English-EWT')
    ap.add_argument('--model', default='bert-base-multilingual-cased')
    ap.add_argument('--probe', choices=['dist', 'depth'], required=True)
    ap.add_argument('--layer', required=True, help='e.g., L7')
    ap.add_argument('--layer_index', type=int, default=None, help='Optional explicit numeric index (e.g., 7)')
    ap.add_argument('--checkpoint', required=True, help='Local path to best.pt (artifact already downloaded)')
    ap.add_argument('--run_id', required=True, help='A unique run identifier for the output directory')
    ap.add_argument('--rank', type=int, default=128)
    args = ap.parse_args()

    paths = load_dataset_paths(args.dataset)
    # Embeddings HDF5 convention
    hdf5_path_template = REPO_ROOT / 'data_staging' / 'embeddings' / args.dataset / args.model / f"{args.dataset}_conllu_{{split}}_layers-all_align-mean.hdf5"
    hdf5_dev = str(hdf5_path_template.with_name(hdf5_path_template.name.replace('{split}', 'dev')).as_posix().replace('{split}', 'dev'))
    hdf5_test = str(hdf5_path_template.with_name(hdf5_path_template.name.replace('{split}', 'test')).as_posix().replace('{split}', 'test'))

    # Determine numeric layer index from 'Lk'
    layer_idx = args.layer_index
    if layer_idx is None:
        if args.layer.startswith('L') and args.layer[1:].isdigit():
            layer_idx = int(args.layer[1:])
        else:
            raise ValueError(f'Cannot infer layer index from --layer {args.layer}')

    # Infer embedding dim by loading one sentence via dataset helper
    tmp_ds = ProbeDataset(
        conllu_filepath=paths['dev'],
        hdf5_filepath=hdf5_dev,
        embedding_layer_index=layer_idx,
        probe_task_type='distance' if args.probe == 'dist' else 'depth',
        preload=False,
        collapse_punct=False,
    )
    embedding_dim = tmp_ds.embedding_dim
    del tmp_ds

    probe_model, device = load_probe_from_checkpoint(
        checkpoint_path=args.checkpoint,
        embedding_dim=embedding_dim,
        probe=args.probe,
        rank=args.rank,
    )

    loss_fn = distance_l1_loss if args.probe == 'dist' else depth_l1_loss

    dev_metrics = run_eval_split(
        probe_model, loss_fn, device, args.probe, paths['dev'], hdf5_dev, layer_idx
    )
    test_metrics = run_eval_split(
        probe_model, loss_fn, device, args.probe, paths['test'], hdf5_test, layer_idx
    )

    out_dir = make_output_dir(args.dataset, args.model, args.probe, args.layer, args.run_id)
    # Summaries
    summary = {
        'dataset': args.dataset,
        'model': args.model,
        'probe': args.probe,
        'layer': args.layer,
        'layer_index': layer_idx,
        'rank': args.rank,
        'dev': {k: v for k, v in dev_metrics.items() if isinstance(v, (int, float))},
        'test': {k: v for k, v in test_metrics.items() if isinstance(v, (int, float))},
    }
    (out_dir / 'metrics_summary.json').write_text(json.dumps(summary, indent=2))

    # Detailed per-sentence payloads (as returned by evaluate_probe)
    def extract_detailed(m: Dict[str, Any], is_dist: bool) -> Dict[str, Any]:
        keys = [
            'spearmanr_hm_individual_scores_in_range',
            'spearmanr_content_only_individual_scores',
        ]
        if is_dist:
            keys += ['uuas_per_sentence']
        else:
            keys += ['root_acc_per_sentence']

        payload = {k: m.get(k, []) for k in keys}

        # For depth: add alignment helpers when available
        if not is_dist:
            # Kept indices (when provided by evaluator later) or empty list
            kept_idx = m.get('kept_sentence_indices') or m.get('eval_sentence_indices') or []
            payload['kept_sentence_indices'] = kept_idx

            # Full-length per-sentence (if provided) or omit
            if 'root_acc_per_sentence_full' in m:
                payload['root_acc_per_sentence_full'] = m.get('root_acc_per_sentence_full')

        return payload

    (out_dir / 'dev_detailed_metrics.json').write_text(
        json.dumps(extract_detailed(dev_metrics, args.probe == 'dist'), indent=2)
    )
    (out_dir / 'test_detailed_metrics.json').write_text(
        json.dumps(extract_detailed(test_metrics, args.probe == 'dist'), indent=2)
    )

    print(f'Wrote: {out_dir}')


if __name__ == '__main__':
    main()


