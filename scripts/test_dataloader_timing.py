#!/usr/bin/env python3
import argparse, time, multiprocessing as mp
import torch
from torch.utils.data import DataLoader
from torch_probe.dataset import ProbeDataset, collate_probe_batch

def main():
    ctx = mp.get_context("fork")

    p = argparse.ArgumentParser(description="Time N data‑loader iterations")
    p.add_argument(
        "--conllu",
        default="data/ud_english_ewt_official/en_ewt-ud-train.conllu",
    )
    p.add_argument(
        "--hdf5",
        default="data_staging/embeddings/ud_english_ewt_full/elmo/ud_english_ewt_full_train_layers-all_align-mean.hdf5",
    )
    p.add_argument("--layer", type=int, default=1)
    p.add_argument("--probe", choices=["distance","depth"], default="distance")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument(
        "--num-iters",
        type=int,
        default=20,
        help="How many batches to time (reduced default for quick multi‑worker runs)",
    )
    p.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[0,4,8],
        help="List of num_workers to try",
    )
    p.add_argument("--pin-memory", action="store_true")
    args = p.parse_args()

    ds = ProbeDataset(
        conllu_filepath=args.conllu,
        hdf5_filepath=args.hdf5,
        embedding_layer_index=args.layer,
        probe_task_type=args.probe,
        embedding_dim=None,
    )

    def time_run(num_workers, pin_memory):
        dl_kwargs = {
            "dataset": ds,
            "batch_size": args.batch,
            "shuffle": False,
            "num_workers": num_workers,
            "pin_memory": pin_memory and torch.cuda.is_available(),
            "collate_fn": collate_probe_batch,
        }
        if num_workers > 0:
            dl_kwargs["multiprocessing_context"] = ctx
        loader = DataLoader(**dl_kwargs)

        t0 = time.time()
        for i, _ in enumerate(loader):
            if i + 1 >= args.num_iters:
                break
        return time.time() - t0

    print(f"Timing {args.num_iters} iters @ batch={args.batch}")
    for w in args.workers:
        t = time_run(w, pin_memory=False)
        print(f" workers={w:2d}, pin_memory=False → {t:.2f}s")
        if args.pin_memory:
            t2 = time_run(w, pin_memory=True)
            print(f" workers={w:2d}, pin_memory=True  → {t2:.2f}s")

if __name__=="__main__":
    main()
