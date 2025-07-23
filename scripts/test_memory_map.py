#!/usr/bin/env python3
import argparse, time, multiprocessing as mp
from torch.utils.data import DataLoader
from torch_probe.dataset import ProbeDataset, collate_probe_batch

# TODO: implement this so it reads embeddings into RAM or via numpy.memmap
class MemProbeDataset(ProbeDataset):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # load entire HDF5 into self.mem_data here

    def __getitem__(self, idx):
        # override to read from self.mem_data
        return super().__getitem__(idx)

def main():
    ctx = mp.get_context("fork")

    p = argparse.ArgumentParser(description="Test HDF5 vs memory‑map speed")
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
    p.add_argument("--num-iters", type=int, default=100)
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    # standard loader
    std_ds = ProbeDataset(
        conllu_filepath=args.conllu,
        hdf5_filepath=args.hdf5,
        embedding_layer_index=args.layer,
        probe_task_type=args.probe,
        embedding_dim=None,
    )
    std_kwargs = {
        "dataset": std_ds,
        "batch_size": args.batch,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": False,
        "collate_fn": collate_probe_batch,
    }
    if args.workers > 0:
        std_kwargs["multiprocessing_context"] = ctx
    std_loader = DataLoader(**std_kwargs)

    # memory‑mapped loader
    mem_ds = MemProbeDataset(
        conllu_filepath=args.conllu,
        hdf5_filepath=args.hdf5,
        embedding_layer_index=args.layer,
        probe_task_type=args.probe,
        embedding_dim=None,
    )
    mem_kwargs = {
        "dataset": mem_ds,
        "batch_size": args.batch,
        "shuffle": False,
        "num_workers": args.workers,
        "pin_memory": False,
        "collate_fn": collate_probe_batch,
    }
    if args.workers > 0:
        mem_kwargs["multiprocessing_context"] = ctx
    mem_loader = DataLoader(**mem_kwargs)

    def time_loader(loader):
        t0 = time.time()
        for i, _ in enumerate(loader):
            if i + 1 >= args.num_iters:
                break
        return time.time() - t0

    t_std = time_loader(std_loader)
    t_mem = time_loader(mem_loader)
    print(f"Standard HDF5:  {t_std:.2f}s for {args.num_iters} iters")
    print(f"Memory‑mapped: {t_mem:.2f}s for {args.num_iters} iters")

if __name__=="__main__":
    main()
