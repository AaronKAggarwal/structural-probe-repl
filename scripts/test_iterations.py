#!/usr/bin/env python3
import argparse
from torch.utils.data import DataLoader
from torch_probe.dataset import ProbeDataset, collate_probe_batch

def main():
    p = argparse.ArgumentParser(
        description="Test iterations/epoch for different batch sizes"
    )
    p.add_argument("--conllu", required=True, help="Path to train .conllu file")
    p.add_argument("--hdf5",  required=True, help="Path to train embeddings .hdf5")
    p.add_argument("--layer", type=int, default=1, help="Embedding layer index")
    p.add_argument(
        "--probe", choices=["distance","depth"], default="distance",
        help="Probe type"
    )
    args = p.parse_args()

    print("Batch‑size → iterations/epoch")
    for bs in [64, 128, 256, 512]:
        ds = ProbeDataset(
            conllu_filepath = args.conllu,
            hdf5_filepath  = args.hdf5,
            embedding_layer_index = args.layer,
            probe_task_type = args.probe,
            embedding_dim   = None
        )
        loader = DataLoader(
            ds, batch_size=bs, shuffle=False,
            num_workers=0, pin_memory=False,
            collate_fn=collate_probe_batch
        )
        print(f"{bs:4d} → {len(loader):4d}")

if __name__=="__main__":
    main()
