#!/usr/bin/env python3
import argparse
import torch
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity
from torch_probe.dataset import ProbeDataset, collate_probe_batch
from torch_probe.probe_models import DistanceProbe, DepthProbe
from torch_probe.loss_functions import distance_l1_loss, depth_l1_loss
from torch_probe.train_utils import get_optimizer

def main():
    p = argparse.ArgumentParser(description="Profile N batches")
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
    p.add_argument("--num-batches", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", choices=["cpu","mps","cuda"], default="mps")
    args = p.parse_args()

    ds = ProbeDataset(
        conllu_filepath=args.conllu,
        hdf5_filepath=args.hdf5,
        embedding_layer_index=args.layer,
        probe_task_type=args.probe,
        embedding_dim=None,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_probe_batch,
    )

    model = (
        DistanceProbe(ds.embedding_dim, 128)
        if args.probe=="distance"
        else DepthProbe(ds.embedding_dim, 128)
    )
    device = torch.device(args.device)
    model.to(device)
    loss_fn = distance_l1_loss if args.probe=="distance" else depth_l1_loss
    optim   = get_optimizer(model.parameters(), {"name":"Adam","lr":args.lr})

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.MPS],
        record_shapes=False,
        with_stack=False
    ) as prof:
        for i, batch in enumerate(loader):
            if i >= args.num_batches:
                break
            emb = batch["embeddings_batch"].to(device, non_blocking=True)
            lab = batch["labels_batch"].to(device, non_blocking=True)
            lengths = batch["lengths_batch"].to(device, non_blocking=True)

            optim.zero_grad()
            preds = model(emb)
            loss  = loss_fn(preds, lab, lengths)
            loss.backward()
            optim.step()

    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

if __name__=="__main__":
    main()
