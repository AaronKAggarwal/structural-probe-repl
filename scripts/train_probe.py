# scripts/train_probe.py
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import random
import os
from pathlib import Path
import logging # Use Python's logging
import json # For saving metrics summary
from typing import Optional, List, Dict, Any, Callable

# W&B Import - try/except for optionality
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None # Define wandb as None if not available

# Assuming src is in pythonpath or project is installed via poetry
from torch_probe.dataset import ProbeDataset, collate_probe_batch
from torch_probe.probe_models import DistanceProbe, DepthProbe
from torch_probe.loss_functions import distance_l1_loss, depth_l1_loss
from torch_probe.train_utils import get_optimizer, EarlyStopper, save_checkpoint, load_checkpoint
from torch_probe.evaluate import evaluate_probe 
from torch.utils.data import DataLoader

# Setup basic logging
log = logging.getLogger(__name__) # Hydra will configure this further


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For MPS, torch.manual_seed is generally sufficient for torch operations
    # For full reproducibility on MPS, you might also need:
    # torch.mps.manual_seed(seed) # If using torch specific MPS ops that need it
    # However, often results are non-deterministic on MPS/GPU anyway due to parallel ops

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> Optional[float]:
    
    # --- Setup ---
    # Hydra automatically changes CWD to the output directory
    # Use hydra.utils.to_absolute_path or get_original_cwd for original paths
    output_dir = Path.cwd() # This is Hydra's output directory
    log.info(f"Hydra output directory: {output_dir}")
    log.info(f"Original working directory: {hydra.utils.get_original_cwd()}")
    log.info(f"Loaded config:\n{OmegaConf.to_yaml(cfg)}")


    set_seeds(cfg.runtime.seed)

    # Device setup
    if cfg.runtime.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg.runtime.device)
    log.info(f"Using device: {device}")

    # W&B Initialization (optional)
    if cfg.logging.wandb.enable and WANDB_AVAILABLE:
        try:
            wandb.init(
                project=cfg.logging.wandb.project,
                entity=cfg.logging.wandb.get("entity"), 
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                name=cfg.logging.get("experiment_name", Path(output_dir).name), # Use Hydra's run dir name if no exp name
                dir=str(output_dir), 
                reinit=True # Allow re-init if called multiple times (e.g. in sweeps)
            )
            log.info("Weights & Biases initialized.")
        except Exception as e:
            log.warning(f"Could not initialize W&B: {e}. Proceeding without W&B.")
            cfg.logging.wandb.enable = False 
    elif cfg.logging.wandb.enable and not WANDB_AVAILABLE:
        log.warning("W&B logging enabled in config, but 'wandb' library not found. Skipping.")
        cfg.logging.wandb.enable = False

    # --- Data Loading ---
    log.info("Loading data...")
    original_cwd = hydra.utils.get_original_cwd()

    def resolve_path(p):
        if p is None: return None
        return Path(original_cwd) / p if not Path(p).is_absolute() else Path(p)

    train_conllu_path = resolve_path(cfg.dataset.paths.conllu_train)
    dev_conllu_path = resolve_path(cfg.dataset.paths.conllu_dev)
    
    train_hdf5_path = resolve_path(cfg.embeddings.paths.train)
    dev_hdf5_path = resolve_path(cfg.embeddings.paths.dev)
    
    train_dataset = ProbeDataset(
        conllu_filepath=str(train_conllu_path), hdf5_filepath=str(train_hdf5_path),
        embedding_layer_index=cfg.embeddings.layer_index, probe_task_type=cfg.probe.type,
        embedding_dim=cfg.embeddings.get("dimension")
    )
    dev_dataset = ProbeDataset(
        conllu_filepath=str(dev_conllu_path), hdf5_filepath=str(dev_hdf5_path),
        embedding_layer_index=cfg.embeddings.layer_index, probe_task_type=cfg.probe.type,
        embedding_dim=train_dataset.embedding_dim 
    )
    
    actual_embedding_dim = train_dataset.embedding_dim 
    if cfg.embeddings.get("dimension") is not None and cfg.embeddings.dimension != actual_embedding_dim:
        log.warning(f"Config embedding_dim {cfg.embeddings.dimension} differs from inferred {actual_embedding_dim}. Using inferred.")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, collate_fn=collate_probe_batch, 
        shuffle=True, num_workers=cfg.runtime.get("num_workers", 0)
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=cfg.training.batch_size, collate_fn=collate_probe_batch, 
        shuffle=False, num_workers=cfg.runtime.get("num_workers", 0)
    )
    log.info(f"Data loaded. Train sentences: {len(train_dataset)}, Dev sentences: {len(dev_dataset)}")

    # --- Model, Loss, Optimizer ---
    log.info("Initializing model, loss, optimizer...")
    if cfg.probe.type == "distance":
        probe_model = DistanceProbe(actual_embedding_dim, cfg.probe.rank)
        loss_fn = distance_l1_loss
        # For early stopping: UUAS is better but requires more setup for per-epoch calculation.
        # Dev loss is simpler. H&M paper mentions early stopping on dev loss.
        monitor_metric = cfg.training.get("early_stopping_metric", "loss") # "loss", "uuas", "spearmanr"
        monitor_mode = "min" if monitor_metric == "loss" else "max"
    elif cfg.probe.type == "depth":
        probe_model = DepthProbe(actual_embedding_dim, cfg.probe.rank)
        loss_fn = depth_l1_loss
        monitor_metric = cfg.training.get("early_stopping_metric", "loss") # "loss", "spearmanr", "root_acc"
        monitor_mode = "min" if monitor_metric == "loss" else "max"
    else:
        raise ValueError(f"Unknown probe type: {cfg.probe.type}")
    
    probe_model.to(device)
    optimizer = get_optimizer(probe_model.parameters(), cfg.training.optimizer)
    early_stopper = EarlyStopper(
        patience=cfg.training.patience, mode=monitor_mode, verbose=True,
        delta=cfg.training.get("early_stopping_delta", 0.001)
    )
    log.info("Model, loss, optimizer initialized.")
    log.info(f"Probe model: {probe_model}")
    log.info(f"Number of parameters: {sum(p.numel() for p in probe_model.parameters() if p.requires_grad)}")


    # --- Training Loop ---
    log.info("Starting training...")
    best_dev_metric_value = float('-inf') if monitor_mode == "max" else float('inf')
    
    for epoch in range(cfg.training.epochs):
        log.info(f"--- Epoch {epoch+1}/{cfg.training.epochs} ---")
        probe_model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0
        
        # Wrap train_loader with tqdm for progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", unit="batch")
        for batch_idx, batch in enumerate(train_pbar):
            embeddings_b = batch["embeddings_batch"].to(device)
            labels_b = batch["labels_batch"].to(device)
            lengths_b = batch["lengths_batch"].to(device) 

            optimizer.zero_grad()
            predictions_b = probe_model(embeddings_b)
            loss = loss_fn(predictions_b, labels_b, lengths_b)
            
            loss.backward()
            if cfg.training.get("clip_grad_norm") is not None: # Check for null explicitly
                torch.nn.utils.clip_grad_norm_(probe_model.parameters(), float(cfg.training.clip_grad_norm))
            optimizer.step()
            
            epoch_train_loss += loss.item()
            num_train_batches += 1
            train_pbar.set_postfix(loss=loss.item())
            if cfg.logging.wandb.enable and batch_idx % cfg.logging.get("log_freq_batch", 20) == 0 : # Log batch loss
                wandb.log({"batch_train_loss": loss.item(), 
                           "epoch_step": epoch + batch_idx/len(train_loader)})


        avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0
        log.info(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.4f}")
        if cfg.logging.wandb.enable:
            wandb.log({"epoch": epoch + 1, "avg_epoch_train_loss": avg_train_loss})

        # Validation
        log.info(f"Running validation for epoch {epoch+1}...")
        dev_metrics = evaluate_probe(probe_model, dev_loader, loss_fn, device, cfg.probe.type)
        
        log_msg = f"Epoch {epoch+1} Dev Metrics: "
        for k, v_met in dev_metrics.items(): log_msg += f"{k}: {v_met:.4f} "
        log.info(log_msg)

        if cfg.logging.wandb.enable:
            wandb_dev_metrics = {f"dev_{k}": v_met for k,v_met in dev_metrics.items()}
            wandb_dev_metrics["epoch"] = epoch + 1
            wandb.log(wandb_dev_metrics)

        current_dev_metric_for_stopping = dev_metrics[monitor_metric]
        
        is_best = False
        if monitor_mode == "max":
            if current_dev_metric_for_stopping > best_dev_metric_value:
                best_dev_metric_value = current_dev_metric_for_stopping
                is_best = True
        else: # min mode
            if current_dev_metric_for_stopping < best_dev_metric_value:
                best_dev_metric_value = current_dev_metric_for_stopping
                is_best = True
        
        if is_best:
            log.info(f"New best {monitor_metric}: {best_dev_metric_value:.4f}. Saving model...")
        save_checkpoint(probe_model, optimizer, epoch + 1, current_dev_metric_for_stopping, 
                        output_dir / "checkpoints", 
                        filename_prefix=f"{cfg.probe.type}_probe_rank{cfg.probe.rank}",
                        is_best=is_best) # Pass is_best to save_checkpoint
        
        if early_stopper(current_dev_metric_for_stopping):
            log.info(f"Early stopping triggered at epoch {epoch+1}.")
            break
    
    log.info("Training finished.")
    
    # --- Post Training ---
    best_checkpoint_path = output_dir / "checkpoints" / f"{cfg.probe.type}_probe_rank{cfg.probe.rank}_best.pt"
    if best_checkpoint_path.exists():
        log.info(f"Loading best model from {best_checkpoint_path} for final reporting...")
        # We don't need optimizer for final eval, can pass None
        _, loaded_metric = load_checkpoint(best_checkpoint_path, probe_model, None, device) 
        log.info(f"Best model loaded (had dev {monitor_metric}: {loaded_metric:.4f}).")
    else:
        log.warning("No best model checkpoint found ('_best.pt'), using model from last epoch for test set if applicable.")

    # Final evaluation on test set
    final_metrics_summary = {"best_dev_metric_on_monitor": best_dev_metric_value}

    if cfg.dataset.paths.get("conllu_test") and cfg.embeddings.paths.get("test"):
        log.info("Evaluating on test set with best model...")
        test_conllu_path = resolve_path(cfg.dataset.paths.conllu_test)
        test_hdf5_path = resolve_path(cfg.embeddings.paths.test)
        
        try:
            test_dataset = ProbeDataset(
                conllu_filepath=str(test_conllu_path), hdf5_filepath=str(test_hdf5_path),
                embedding_layer_index=cfg.embeddings.layer_index, probe_task_type=cfg.probe.type,
                embedding_dim=actual_embedding_dim
            )
            test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, collate_fn=collate_probe_batch)
            
            test_metrics = evaluate_probe(probe_model, test_loader, loss_fn, device, cfg.probe.type)
            log_msg_test = f"Test Metrics: "
            for k, v_met in test_metrics.items(): log_msg_test += f"{k}: {v_met:.4f} "
            log.info(log_msg_test)

            if cfg.logging.wandb.enable:
                wandb.log({f"final_test_{k}": v_met for k,v_met in test_metrics.items()})
            final_metrics_summary.update({f"test_{k}": v_met for k,v_met in test_metrics.items()})
            test_dataset.close_hdf5()
        except Exception as e:
            log.error(f"Error during test set evaluation: {e}")
    else:
        log.info("No test set specified in config, skipping final test evaluation.")
    
    # Save final metrics summary
    summary_path = output_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(final_metrics_summary, f, indent=2)
    log.info(f"Final metrics summary saved to {summary_path}")

    train_dataset.close_hdf5()
    dev_dataset.close_hdf5()

    if cfg.logging.wandb.enable:
        wandb.finish()
    
    log.info(f"Run finished. Results and checkpoints in: {output_dir}")
    return best_dev_metric_value 

if __name__ == "__main__":
    # Configure basic logging for standalone runs if Hydra doesn't override
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # Example of how to load tqdm if not using Hydra's default job logging
    try:
        from tqdm import tqdm # Ensure tqdm is available
    except ImportError:
        def tqdm(x, *args, **kwargs): return x # Dummy tqdm if not installed
        log_warning("tqdm not found, progress bars will be basic.")

    train()