# scripts/train_probe.py
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import random
import os
from pathlib import Path
import logging 
import json 
import sys 
from typing import Optional, Dict, Any # Added for clarity

# --- Add src to path for direct execution ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))
# --- End Path Addition ---

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None 

from torch_probe.dataset import ProbeDataset, collate_probe_batch
from torch_probe.probe_models import DistanceProbe, DepthProbe
from torch_probe.loss_functions import distance_l1_loss, depth_l1_loss
from torch_probe.train_utils import get_optimizer, EarlyStopper, LRSchedulerWithOptimizerReset, save_checkpoint, load_checkpoint
from torch_probe.evaluate import evaluate_probe 
from torch.utils.data import DataLoader

log = logging.getLogger(__name__) 

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> Optional[float]:
    
    output_dir = Path.cwd() 
    log.info(f"Hydra output directory: {output_dir}")
    original_cwd = hydra.utils.get_original_cwd()
    log.info(f"Original working directory: {original_cwd}")
    # log.info(f"Loaded config:\n{OmegaConf.to_yaml(cfg)}") # Can be very verbose

    set_seeds(cfg.runtime.seed)

    if cfg.runtime.device == "auto":
        if torch.cuda.is_available(): device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps"); log.info("MPS device selected and available.")
        else: device = torch.device("cpu"); log.info("MPS/CUDA not available. Using CPU.")
    else: device = torch.device(cfg.runtime.device)
    log.info(f"Using device: {device}")

    if cfg.logging.wandb.enable and WANDB_AVAILABLE:
        try:
            wandb.init(
                project=cfg.logging.wandb.project, entity=cfg.logging.wandb.get("entity"), 
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                name=cfg.logging.get("experiment_name", Path(output_dir).name), 
                dir=str(output_dir), reinit=True 
            )
            log.info("Weights & Biases initialized.")
        except Exception as e:
            log.warning(f"Could not initialize W&B: {e}. Proceeding without W&B."); cfg.logging.wandb.enable = False 
    elif cfg.logging.wandb.enable and not WANDB_AVAILABLE:
        log.warning("W&B logging enabled, but 'wandb' not found. Skipping."); cfg.logging.wandb.enable = False

    log.info("Loading data...")
    def resolve_path(p_str: Optional[str]) -> Optional[Path]:
        if p_str is None: return None
        path = Path(p_str)
        return Path(original_cwd) / path if not path.is_absolute() else path

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
        log.warning(f"Config embedding_dim {cfg.embeddings.dimension} != inferred {actual_embedding_dim}. Using inferred.")

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, collate_fn=collate_probe_batch, 
        shuffle=True, num_workers=cfg.runtime.get("num_workers", 0)
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=cfg.training.batch_size, collate_fn=collate_probe_batch, 
        shuffle=False, num_workers=cfg.runtime.get("num_workers", 0)
    )
    log.info(f"Data loaded. Train: {len(train_dataset)}, Dev: {len(dev_dataset)} sentences.")

    log.info("Initializing model, loss, optimizer, schedulers...")
    monitor_metric = cfg.training.early_stopping_metric
    monitor_mode = "min" if monitor_metric == "loss" else "max"
        
    if cfg.probe.type == "distance":
        probe_model = DistanceProbe(actual_embedding_dim, cfg.probe.rank)
        loss_fn = distance_l1_loss
    elif cfg.probe.type == "depth":
        probe_model = DepthProbe(actual_embedding_dim, cfg.probe.rank)
        loss_fn = depth_l1_loss
    else:
        raise ValueError(f"Unknown probe type: {cfg.probe.type}")
    
    probe_model.to(device)
    optimizer = get_optimizer(probe_model.parameters(), cfg.training.optimizer)
    
    early_stopper = EarlyStopper(
        patience=cfg.training.patience, mode=monitor_mode, verbose=True,
        delta=cfg.training.early_stopping_delta
    )
    lr_scheduler_custom = None
    if cfg.training.lr_scheduler_with_reset.get("enable", False):
        lr_scheduler_custom = LRSchedulerWithOptimizerReset(
            optimizer_cfg=cfg.training.optimizer, 
            lr_decay_factor=cfg.training.lr_scheduler_with_reset.lr_decay_factor,
            lr_decay_patience=cfg.training.lr_scheduler_with_reset.lr_decay_patience,
            min_lr=cfg.training.lr_scheduler_with_reset.min_lr,
            monitor_metric_mode=monitor_mode, 
            delta=cfg.training.early_stopping_delta, # Using same delta for LR schedule improvement check
            verbose=True
        )
        log.info("H&M-style LR decay with optimizer reset is ENABLED.")
    else:
        log.info("H&M-style LR decay with optimizer reset is DISABLED.")

    log.info(f"Model, loss, optimizer, schedulers initialized. Monitoring '{monitor_metric}' in '{monitor_mode}' mode.")
    log.info(f"Probe model: {probe_model}")
    log.info(f"Number of parameters: {sum(p.numel() for p in probe_model.parameters() if p.requires_grad)}")

    log.info("Starting training...")
    best_dev_metric_for_checkpointing = float('-inf') if monitor_mode == "max" else float('inf')
    
    try: # tqdm import for progress bar
        from tqdm import tqdm
    except ImportError:
        log.warning("tqdm not found. Progress bar will not be shown."); 
        def tqdm(iterable, *args, **kwargs): return iterable

    best_dev_metric_value_for_checkpointing = float('-inf') if monitor_mode == "max" else float('inf')
    log.info(f"Initial best_dev_metric_for_checkpointing set to: {best_dev_metric_value_for_checkpointing}") # Add for debug
    log.info("Starting training...")

    # Training loop
    for epoch in range(cfg.training.epochs):
        log.info(f"--- Epoch {epoch+1}/{cfg.training.epochs} ---")
        probe_model.train()
        epoch_train_loss = 0.0; num_train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", unit="batch", leave=False)
        for batch_idx, batch in enumerate(train_pbar):
            embeddings_b = batch["embeddings_batch"].to(device)
            labels_b = batch["labels_batch"].to(device) 
            lengths_b = batch["lengths_batch"] 

            optimizer.zero_grad()
            predictions_b = probe_model(embeddings_b)
            loss = loss_fn(predictions_b, labels_b, lengths_b.to(device))
            
            loss.backward()
            if cfg.training.get("clip_grad_norm") is not None:
                torch.nn.utils.clip_grad_norm_(probe_model.parameters(), float(cfg.training.clip_grad_norm))
            optimizer.step()
            
            epoch_train_loss += loss.item(); num_train_batches += 1
            train_pbar.set_postfix(loss=loss.item())
            if cfg.logging.wandb.enable and batch_idx > 0 and batch_idx % cfg.logging.get("log_freq_batch", 20) == 0 :
                wandb.log({"batch_train_loss": loss.item(), "epoch_float": epoch + (batch_idx / len(train_loader))})

        avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0.0
        log.info(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.4f}")
        
        wandb_log_data = {"epoch": epoch + 1, "avg_epoch_train_loss": avg_train_loss, "current_lr": optimizer.param_groups[0]['lr']}

        log.info(f"Running validation for epoch {epoch+1}...")
        dev_metrics = evaluate_probe(probe_model, dev_loader, loss_fn, device, cfg.probe.type)
        
        log_msg = f"Epoch {epoch+1} Dev Metrics: "
        for k, v_met in dev_metrics.items(): log_msg += f"{k}: {v_met:.4f} "
        log.info(log_msg)

        wandb_dev_metrics = {f"dev_{k}": v_met for k,v_met in dev_metrics.items()}
        wandb_log_data.update(wandb_dev_metrics)
        if cfg.logging.wandb.enable and wandb.run: wandb.log(wandb_log_data)

        current_dev_metric_to_monitor = dev_metrics.get(monitor_metric)
        if current_dev_metric_to_monitor is None:
            log.warning(f"Monitor metric '{monitor_metric}' not found in dev_metrics. Using 'loss' for decisions.")
            current_dev_metric_to_monitor = dev_metrics["loss"]
            effective_monitor_mode = "min" # Loss is always minimized
        else:
            effective_monitor_mode = monitor_mode
        
        is_best_for_checkpoint = False
        if effective_monitor_mode == "max":
            if current_dev_metric_to_monitor > best_dev_metric_value_for_checkpointing:
                best_dev_metric_value_for_checkpointing = current_dev_metric_to_monitor
                is_best_for_checkpoint = True
        else: # min mode
            if current_dev_metric_to_monitor < best_dev_metric_value_for_checkpointing:
                best_dev_metric_value_for_checkpointing = current_dev_metric_to_monitor
                is_best_for_checkpoint = True
        
        if is_best_for_checkpoint:
            log.info(f"New best {monitor_metric} for checkpointing: {best_dev_metric_value_for_checkpointing:.4f}.")
        
        save_checkpoint(probe_model, optimizer, epoch + 1, current_dev_metric_to_monitor, 
                        output_dir / "checkpoints", 
                        filename_prefix=f"{cfg.probe.type}_probe_rank{cfg.probe.rank}",
                        is_best=is_best_for_checkpoint)
        
        if lr_scheduler_custom:
            new_opt = lr_scheduler_custom.step(current_dev_metric_to_monitor, probe_model.parameters()) # Pass model.parameters()
            if new_opt:
                log.info(f"Optimizer has been reset by LRScheduler. Old LR: {optimizer.param_groups[0]['lr']:.2e}, New LR: {new_opt.param_groups[0]['lr']:.2e}")
                optimizer = new_opt 
                early_stopper.best_actual_metric = None # Reset early stopper's best score
                log.info("EarlyStopper's best score has been reset due to LR change by custom scheduler.")
        
        if early_stopper(current_dev_metric_to_monitor):
            log.info(f"Early stopping for overall training triggered at epoch {epoch+1}.")
            break
    
    log.info("Training finished.")
    final_metrics_summary: Dict[str, Any] = {"best_dev_monitored_metric_value": early_stopper.best_actual_metric} 

    best_checkpoint_filename = f"{cfg.probe.type}_probe_rank{cfg.probe.rank}_best.pt"
    best_checkpoint_path = output_dir / "checkpoints" / best_checkpoint_filename
    
    if best_checkpoint_path.exists():
        log.info(f"Loading best model from {best_checkpoint_path} for final reporting...")
        loaded_epoch, loaded_metric_val = load_checkpoint(best_checkpoint_path, probe_model, None, device) 
        log.info(f"Best model (from completed epoch {loaded_epoch-1}, dev {monitor_metric}: {loaded_metric_val:.4f}) loaded.")
        final_metrics_summary["best_model_epoch"] = loaded_epoch -1
        final_metrics_summary["best_model_metric_value"] = loaded_metric_val
    else:
        log.warning(f"No best model checkpoint '{best_checkpoint_filename}' found in {output_dir / 'checkpoints'}. Using model from last trained epoch for test set.")
        final_metrics_summary["best_model_epoch"] = epoch + 1 # Last completed epoch
        final_metrics_summary["best_model_metric_value"] = current_dev_metric_to_monitor # Metric from last epoch

    if cfg.dataset.paths.get("conllu_test") and cfg.embeddings.paths.get("test"):
        log.info("Evaluating on test set with best/final model...")
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
            log_msg_test = f"Test Metrics with best/final model: "
            for k, v_met in test_metrics.items(): log_msg_test += f"{k}: {v_met:.4f} "
            log.info(log_msg_test)
            if cfg.logging.wandb.enable and wandb.run: wandb.log({f"final_test_{k}": v_met for k,v_met in test_metrics.items()})
            final_metrics_summary.update({f"test_{k}": v_met for k,v_met in test_metrics.items()})
            test_dataset.close_hdf5()
        except Exception as e:
            log.error(f"Error during test set evaluation: {e}", exc_info=True)
    else:
        log.info("No test set specified in config, skipping final test evaluation.")
    
    summary_path = output_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(final_metrics_summary, f, indent=2)
    log.info(f"Final metrics summary saved to {summary_path}")

    train_dataset.close_hdf5()
    dev_dataset.close_hdf5()

    if cfg.logging.wandb.enable and wandb.run: wandb.finish()
    
    log.info(f"Run finished. Results and checkpoints in: {output_dir}")
    return early_stopper.best_actual_metric if early_stopper.best_actual_metric is not None else (float('-inf') if monitor_mode == "max" else float('inf'))

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, 
                        format='%(asctime)s [%(name)s:%(levelname)s] %(message)s')
    train()