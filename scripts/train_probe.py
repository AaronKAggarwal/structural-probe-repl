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
from hydra.core.hydra_config import HydraConfig

# --- Add src to path for direct execution ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))
# --- End Path Addition ---

# --- W&B Import ---
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None # Ensure wandb is None if not available

# --- Matplotlib Import ---
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

# --- Project Module Imports ---
from torch_probe.dataset import ProbeDataset, collate_probe_batch
from torch_probe.probe_models import DistanceProbe, DepthProbe
from torch_probe.loss_functions import distance_l1_loss, depth_l1_loss
from torch_probe.train_utils import get_optimizer, EarlyStopper, LRSchedulerWithOptimizerReset, save_checkpoint, load_checkpoint
from torch_probe.evaluate import evaluate_probe 
from torch.utils.data import DataLoader

# --- tqdm Import for progress bar ---
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs): # Fallback if tqdm not installed
        log.warning("tqdm not found. Progress bar will not be shown.")
        return iterable

log = logging.getLogger(__name__) 

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> Optional[float]:
    
    # --- Get Hydra's runtime configuration ---
    hydra_cfg = HydraConfig.get()

    # --- Output Directory Determination & CWD Logging ---
    output_dir = Path(hydra_cfg.runtime.output_dir)
    original_cwd = Path(hydra_cfg.runtime.cwd)
    current_process_cwd = Path.cwd()

    log.info(f"Output directory for this run (from HydraConfig): {output_dir}")
    log.info(f"Original CWD (script launch location): {original_cwd}")
    log.info(f"Process CWD after @hydra.main decorator: {current_process_cwd}")

    job_cfg_from_hydraconfig = OmegaConf.select(hydra_cfg, "job", default=None)
    chdir_status = None
    if job_cfg_from_hydraconfig:
        chdir_status = OmegaConf.select(job_cfg_from_hydraconfig, "chdir", default=None)

    if chdir_status is True:
        if current_process_cwd == output_dir:
            log.info(f"Hydra CWD successfully changed to output directory: {current_process_cwd}")
        else:
            log.warning(
                f"HydraConfig indicates job.chdir=TRUE, but CWD ({current_process_cwd}) "
                f"does NOT match runtime output_dir ({output_dir}). Using explicit output_dir for saves."
            )
    else: 
        log.warning(
            f"Hydra job.chdir is '{chdir_status}'. CWD was NOT changed by Hydra from '{original_cwd}'. "
            f"Process CWD remains: {current_process_cwd}. "
            f"All outputs will be relative to the designated output_dir: {output_dir}"
        )
    # --- End Output Directory Determination & CWD Logging ---
    output_dir.mkdir(parents=True, exist_ok=True) 

    set_seeds(cfg.runtime.seed)

    # --- Device Setup ---
    if cfg.runtime.device == "auto":
        if torch.cuda.is_available(): device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps"); log.info("MPS device selected and available.")
        else: device = torch.device("cpu"); log.info("MPS/CUDA not available. Using CPU.")
    else: device = torch.device(cfg.runtime.device)
    log.info(f"Using device: {device}")

    # --- W&B Initialization ---
    if cfg.logging.wandb.enable: # Check cfg first
        if WANDB_AVAILABLE:
            try:
                run_name = cfg.logging.get("experiment_name", None)
                if run_name is None:
                    run_name = OmegaConf.select(cfg, "hydra.job.name", default=output_dir.name)
                
                notes_str = cfg.logging.wandb.get("notes", None)
                if notes_str is not None and not isinstance(notes_str, str):
                    notes_str = str(notes_str)

                wandb.init(
                    project=cfg.logging.wandb.project,
                    entity=cfg.logging.wandb.get("entity"),
                    config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
                    name=run_name,
                    notes=notes_str,
                    dir=str(output_dir), # Save W&B files locally in Hydra's run dir
                    resume="allow",
                    id=wandb.util.generate_id() 
                )
                log.info(f"Weights & Biases initialized for run: {wandb.run.name} (ID: {wandb.run.id})")
                # wandb.watch(probe_model, log="all", log_freq=100) # Call after model init if used
            except Exception as e:
                log.error(f"Could not initialize W&B: {e}. Proceeding without W&B.", exc_info=True)
                cfg.logging.wandb.enable = False 
        else:
            log.warning("W&B logging enabled in config, but 'wandb' library not found. Skipping.")
            cfg.logging.wandb.enable = False
    # --- End W&B Initialization ---

    log.info("Loading data...")
    def resolve_path(p_str: Optional[str]) -> Optional[Path]:
        if p_str is None: return None
        path = Path(p_str)
        # Paths from config are resolved relative to the original CWD where the script was launched
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
        shuffle=True, num_workers=cfg.runtime.get("num_workers", 0), pin_memory=torch.cuda.is_available()
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=cfg.training.batch_size, collate_fn=collate_probe_batch, 
        shuffle=False, num_workers=cfg.runtime.get("num_workers", 0), pin_memory=torch.cuda.is_available()
    )
    log.info(f"Data loaded. Train: {len(train_dataset)}, Dev: {len(dev_dataset)} sentences.")

    log.info("Initializing model, loss, optimizer, schedulers...")
    monitor_metric = cfg.training.early_stopping_metric
    monitor_mode = "min" if "loss" in monitor_metric.lower() else "max" # More robust mode detection
        
    if cfg.probe.type == "distance":
        probe_model = DistanceProbe(actual_embedding_dim, cfg.probe.rank)
        loss_fn = distance_l1_loss
    elif cfg.probe.type == "depth":
        probe_model = DepthProbe(actual_embedding_dim, cfg.probe.rank)
        loss_fn = depth_l1_loss
    else:
        raise ValueError(f"Unknown probe type: {cfg.probe.type}")
    
    probe_model.to(device)
    # --- W&B Watch (optional, call after model is on device) ---
    if cfg.logging.wandb.enable and wandb.run and cfg.logging.wandb.get("watch_model", False):
        log.info("W&B watching model.")
        wandb.watch(probe_model, log=cfg.logging.wandb.get("watch_log_type", "all"), 
                    log_freq=cfg.logging.wandb.get("watch_log_freq", 100))
    # --- End W&B Watch ---

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
            delta=cfg.training.early_stopping_delta,
            verbose=True
        )
        log.info("H&M-style LR decay with optimizer reset is ENABLED.")
    else:
        log.info("H&M-style LR decay with optimizer reset is DISABLED.")

    log.info(f"Model, loss, optimizer, schedulers initialized. Monitoring '{monitor_metric}' in '{monitor_mode}' mode.")
    log.info(f"Probe model: {probe_model}")
    log.info(f"Number of parameters: {sum(p.numel() for p in probe_model.parameters() if p.requires_grad)}")

    log.info("Starting training...")
    # best_dev_metric_for_checkpointing was defined but not used; early_stopper.best_actual_metric is used.
    # Removed: best_dev_metric_value_for_checkpointing = float('-inf') if monitor_mode == "max" else float('inf')
    # log.info(f"Initial best_dev_metric_for_checkpointing set to: {best_dev_metric_value_for_checkpointing}")

    epoch_train_losses = []
    epoch_dev_losses = []
    epoch_dev_primary_metrics = []
    epochs_list = []
    last_epoch_completed = 0

    for epoch in range(cfg.training.epochs):
        last_epoch_completed = epoch + 1
        log.info(f"--- Epoch {epoch+1}/{cfg.training.epochs} ---")
        probe_model.train()
        epoch_train_loss = 0.0; num_train_batches = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} Training", unit="batch", leave=False)
        for batch_idx, batch in enumerate(train_pbar):
            embeddings_b = batch["embeddings_batch"].to(device, non_blocking=True)
            labels_b = batch["labels_batch"].to(device, non_blocking=True) 
            lengths_b = batch["lengths_batch"] # Stays on CPU for loss_fn

            optimizer.zero_grad(set_to_none=True) # Optimization
            predictions_b = probe_model(embeddings_b)
            loss = loss_fn(predictions_b, labels_b, lengths_b.to(device)) # Ensure lengths on device if needed by loss_fn
            
            loss.backward()
            if cfg.training.get("clip_grad_norm") is not None and float(cfg.training.clip_grad_norm) > 0:
                torch.nn.utils.clip_grad_norm_(probe_model.parameters(), float(cfg.training.clip_grad_norm))
            optimizer.step()
            
            epoch_train_loss += loss.item(); num_train_batches += 1
            train_pbar.set_postfix(loss=loss.item())

            if cfg.logging.wandb.enable and wandb.run and \
               cfg.logging.wandb.get("log_batch_metrics", False) and \
               (batch_idx + 1) % cfg.logging.wandb.get("log_freq_batch", 50) == 0:
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "trainer/global_step": epoch * len(train_loader) + batch_idx + 1,
                    "epoch_float": epoch + ((batch_idx + 1) / len(train_loader))
                }, commit=True) # Commit batch metrics

        avg_train_loss = epoch_train_loss / num_train_batches if num_train_batches > 0 else 0.0
        log.info(f"Epoch {epoch+1} Average Train Loss: {avg_train_loss:.4f}")
        
        wandb_log_data_epoch = {"epoch": epoch + 1} # W&B uses its own step or epoch by default if not specified
        wandb_log_data_epoch["train/epoch_loss"] = avg_train_loss
        wandb_log_data_epoch["trainer/learning_rate"] = optimizer.param_groups[0]['lr']

        log.info(f"Running validation for epoch {epoch+1}...")
        dev_metrics_full = evaluate_probe(probe_model, dev_loader, loss_fn, device, cfg.probe.type)
        
        dev_detailed_metrics_data = {k: v for k, v in dev_metrics_full.items() if "_per_sentence" in k}
        dev_summary_metrics = {k: v for k, v in dev_metrics_full.items() if "_per_sentence" not in k}
        
        log_msg = f"Epoch {epoch+1} Dev Metrics (Summary): "
        for k, v_met in dev_summary_metrics.items(): log_msg += f"{k}: {v_met:.4f} "
        log.info(log_msg)

        dev_detailed_metrics_path = output_dir / f"dev_detailed_metrics_epoch{epoch+1}.json" # Unique per epoch
        with open(dev_detailed_metrics_path, "w") as f:
            json.dump(dev_detailed_metrics_data, f, indent=2)
        log.info(f"Saved detailed dev metrics to {dev_detailed_metrics_path}")

        if cfg.logging.wandb.enable and wandb.run and cfg.logging.wandb.get("log_dev_detailed_artifact", False):
            dev_artifact = wandb.Artifact(name=f'{wandb.run.name}-dev_metrics_epoch_{epoch+1}', type='metrics_detailed')
            dev_artifact.add_file(str(dev_detailed_metrics_path))
            wandb.log_artifact(dev_artifact)

        for k, v_met in dev_summary_metrics.items():
            wandb_log_data_epoch[f"dev/{k}"] = v_met
        
        log.info(f"Running evaluation on training set for epoch {epoch+1}...")
        train_eval_metrics_full = evaluate_probe(probe_model, train_loader, loss_fn, device, cfg.probe.type)
        
        train_eval_detailed_metrics_data = {k: v for k, v in train_eval_metrics_full.items() if "_per_sentence" in k}
        train_eval_summary_metrics = {k: v for k, v in train_eval_metrics_full.items() if "_per_sentence" not in k}
        
        log_msg_train_eval = f"Epoch {epoch+1} Train Eval Metrics (Summary): "
        for k, v_met in train_eval_summary_metrics.items(): log_msg_train_eval += f"{k}: {v_met:.4f} "
        log.info(log_msg_train_eval)

        train_detailed_metrics_path = output_dir / f"train_detailed_metrics_epoch{epoch+1}.json" # Unique per epoch
        with open(train_detailed_metrics_path, "w") as f:
            json.dump(train_eval_detailed_metrics_data, f, indent=2)
        log.info(f"Saved detailed train set evaluation metrics to {train_detailed_metrics_path}")

        if cfg.logging.wandb.enable and wandb.run and cfg.logging.wandb.get("log_train_detailed_artifact", False):
            train_eval_artifact = wandb.Artifact(name=f'{wandb.run.name}-train_metrics_epoch_{epoch+1}', type='metrics_detailed')
            train_eval_artifact.add_file(str(train_detailed_metrics_path))
            wandb.log_artifact(train_eval_artifact)

        for k, v_met in train_eval_summary_metrics.items():
            wandb_log_data_epoch[f"train_eval/{k}"] = v_met
        
        if cfg.logging.wandb.enable and wandb.run:
            wandb.log(wandb_log_data_epoch, step=epoch + 1) # Explicitly use epoch as step

        current_dev_metric_to_monitor = dev_summary_metrics.get(monitor_metric)
        if current_dev_metric_to_monitor is None:
            log.warning(f"Monitor metric '{monitor_metric}' not found in dev_metrics. Using 'loss' for decisions.")
            current_dev_metric_to_monitor = dev_summary_metrics["loss"] # Fallback to loss
            effective_monitor_mode = "min" 
        else:
            effective_monitor_mode = monitor_mode

        epochs_list.append(epoch + 1)
        epoch_train_losses.append(avg_train_loss)
        epoch_dev_losses.append(dev_summary_metrics["loss"])
        epoch_dev_primary_metrics.append(current_dev_metric_to_monitor)
        
        is_best_for_checkpoint = False
        # Using early_stopper.best_actual_metric to compare, as it handles the mode (min/max) and delta
        if early_stopper.best_actual_metric is None: # First epoch after init or reset
            is_best_for_checkpoint = True # Save first one
            # No need to log "New best..." here, EarlyStopper/LRScheduler will log its init
        elif (effective_monitor_mode == "max" and current_dev_metric_to_monitor > early_stopper.best_actual_metric + early_stopper.delta) or \
             (effective_monitor_mode == "min" and current_dev_metric_to_monitor < early_stopper.best_actual_metric - early_stopper.delta):
            is_best_for_checkpoint = True
            log.info(f"New best {monitor_metric} for checkpointing: {current_dev_metric_to_monitor:.4f} (was {early_stopper.best_actual_metric:.4f}).")
        
        if is_best_for_checkpoint or cfg.training.get("save_every_epoch_checkpoint", False):
            save_checkpoint(probe_model, optimizer, epoch + 1, current_dev_metric_to_monitor, 
                            output_dir / "checkpoints", 
                            filename_prefix=f"{cfg.probe.type}_probe_rank{cfg.probe.rank}",
                            is_best=is_best_for_checkpoint) # is_best flag for _best.pt symlink/copy
        
        if lr_scheduler_custom:
            new_opt = lr_scheduler_custom.step(current_dev_metric_to_monitor, probe_model.parameters())
            if new_opt:
                log.info(f"Optimizer has been reset by LRScheduler. Old LR: {optimizer.param_groups[0]['lr']:.2e}, New LR: {new_opt.param_groups[0]['lr']:.2e}")
                optimizer = new_opt 
                if cfg.logging.wandb.enable and wandb.run:
                    wandb.log({"trainer/lr_decay_event": 1, "trainer/new_learning_rate": new_opt.param_groups[0]['lr']}, step=epoch+1)
                early_stopper.reset() # Full reset of early stopper state
                log.info("EarlyStopper has been reset due to LR change by custom scheduler.")
        
        if early_stopper(current_dev_metric_to_monitor): # This updates early_stopper.best_actual_metric if improved
            log.info(f"Early stopping for overall training triggered at epoch {epoch+1}.")
            break
    
    log.info("Training finished.")
    final_metrics_summary: Dict[str, Any] = {
        "best_dev_monitored_metric_value": early_stopper.best_actual_metric if early_stopper.best_actual_metric is not None else "N/A",
        "epochs_completed": last_epoch_completed
    }

    best_checkpoint_filename = f"{cfg.probe.type}_probe_rank{cfg.probe.rank}_best.pt"
    best_checkpoint_path = output_dir / "checkpoints" / best_checkpoint_filename
    
    if best_checkpoint_path.exists():
        log.info(f"Loading best model from {best_checkpoint_path} for final reporting...")
        loaded_epoch_completed, loaded_metric_val = load_checkpoint(best_checkpoint_path, probe_model, None, device) 
        log.info(f"Best model (from completed epoch {loaded_epoch_completed-1}, dev {monitor_metric}: {loaded_metric_val:.4f}) loaded.")
        final_metrics_summary["best_model_epoch_completed"] = loaded_epoch_completed -1 # epoch it finished
        final_metrics_summary["best_model_metric_value_on_dev"] = loaded_metric_val

        if cfg.logging.wandb.enable and wandb.run and cfg.logging.wandb.get("log_best_checkpoint_artifact", True):
            model_artifact_name = f"{wandb.run.name}-best_model" if wandb.run else f"{cfg.logging.experiment_name}-best_model"
            model_artifact = wandb.Artifact(
                name=model_artifact_name, type="model",
                description=f"Best {cfg.probe.type} probe (rank {cfg.probe.rank}) based on dev {monitor_metric}.",
                metadata={
                    "probe_config": OmegaConf.to_container(cfg.probe, resolve=True),
                    "best_dev_metric": loaded_metric_val,
                    "best_epoch_completed": loaded_epoch_completed -1
                }
            )
            model_artifact.add_file(str(best_checkpoint_path))
            wandb.log_artifact(model_artifact)
            log.info(f"Logged best model checkpoint {best_checkpoint_path} as W&B artifact.")
    else:
        log.warning(f"No best model checkpoint '{best_checkpoint_filename}' found in {output_dir / 'checkpoints'}. Test set evaluation will use model from last trained epoch.")
        # These were assigned using `epoch` from loop, which might be off if loop broke early. Use `last_epoch_completed`.
        final_metrics_summary["best_model_epoch_completed"] = last_epoch_completed -1 
        final_metrics_summary["best_model_metric_value_on_dev"] = current_dev_metric_to_monitor # Metric from last completed epoch


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
            test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, collate_fn=collate_probe_batch, 
                                     num_workers=cfg.runtime.get("num_workers", 0), pin_memory=torch.cuda.is_available())
            test_metrics_full = evaluate_probe(probe_model, test_loader, loss_fn, device, cfg.probe.type)
            
            test_detailed_metrics_data = {k: v for k, v in test_metrics_full.items() if "_per_sentence" in k}
            test_summary_metrics = {k: v for k, v in test_metrics_full.items() if "_per_sentence" not in k}

            log_msg_test = f"Test Metrics with best/final model (Summary): "
            for k, v_met in test_summary_metrics.items(): log_msg_test += f"{k}: {v_met:.4f} "
            log.info(log_msg_test)

            test_detailed_metrics_path = output_dir / "test_detailed_metrics_final.json" # Unique name
            with open(test_detailed_metrics_path, "w") as f:
                json.dump(test_detailed_metrics_data, f, indent=2)
            log.info(f"Saved detailed test metrics to {test_detailed_metrics_path}")

            if cfg.logging.wandb.enable and wandb.run:
                wandb_test_log_payload = {}
                for k,v_met in test_summary_metrics.items():
                    wandb.summary[f"final_test/{k}"] = v_met 
                    wandb_test_log_payload[f"final_test/{k}"] = v_met
                wandb.log(wandb_test_log_payload, step=last_epoch_completed) # Log at final step

                if cfg.logging.wandb.get("log_test_detailed_artifact", False):
                    test_artifact_name = f"{wandb.run.name}-test_metrics_final" if wandb.run else f"{cfg.logging.experiment_name}-test_metrics_final"
                    test_artifact = wandb.Artifact(name=test_artifact_name, type='metrics_detailed')
                    test_artifact.add_file(str(test_detailed_metrics_path))
                    wandb.log_artifact(test_artifact)

            final_metrics_summary.update({f"test_{k}": v_met for k,v_met in test_summary_metrics.items()})
            test_dataset.close_hdf5()
        except Exception as e:
            log.error(f"Error during test set evaluation: {e}", exc_info=True)
    else:
        log.info("No test set specified in config, skipping final test evaluation.")
    
    summary_path = output_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(final_metrics_summary, f, indent=2)
    log.info(f"Final metrics summary saved to {summary_path}")

    if cfg.logging.wandb.enable and wandb.run and cfg.logging.wandb.get("log_summary_json_artifact", True):
        summary_artifact_name = f"{wandb.run.name}-summary_metrics" if wandb.run else f"{cfg.logging.experiment_name}-summary_metrics"
        summary_artifact = wandb.Artifact(name=summary_artifact_name, type='metrics_summary')
        summary_artifact.add_file(str(summary_path))
        wandb.log_artifact(summary_artifact)
        log.info(f"Logged {summary_path} as W&B artifact.")


    if cfg.logging.get("enable_plots", True):
        if MATPLOTLIB_AVAILABLE and plt is not None:
            log.info("Generating plots...")
            plot_paths_dict = {} # To store paths for W&B logging

            plt.figure()
            plt.plot(epochs_list, epoch_train_losses, marker='o', label="Train Loss")
            plt.plot(epochs_list, epoch_dev_losses, marker='x', label="Dev Loss")
            plt.title("Loss vs. Epoch")
            plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True)
            plot_path_loss = output_dir / "loss_vs_epoch.png"
            plt.savefig(plot_path_loss); plt.close()
            log.info(f"Saved loss plot to {plot_path_loss}")
            plot_paths_dict["charts/loss_curves"] = plot_path_loss
            
            plt.figure()
            plt.plot(epochs_list, epoch_dev_primary_metrics, marker='o')
            plt.title(f"Dev {monitor_metric} vs. Epoch")
            plt.xlabel("Epoch"); plt.ylabel(f"Dev {monitor_metric}"); plt.grid(True)
            plot_path_dev_metric = output_dir / f"dev_{monitor_metric}_vs_epoch.png"
            plt.savefig(plot_path_dev_metric); plt.close()
            log.info(f"Saved dev {monitor_metric} plot to {plot_path_dev_metric}")
            plot_paths_dict[f"charts/dev_{monitor_metric}_curve"] = plot_path_dev_metric
            
            log.info("Plot generation complete.")
            if cfg.logging.wandb.enable and wandb.run and cfg.logging.wandb.get("log_plots_as_images", True):
                for chart_name, chart_path in plot_paths_dict.items():
                    if Path(chart_path).exists():
                        wandb.log({chart_name: wandb.Image(str(chart_path))}, step=last_epoch_completed)
        elif not MATPLOTLIB_AVAILABLE and cfg.logging.get("enable_plots", True): # Only warn if plots were enabled but lib missing
            log.warning("Matplotlib not found, but plotting was requested/enabled. Plots will not be generated.")
    else:
        log.info("Plotting is disabled via configuration.")
    
    train_dataset.close_hdf5()
    dev_dataset.close_hdf5()

    if cfg.logging.wandb.enable and wandb.run:
        wandb.finish()
    
    log.info(f"Run finished. Results and checkpoints should be in: {output_dir}")
    return early_stopper.best_actual_metric if early_stopper.best_actual_metric is not None else \
           (float('-inf') if monitor_mode == "max" else float('inf'))

if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, 
                        format='%(asctime)s [%(name)s:%(levelname)s] %(message)s')
    train()