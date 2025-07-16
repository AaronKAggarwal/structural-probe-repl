# scripts/train_probe.py
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

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
    wandb = None

# --- Matplotlib Import ---
try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None

# --- Project Module Imports ---
from torch.utils.data import DataLoader

from torch_probe.dataset import ProbeDataset, collate_probe_batch
from torch_probe.evaluate import evaluate_probe
from torch_probe.loss_functions import depth_l1_loss, distance_l1_loss
from torch_probe.probe_models import DepthProbe, DistanceProbe
from torch_probe.train_utils import (
    EarlyStopper,
    LRSchedulerWithOptimizerReset,
    get_optimizer,
    load_checkpoint,
    save_checkpoint,
)

# --- tqdm Import for progress bar ---
try:
    from tqdm import tqdm
except ImportError:
    log = logging.getLogger(__name__)  # Ensure log is defined before use

    def tqdm(iterable, *args, **kwargs):
        log.warning("tqdm not found. Progress bar will not be shown.")
        return iterable


log = logging.getLogger(__name__)


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_resolved_config(cfg: DictConfig, original_cwd: Path) -> None:
    """Logs the resolved Hydra configuration for key parameters."""
    log.info("--- Resolved Experiment Configuration ---")

    # Helper to resolve paths relative to the original working directory
    def resolve_and_format_path(p_str: Optional[str]) -> str:
        if p_str is None:
            return "N/A"
        path = Path(p_str)
        resolved_path = original_cwd / path if not path.is_absolute() else path
        exists_str = " (exists)" if resolved_path.exists() else " (DOES NOT EXIST)"
        return f"{str(resolved_path)}{exists_str}"

    # Log Experiment Info
    log.info(f"{'Experiment Name':<20}: {cfg.logging.get('experiment_name', 'N/A')}")

    # Log Dataset Info
    log.info("  Dataset:")
    log.info(f"    {'Name':<18}: {cfg.dataset.get('name', 'N/A')}")
    log.info(f"    {'Train CoNLL':<18}: {resolve_and_format_path(cfg.dataset.paths.get('conllu_train'))}")
    log.info(f"    {'Dev CoNLL':<18}: {resolve_and_format_path(cfg.dataset.paths.get('conllu_dev'))}")
    log.info(f"    {'Test CoNLL':<18}: {resolve_and_format_path(cfg.dataset.paths.get('conllu_test'))}")

    # Log Embeddings Info
    log.info("  Embeddings:")
    log.info(f"    {'Model Source':<18}: {cfg.embeddings.get('source_model_name', 'N/A')}")
    log.info(f"    {'Layer Index':<18}: {cfg.embeddings.get('layer_index', 'N/A')}")
    log.info(f"    {'Dimension':<18}: {cfg.embeddings.get('dimension', 'N/A')}")
    log.info(f"    {'Train HDF5':<18}: {resolve_and_format_path(cfg.embeddings.paths.get('train'))}")
    log.info(f"    {'Dev HDF5':<18}: {resolve_and_format_path(cfg.embeddings.paths.get('dev'))}")
    log.info(f"    {'Test HDF5':<18}: {resolve_and_format_path(cfg.embeddings.paths.get('test'))}")

    # Log Probe Info
    log.info("  Probe:")
    log.info(f"    {'Type':<18}: {cfg.probe.get('type', 'N/A')}")
    log.info(f"    {'Rank':<18}: {cfg.probe.get('rank', 'N/A')}")

    # Log Training Info
    log.info("  Training:")
    log.info(f"    {'Optimizer':<18}: {cfg.training.optimizer.get('name', 'N/A')}")
    log.info(f"    {'LR':<18}: {cfg.training.optimizer.get('lr', 'N/A')}")
    log.info(f"    {'Epochs':<18}: {cfg.training.get('epochs', 'N/A')}")
    log.info(f"    {'Batch Size':<18}: {cfg.training.get('batch_size', 'N/A')}")
    log.info(f"    {'Early Stop Metric':<18}: {cfg.training.get('early_stopping_metric', 'N/A')}")

    # Log Evaluation Info
    log.info("  Evaluation:")
    log.info(f"    {'Punctuation Strategy':<18}: {cfg.evaluation.get('punctuation_strategy', 'N/A')}")
    log.info("---------------------------------------")


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> Optional[float]:
    hydra_cfg = HydraConfig.get()
    output_dir = Path(hydra_cfg.runtime.output_dir)
    original_cwd = Path(hydra_cfg.runtime.cwd)
    current_process_cwd = Path.cwd()

    exp = cfg.get("experiment", None)
    log.info("Experiment.probe subtree:\n" + OmegaConf.to_yaml(exp.probe))
    log.info("Raw experiment subtree: " + OmegaConf.to_yaml(exp))

    if exp and exp.get("name"):
        # 1) Pull the run name up for WandB / output naming
        cfg.experiment_name = exp.name

        # 2) Merge all the known top‑level sections so that
        #    only the experiment overrides apply, but defaults remain.

        for section in ("dataset", "embeddings", "probe", "training", "evaluation", "runtime"):
            if section in exp:
                cfg[section] = OmegaConf.merge(cfg[section], exp[section])

        # 3) Logging: merge only the wandb sub‑dict (experiment_name is already handled)
        if exp.logging and exp.logging.get("wandb"):
            cfg.logging.wandb = OmegaConf.merge(cfg.logging.wandb, exp.logging.wandb)

    log_resolved_config(cfg, original_cwd)

    log.info(f"Output directory for this run (from HydraConfig): {output_dir}")
    log.info(f"Original CWD (script launch location): {original_cwd}")
    log.info(f"Process CWD after @hydra.main decorator: {current_process_cwd}")

    job_cfg_from_hydraconfig = OmegaConf.select(hydra_cfg, "job", default=None)
    chdir_status = (
        OmegaConf.select(job_cfg_from_hydraconfig, "chdir", default=None)
        if job_cfg_from_hydraconfig
        else None
    )

    if chdir_status is True:
        if current_process_cwd == output_dir:
            log.info(
                f"Hydra CWD successfully changed to output directory: {current_process_cwd}"
            )
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
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seeds(cfg.runtime.seed)

    if cfg.runtime.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        ):
            device = torch.device("mps")
            log.info("MPS device selected and available.")
        else:
            device = torch.device("cpu")
            log.info("MPS/CUDA not available. Using CPU.")
    else:
        device = torch.device(cfg.runtime.device)
    log.info(f"Using device: {device}")

    if cfg.logging.wandb.enable:
        if WANDB_AVAILABLE:
            try:
                run_name = cfg.logging.get("experiment_name", None)
                if run_name is None:
                    run_name = OmegaConf.select(
                        cfg, "hydra.job.name", default=output_dir.name
                    )

                notes_str = cfg.logging.wandb.get("notes", None)
                if notes_str is not None and not isinstance(notes_str, str):
                    notes_str = str(notes_str)

                wandb.init(
                    project=cfg.logging.wandb.project,
                    entity=cfg.logging.wandb.get("entity"),
                    config=OmegaConf.to_container(
                        cfg, resolve=True, throw_on_missing=True
                    ),
                    name=run_name,
                    notes=notes_str,
                    dir=str(output_dir),
                    resume="allow",
                    id=wandb.util.generate_id(),
                )
                log.info(
                    f"Weights & Biases initialized for run: {wandb.run.name} (ID: {wandb.run.id})"
                )
            except Exception as e:
                log.error(
                    f"Could not initialize W&B: {e}. Proceeding without W&B.",
                    exc_info=True,
                )
                cfg.logging.wandb.enable = False
        else:
            log.warning(
                "W&B logging enabled in config, but 'wandb' library not found. Skipping."
            )
            cfg.logging.wandb.enable = False

    log.info("Loading data...")

    def resolve_path(p_str: Optional[str]) -> Optional[Path]:
        if p_str is None:
            return None
        path = Path(p_str)
        return Path(original_cwd) / path if not path.is_absolute() else path

    train_conllu_path = resolve_path(cfg.dataset.paths.conllu_train)
    dev_conllu_path = resolve_path(cfg.dataset.paths.conllu_dev)
    train_hdf5_path = resolve_path(cfg.embeddings.paths.train)
    dev_hdf5_path = resolve_path(cfg.embeddings.paths.dev)

    train_dataset = ProbeDataset(
        conllu_filepath=str(train_conllu_path),
        hdf5_filepath=str(train_hdf5_path),
        embedding_layer_index=cfg.embeddings.layer_index,
        probe_task_type=cfg.probe.type,
        embedding_dim=cfg.embeddings.get("dimension"),
    )
    dev_dataset = ProbeDataset(
        conllu_filepath=str(dev_conllu_path),
        hdf5_filepath=str(dev_hdf5_path),
        embedding_layer_index=cfg.embeddings.layer_index,
        probe_task_type=cfg.probe.type,
        embedding_dim=train_dataset.embedding_dim,
    )

    actual_embedding_dim = train_dataset.embedding_dim
    if (
        cfg.embeddings.get("dimension") is not None
        and cfg.embeddings.dimension != actual_embedding_dim
    ):
        log.warning(
            f"Config embedding_dim {cfg.embeddings.dimension} != inferred {actual_embedding_dim}. Using inferred."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        collate_fn=collate_probe_batch,
        shuffle=True,
        num_workers=cfg.runtime.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=cfg.training.batch_size,
        collate_fn=collate_probe_batch,
        shuffle=False,
        num_workers=cfg.runtime.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )
    log.info(
        f"Data loaded. Train: {len(train_dataset)}, Dev: {len(dev_dataset)} sentences."
    )

    log.info("Initializing model, loss, optimizer, schedulers...")
    monitor_metric = cfg.training.early_stopping_metric
    monitor_mode = "min" if "loss" in monitor_metric.lower() else "max"

    if cfg.probe.type == "distance":
        probe_model = DistanceProbe(actual_embedding_dim, cfg.probe.rank)
        loss_fn = distance_l1_loss
    elif cfg.probe.type == "depth":
        probe_model = DepthProbe(actual_embedding_dim, cfg.probe.rank)
        loss_fn = depth_l1_loss
    else:
        raise ValueError(f"Unknown probe type: {cfg.probe.type}")

    probe_model.to(device)
    if (
        cfg.logging.wandb.enable
        and wandb.run
        and cfg.logging.wandb.get("watch_model", False)
    ):
        log.info("W&B watching model.")
        wandb.watch(
            probe_model,
            log=cfg.logging.wandb.get("watch_log_type", "all"),
            log_freq=cfg.logging.wandb.get("watch_log_freq", 100),
        )

    optimizer = get_optimizer(probe_model.parameters(), cfg.training.optimizer)

    early_stopper = EarlyStopper(
        patience=cfg.training.patience,
        mode=monitor_mode,
        verbose=True,
        delta=cfg.training.early_stopping_delta,
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
            verbose=True,
        )
        log.info("H&M-style LR decay with optimizer reset is ENABLED.")
    else:
        log.info("H&M-style LR decay with optimizer reset is DISABLED.")

    log.info(
        f"Model, loss, optimizer, schedulers initialized. Monitoring '{monitor_metric}' in '{monitor_mode}' mode."
    )
    log.info(f"Probe model: {probe_model}")
    log.info(
        f"Number of parameters: {sum(p.numel() for p in probe_model.parameters() if p.requires_grad)}"
    )

    log.info("Starting training...")
    epoch_train_losses = []
    epoch_dev_losses = []
    epoch_dev_primary_metrics = []
    epochs_list = []
    last_epoch_completed = 0

    for epoch in range(cfg.training.epochs):
        last_epoch_completed = epoch + 1
        log.info(f"--- Epoch {epoch + 1}/{cfg.training.epochs} ---")
        probe_model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch + 1} Training", unit="batch", leave=False
        )
        for batch_idx, batch in enumerate(train_pbar):
            embeddings_b = batch["embeddings_batch"].to(device, non_blocking=True)
            labels_b = batch["labels_batch"].to(device, non_blocking=True)
            lengths_b = batch["lengths_batch"]

            optimizer.zero_grad(set_to_none=True)
            predictions_b = probe_model(embeddings_b)
            loss = loss_fn(predictions_b, labels_b, lengths_b.to(device))

            loss.backward()
            if (
                cfg.training.get("clip_grad_norm") is not None
                and float(cfg.training.clip_grad_norm) > 0
            ):
                torch.nn.utils.clip_grad_norm_(
                    probe_model.parameters(), float(cfg.training.clip_grad_norm)
                )
            optimizer.step()

            epoch_train_loss += loss.item()
            num_train_batches += 1
            train_pbar.set_postfix(loss=loss.item())

            if (
                cfg.logging.wandb.enable
                and wandb.run
                and cfg.logging.wandb.get("log_batch_metrics", False)
                and (batch_idx + 1) % cfg.logging.wandb.get("log_freq_batch", 50) == 0
            ):
                wandb.log(
                    {
                        "train/batch_loss": loss.item(),
                        "trainer/global_step": epoch * len(train_loader)
                        + batch_idx
                        + 1,
                        "epoch_float": epoch + ((batch_idx + 1) / len(train_loader)),
                    },
                    commit=True,
                )

        avg_train_loss = (
            epoch_train_loss / num_train_batches if num_train_batches > 0 else 0.0
        )
        log.info(f"Epoch {epoch + 1} Average Train Loss: {avg_train_loss:.4f}")

        wandb_log_data_epoch = {"epoch": epoch + 1}
        wandb_log_data_epoch["train/epoch_loss"] = avg_train_loss
        wandb_log_data_epoch["trainer/learning_rate"] = optimizer.param_groups[0]["lr"]

        log.info(f"Running validation for epoch {epoch + 1}...")
        # <<< MERGED CHANGE 1 of 3 >>>
        dev_metrics_full = evaluate_probe(
            probe_model,
            dev_loader,
            loss_fn,
            device,
            cfg.probe.type,
            filter_by_non_punct_len=cfg.evaluation.get("filter_by_non_punct_len", True),
            punctuation_strategy=cfg.evaluation.get("punctuation_strategy", "upos"),
            spearman_min_len=cfg.evaluation.get("spearman_min_len", 5),
            spearman_max_len=cfg.evaluation.get("spearman_max_len", 50),
        )

        # Filter for summary metrics (scalar values) for concise logging
        dev_summary_metrics = {}
        dev_detailed_metrics_data = {}  # For detailed JSON saving
        scalar_metric_keys_to_log = [
            "loss",
            "spearmanr_hm",
            "uuas",
            "root_acc",
        ]  # Define expected scalar keys

        for k, v_met in dev_metrics_full.items():
            if k in scalar_metric_keys_to_log:
                if isinstance(v_met, (float, int, np.number)):
                    dev_summary_metrics[k] = v_met
            elif (
                "_per_sentence" in k
                or "_individual_scores_in_range" in k
                or "_by_length_group" in k
            ):
                dev_detailed_metrics_data[k] = v_met

        log_msg = f"Epoch {epoch + 1} Dev Metrics (Summary): "
        for k in scalar_metric_keys_to_log:
            if k in dev_summary_metrics:
                v_met = dev_summary_metrics[k]
                log_msg += f"{k}: {v_met:.4f} "
        log.info(log_msg)

        dev_detailed_metrics_path = (
            output_dir / f"dev_detailed_metrics_epoch{epoch + 1}.json"
        )
        with open(dev_detailed_metrics_path, "w") as f:
            json.dump(
                dev_detailed_metrics_data, f, indent=2
            )  # Save only detailed parts
        log.info(f"Saved detailed dev metrics to {dev_detailed_metrics_path}")

        if (
            cfg.logging.wandb.enable
            and wandb.run
            and cfg.logging.wandb.get("log_dev_detailed_artifact", False)
        ):
            dev_artifact_name = (
                f"{wandb.run.name}-dev_metrics_epoch_{epoch + 1}"
                if wandb.run
                else f"{cfg.logging.experiment_name}-dev_metrics_epoch_{epoch + 1}"
            )
            dev_artifact = wandb.Artifact(
                name=dev_artifact_name, type="metrics_detailed"
            )
            dev_artifact.add_file(str(dev_detailed_metrics_path))
            wandb.log_artifact(dev_artifact)

        for k, v_met in dev_summary_metrics.items():  # Log summary scalars to W&B
            wandb_log_data_epoch[f"dev/{k}"] = v_met

        log.info(f"Running evaluation on training set for epoch {epoch + 1}...")
        # <<< MERGED CHANGE 2 of 3 >>>
        train_eval_metrics_full = evaluate_probe(
            probe_model,
            train_loader,
            loss_fn,
            device,
            cfg.probe.type,
            filter_by_non_punct_len=cfg.evaluation.get("filter_by_non_punct_len", True),
            punctuation_strategy=cfg.evaluation.get("punctuation_strategy", "upos"),
            spearman_min_len=cfg.evaluation.get("spearman_min_len", 5),
            spearman_max_len=cfg.evaluation.get("spearman_max_len", 50),
        )

        train_eval_summary_metrics = {}
        train_eval_detailed_metrics_data = {}
        for k, v_met in train_eval_metrics_full.items():
            if k in scalar_metric_keys_to_log:
                if isinstance(v_met, (float, int, np.number)):
                    train_eval_summary_metrics[k] = v_met
            elif (
                "_per_sentence" in k
                or "_individual_scores_in_range" in k
                or "_by_length_group" in k
            ):
                train_eval_detailed_metrics_data[k] = v_met

        log_msg_train_eval = f"Epoch {epoch + 1} Train Eval Metrics (Summary): "
        for k in scalar_metric_keys_to_log:
            if k in train_eval_summary_metrics:
                v_met = train_eval_summary_metrics[k]
                log_msg_train_eval += f"{k}: {v_met:.4f} "
        log.info(log_msg_train_eval)

        train_detailed_metrics_path = (
            output_dir / f"train_detailed_metrics_epoch{epoch + 1}.json"
        )
        with open(train_detailed_metrics_path, "w") as f:
            json.dump(train_eval_detailed_metrics_data, f, indent=2)
        log.info(
            f"Saved detailed train set evaluation metrics to {train_detailed_metrics_path}"
        )

        if (
            cfg.logging.wandb.enable
            and wandb.run
            and cfg.logging.wandb.get("log_train_detailed_artifact", False)
        ):
            train_eval_artifact_name = (
                f"{wandb.run.name}-train_metrics_epoch_{epoch + 1}"
                if wandb.run
                else f"{cfg.logging.experiment_name}-train_metrics_epoch_{epoch + 1}"
            )
            train_eval_artifact = wandb.Artifact(
                name=train_eval_artifact_name, type="metrics_detailed"
            )
            train_eval_artifact.add_file(str(train_detailed_metrics_path))
            wandb.log_artifact(train_eval_artifact)

        for (
            k,
            v_met,
        ) in train_eval_summary_metrics.items():  # Log summary scalars to W&B
            wandb_log_data_epoch[f"train_eval/{k}"] = v_met

        if cfg.logging.wandb.enable and wandb.run:
            wandb.log(wandb_log_data_epoch, step=epoch + 1)

        current_dev_metric_to_monitor = dev_summary_metrics.get(monitor_metric)
        if current_dev_metric_to_monitor is None:
            log.warning(
                f"Monitor metric '{monitor_metric}' not found in dev_summary_metrics. Using 'loss' for decisions."
            )
            current_dev_metric_to_monitor = dev_summary_metrics["loss"]
            effective_monitor_mode = "min"
        else:
            effective_monitor_mode = monitor_mode

        epochs_list.append(epoch + 1)
        epoch_train_losses.append(avg_train_loss)
        epoch_dev_losses.append(
            dev_summary_metrics.get("loss", float("nan"))
        )  # Use .get for safety
        epoch_dev_primary_metrics.append(current_dev_metric_to_monitor)

        is_best_for_checkpoint = False
        if early_stopper.best_actual_metric is None:
            is_best_for_checkpoint = True
        elif (
            effective_monitor_mode == "max"
            and current_dev_metric_to_monitor
            > early_stopper.best_actual_metric + early_stopper.delta
        ) or (
            effective_monitor_mode == "min"
            and current_dev_metric_to_monitor
            < early_stopper.best_actual_metric - early_stopper.delta
        ):
            is_best_for_checkpoint = True

        if (
            is_best_for_checkpoint
        ):  # Log only if it's truly a new best according to EarlyStopper's logic
            log.info(
                f"New best {monitor_metric} for checkpointing: {current_dev_metric_to_monitor:.4f} (was {early_stopper.best_actual_metric if early_stopper.best_actual_metric is not None else 'N/A'})."
            )

        # --- Modified Checkpointing Logic ---
        should_call_save_checkpoint_func = False
        if is_best_for_checkpoint:
            should_call_save_checkpoint_func = True

        save_every_epoch = cfg.training.get("save_every_epoch_checkpoint", False)
        if save_every_epoch:
            should_call_save_checkpoint_func = True
            if not is_best_for_checkpoint:  # Log only if not already logged as best
                log.info(
                    f"Saving checkpoint for epoch {epoch + 1} as per 'save_every_epoch_checkpoint' config."
                )

        save_interval = cfg.training.get("save_checkpoint_every_n_epochs", -1)
        is_nth_epoch_for_periodic_save = (
            save_interval > 0 and (epoch + 1) % save_interval == 0
        )
        if is_nth_epoch_for_periodic_save:
            should_call_save_checkpoint_func = True
            if (
                not is_best_for_checkpoint and not save_every_epoch
            ):  # Log only if not already covered
                log.info(
                    f"Saving checkpoint for epoch {epoch + 1} as per 'save_checkpoint_every_n_epochs={save_interval}' config."
                )

        if should_call_save_checkpoint_func:
            save_checkpoint(
                model=probe_model,
                optimizer=optimizer,
                epoch=epoch + 1,
                current_metric_value=current_dev_metric_to_monitor,
                checkpoint_dir=output_dir / "checkpoints",
                filename_prefix=f"{cfg.probe.type}_probe_rank{cfg.probe.rank}",
                is_best=is_best_for_checkpoint,
            )
        elif not is_best_for_checkpoint:
            log.debug(
                f"Skipping non-best checkpoint for epoch {epoch + 1} as per configuration."
            )
        # --- End Modified Checkpointing Logic ---

        if lr_scheduler_custom:
            new_opt = lr_scheduler_custom.step(
                current_dev_metric_to_monitor, probe_model.parameters()
            )
            if new_opt:
                log.info(
                    f"Optimizer has been reset by LRScheduler. Old LR: {optimizer.param_groups[0]['lr']:.2e}, New LR: {new_opt.param_groups[0]['lr']:.2e}"
                )
                optimizer = new_opt
                if cfg.logging.wandb.enable and wandb.run:
                    wandb.log(
                        {
                            "trainer/lr_decay_event": 1,
                            "trainer/new_learning_rate": new_opt.param_groups[0]["lr"],
                        },
                        step=epoch + 1,
                    )
                early_stopper.reset()
                log.info(
                    "EarlyStopper has been reset due to LR change by custom scheduler."
                )

        if early_stopper(current_dev_metric_to_monitor):
            log.info(
                f"Early stopping for overall training triggered at epoch {epoch + 1}."
            )
            break

    log.info("Training finished.")
    final_metrics_summary: Dict[str, Any] = {
        "best_dev_monitored_metric_value": early_stopper.best_actual_metric
        if early_stopper.best_actual_metric is not None
        else "N/A",
        "epochs_completed": last_epoch_completed,
    }

    best_checkpoint_filename = f"{cfg.probe.type}_probe_rank{cfg.probe.rank}_best.pt"
    best_checkpoint_path = output_dir / "checkpoints" / best_checkpoint_filename

    loaded_metric_val_for_summary = (
        current_dev_metric_to_monitor  # Fallback to last epoch's metric
    )
    if best_checkpoint_path.exists():
        log.info(
            f"Loading best model from {best_checkpoint_path} for final reporting..."
        )
        loaded_epoch_completed, loaded_metric_val = load_checkpoint(
            best_checkpoint_path, probe_model, None, device
        )
        log.info(
            f"Best model (from completed epoch {loaded_epoch_completed - 1}, dev {monitor_metric}: {loaded_metric_val:.4f}) loaded."
        )
        final_metrics_summary["best_model_epoch_completed"] = loaded_epoch_completed - 1
        final_metrics_summary["best_model_metric_value_on_dev"] = loaded_metric_val
        loaded_metric_val_for_summary = loaded_metric_val  # Use the actual best metric

        if (
            cfg.logging.wandb.enable
            and wandb.run
            and cfg.logging.wandb.get("log_best_checkpoint_artifact", True)
        ):
            model_artifact_name = (
                f"{wandb.run.name}-best_model"
                if wandb.run
                else f"{cfg.logging.get('experiment_name', 'probe')}-best_model"
            )
            model_artifact = wandb.Artifact(
                name=model_artifact_name,
                type="model",
                description=f"Best {cfg.probe.type} probe (rank {cfg.probe.rank}) based on dev {monitor_metric}.",
                metadata={
                    "probe_config": OmegaConf.to_container(cfg.probe, resolve=True),
                    "best_dev_metric": loaded_metric_val,
                    "best_epoch_completed": loaded_epoch_completed - 1,
                },
            )
            model_artifact.add_file(str(best_checkpoint_path))
            wandb.log_artifact(model_artifact)
            log.info(
                f"Logged best model checkpoint {best_checkpoint_path} as W&B artifact."
            )
    else:
        log.warning(
            f"No best model checkpoint '{best_checkpoint_filename}' found. Test set will use model from last epoch."
        )
        final_metrics_summary["best_model_epoch_completed"] = last_epoch_completed - 1
        final_metrics_summary["best_model_metric_value_on_dev"] = (
            loaded_metric_val_for_summary
        )

    if cfg.dataset.paths.get("conllu_test") and cfg.embeddings.paths.get("test"):
        log.info("Evaluating on test set with best/final model...")
        test_conllu_path = resolve_path(cfg.dataset.paths.conllu_test)
        test_hdf5_path = resolve_path(cfg.embeddings.paths.test)

        try:
            test_dataset = ProbeDataset(
                conllu_filepath=str(test_conllu_path),
                hdf5_filepath=str(test_hdf5_path),
                embedding_layer_index=cfg.embeddings.layer_index,
                probe_task_type=cfg.probe.type,
                embedding_dim=actual_embedding_dim,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.training.batch_size,
                collate_fn=collate_probe_batch,
                num_workers=cfg.runtime.get("num_workers", 0),
                pin_memory=torch.cuda.is_available(),
            )
            # <<< MERGED CHANGE 3 of 3 >>>
            test_metrics_full = evaluate_probe(
                probe_model,
                test_loader,
                loss_fn,
                device,
                cfg.probe.type,
                filter_by_non_punct_len=cfg.evaluation.get(
                    "filter_by_non_punct_len", True
                ),
                punctuation_strategy=cfg.evaluation.get("punctuation_strategy", "upos"),
                spearman_min_len=cfg.evaluation.get("spearman_min_len", 5),
                spearman_max_len=cfg.evaluation.get("spearman_max_len", 50),
            )

            test_summary_metrics = {}
            test_detailed_metrics_data = {}
            for k, v_met in test_metrics_full.items():
                if k in scalar_metric_keys_to_log:
                    if isinstance(v_met, (float, int, np.number)):
                        test_summary_metrics[k] = v_met
                elif (
                    "_per_sentence" in k
                    or "_individual_scores_in_range" in k
                    or "_by_length_group" in k
                ):
                    test_detailed_metrics_data[k] = v_met

            log_msg_test = "Test Metrics with best/final model (Summary): "
            for k in scalar_metric_keys_to_log:
                if k in test_summary_metrics:
                    v_met = test_summary_metrics[k]
                    log_msg_test += f"{k}: {v_met:.4f} "
            log.info(log_msg_test)

            test_detailed_metrics_path = output_dir / "test_detailed_metrics_final.json"
            with open(test_detailed_metrics_path, "w") as f:
                json.dump(test_detailed_metrics_data, f, indent=2)
            log.info(f"Saved detailed test metrics to {test_detailed_metrics_path}")

            if cfg.logging.wandb.enable and wandb.run:
                wandb_test_log_payload = {}
                for (
                    k,
                    v_met_test,
                ) in test_summary_metrics.items():  # Ensure using test_summary_metrics
                    wandb.summary[f"final_test/{k}"] = v_met_test
                    wandb_test_log_payload[f"final_test/{k}"] = v_met_test
                if wandb_test_log_payload:  # Only log if there are metrics
                    wandb.log(wandb_test_log_payload, step=last_epoch_completed)

                if cfg.logging.wandb.get("log_test_detailed_artifact", False):
                    test_artifact_name = (
                        f"{wandb.run.name}-test_metrics_final"
                        if wandb.run
                        else f"{cfg.logging.get('experiment_name', 'probe')}-test_metrics_final"
                    )
                    test_artifact = wandb.Artifact(
                        name=test_artifact_name, type="metrics_detailed"
                    )
                    test_artifact.add_file(str(test_detailed_metrics_path))
                    wandb.log_artifact(test_artifact)

            final_metrics_summary.update(
                {f"test_{k}": v_met for k, v_met in test_summary_metrics.items()}
            )
            test_dataset.close_hdf5()
        except Exception as e:
            log.error(f"Error during test set evaluation: {e}", exc_info=True)
    else:
        log.info("No test set specified in config, skipping final test evaluation.")

    summary_path = output_dir / "metrics_summary.json"
    with open(summary_path, "w") as f:
        json.dump(final_metrics_summary, f, indent=2)
    log.info(f"Final metrics summary saved to {summary_path}")

    if (
        cfg.logging.wandb.enable
        and wandb.run
        and cfg.logging.wandb.get("log_summary_json_artifact", True)
    ):
        summary_artifact_name = (
            f"{wandb.run.name}-summary_metrics"
            if wandb.run
            else f"{cfg.logging.get('experiment_name', 'probe')}-summary_metrics"
        )
        summary_artifact = wandb.Artifact(
            name=summary_artifact_name, type="metrics_summary"
        )
        summary_artifact.add_file(str(summary_path))
        wandb.log_artifact(summary_artifact)
        log.info(f"Logged {summary_path} as W&B artifact.")

    if cfg.logging.get("enable_plots", True):
        if MATPLOTLIB_AVAILABLE and plt is not None:
            log.info("Generating plots...")
            plot_paths_dict = {}

            plt.figure()
            plt.plot(epochs_list, epoch_train_losses, marker="o", label="Train Loss")
            plt.plot(epochs_list, epoch_dev_losses, marker="x", label="Dev Loss")
            plt.title("Loss vs. Epoch")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plot_path_loss = output_dir / "loss_vs_epoch.png"
            plt.savefig(plot_path_loss)
            plt.close()
            log.info(f"Saved loss plot to {plot_path_loss}")
            plot_paths_dict["charts/loss_curves"] = plot_path_loss

            plt.figure()
            plt.plot(epochs_list, epoch_dev_primary_metrics, marker="o")
            plt.title(f"Dev {monitor_metric} vs. Epoch")
            plt.xlabel("Epoch")
            plt.ylabel(f"Dev {monitor_metric}")
            plt.grid(True)
            plot_path_dev_metric = output_dir / f"dev_{monitor_metric}_vs_epoch.png"
            plt.savefig(plot_path_dev_metric)
            plt.close()
            log.info(f"Saved dev {monitor_metric} plot to {plot_path_dev_metric}")
            plot_paths_dict[f"charts/dev_{monitor_metric}_curve"] = plot_path_dev_metric

            log.info("Plot generation complete.")
            if (
                cfg.logging.wandb.enable
                and wandb.run
                and cfg.logging.wandb.get("log_plots_as_images", True)
            ):
                for chart_name, chart_path in plot_paths_dict.items():
                    if Path(chart_path).exists():
                        wandb.log(
                            {chart_name: wandb.Image(str(chart_path))},
                            step=last_epoch_completed,
                        )
        elif not MATPLOTLIB_AVAILABLE and cfg.logging.get("enable_plots", True):
            log.warning(
                "Matplotlib not found, but plotting was requested/enabled. Plots will not be generated."
            )
    else:
        log.info("Plotting is disabled via configuration.")

    train_dataset.close_hdf5()
    dev_dataset.close_hdf5()

    if cfg.logging.wandb.enable and wandb.run:
        wandb.finish()

    log.info(f"Run finished. Results and checkpoints should be in: {output_dir}")
    return (
        early_stopper.best_actual_metric
        if early_stopper.best_actual_metric is not None
        else (float("-inf") if monitor_mode == "max" else float("inf"))
    )


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="%(asctime)s [%(name)s:%(levelname)s] %(message)s",
    )
    train()
