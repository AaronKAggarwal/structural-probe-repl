# src/torch_probe/train_utils.py
from typing import Iterator, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import shutil 
import logging

# from omegaconf import DictConfig # Using Any for cfg types for simplicity in this module

logger = logging.getLogger(__name__)


def get_optimizer(model_parameters: Iterator[nn.Parameter], cfg_optimizer: Any) -> optim.Optimizer:
    """Instantiates a PyTorch optimizer based on Hydra configuration."""
    name = cfg_optimizer.get("name", "Adam").lower()
    lr = cfg_optimizer.get("lr", 1e-3) # Initial LR
    weight_decay = cfg_optimizer.get("weight_decay", 0.0)

    logger.info(f"Initializing optimizer: {name} with LR: {lr}, WeightDecay: {weight_decay}")

    if name == "adam":
        betas = tuple(cfg_optimizer.get("betas", (0.9, 0.999))) 
        eps = cfg_optimizer.get("eps", 1e-8)
        return optim.Adam(model_parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name == "adamw":
        betas = tuple(cfg_optimizer.get("betas", (0.9, 0.999))) 
        eps = cfg_optimizer.get("eps", 1e-8)
        return optim.AdamW(model_parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name == "sgd":
        momentum = cfg_optimizer.get("momentum", 0.0)
        return optim.SGD(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer name: {name}")


class LRSchedulerWithOptimizerReset:
    def __init__(self, 
                 optimizer_cfg: Any, # Original optimizer config (e.g., cfg.training.optimizer)
                 lr_decay_factor: float = 0.1, 
                 lr_decay_patience: int = 3, 
                 min_lr: float = 1e-6,
                 monitor_metric_mode: str = "min", # "min" or "max" for the metric being monitored
                 delta: float = 0.001, # Min change to be considered an improvement for LR decay
                 verbose: bool = False):
        
        if not (0 < lr_decay_factor < 1):
            raise ValueError("lr_decay_factor should be between 0 and 1.")
        if lr_decay_patience < 1:
            raise ValueError("lr_decay_patience should be at least 1.")

        self.optimizer_cfg_template = optimizer_cfg # Keep the original optimizer config structure
        self.current_lr = float(optimizer_cfg.lr) # Start with the initial LR
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_patience = lr_decay_patience
        self.min_lr = min_lr
        self.monitor_metric_mode = monitor_metric_mode
        self.delta = abs(delta)
        self.verbose = verbose

        self.epochs_without_improvement = 0
        self.best_metric_for_lr_decay: Optional[float] = None
        self.lr_decays_done = 0

    def step(self, current_metric_value: float, model_params: Iterator[nn.Parameter]) -> Optional[optim.Optimizer]:
        """
        Checks if LR should be decayed based on current_metric_value.
        If so, updates self.current_lr and returns a NEW optimizer instance.
        Otherwise, returns None.
        """
        new_optimizer_to_return = None
        
        if self.best_metric_for_lr_decay is None: # First call
            self.best_metric_for_lr_decay = current_metric_value
            if self.verbose:
                logger.info(f"LR Scheduler: Initialized best metric for LR decay to {self.best_metric_for_lr_decay:.6f}")
            return None 

        improved = False
        if self.monitor_metric_mode == "min":
            if current_metric_value < self.best_metric_for_lr_decay - self.delta:
                improved = True
        else: # mode == "max"
            if current_metric_value > self.best_metric_for_lr_decay + self.delta:
                improved = True
        
        if improved:
            if self.verbose:
                logger.info(f"LR Scheduler: Metric improved to {current_metric_value:.6f}. Resetting LR decay patience.")
            self.best_metric_for_lr_decay = current_metric_value
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.verbose:
                logger.info(f"LR Scheduler: No metric improvement for {self.epochs_without_improvement}/{self.lr_decay_patience} epochs for LR decay. "
                            f"Best: {self.best_metric_for_lr_decay:.6f}, Current: {current_metric_value:.6f}")
            
            if self.epochs_without_improvement >= self.lr_decay_patience:
                new_lr_candidate = self.current_lr * self.lr_decay_factor
                if new_lr_candidate >= self.min_lr:
                    self.current_lr = new_lr_candidate
                    self.lr_decays_done += 1
                    self.epochs_without_improvement = 0 
                    self.best_metric_for_lr_decay = current_metric_value # Reset best metric after LR decay
                    if self.verbose:
                        logger.info(f"LR Scheduler: Decaying LR to {self.current_lr:.7e}. Optimizer will be reset. Decays done: {self.lr_decays_done}.")
                    
                    # Create a new optimizer config with the updated LR
                    # Ensure this works with OmegaConf DictConfig
                    from omegaconf import OmegaConf # Local import for safety
                    if isinstance(self.optimizer_cfg_template, OmegaConf):
                        new_optimizer_cfg = OmegaConf.to_container(self.optimizer_cfg_template, resolve=True)
                        new_optimizer_cfg['lr'] = self.current_lr
                        new_optimizer_cfg = OmegaConf.create(new_optimizer_cfg)
                    else: # Fallback if it's a simple dict
                        new_optimizer_cfg = self.optimizer_cfg_template.copy()
                        new_optimizer_cfg['lr'] = self.current_lr
                    
                    new_optimizer_to_return = get_optimizer(model_params, new_optimizer_cfg)
                else:
                    if self.verbose:
                        logger.info(f"LR Scheduler: Proposed new LR ({new_lr_candidate:.7e}) is below min_lr ({self.min_lr:.7e}). No further decay.")
                        # Optionally, could set LR to min_lr if it's close and not exactly min_lr yet.
                        self.epochs_without_improvement = 0 # Still reset patience
        
        return new_optimizer_to_return


class EarlyStopper:
    def __init__(self, patience: int = 5, mode: str = "min", delta: float = 0.0001, verbose: bool = False):
        if patience < 1:
            raise ValueError("Patience should be at least 1.")
        if mode not in ["min", "max"]:
            raise ValueError("Mode should be 'min' or 'max'.")

        self.patience = patience
        self.mode = mode
        self.delta = abs(delta) 
        self.verbose = verbose
        
        self.counter = 0
        self.best_actual_metric: Optional[float] = None
        self.early_stop = False

    def __call__(self, current_metric_value: float) -> bool:
        if self.best_actual_metric is None: 
            self.best_actual_metric = current_metric_value
            if self.verbose:
                logger.info(f"EarlyStopper: Best metric initialized to {self.best_actual_metric:.6f}")
            return False 

        improved = False
        if self.mode == "min":
            if current_metric_value < self.best_actual_metric - self.delta:
                improved = True
        else: # mode == "max"
            if current_metric_value > self.best_actual_metric + self.delta:
                improved = True
        
        if improved:
            self.best_actual_metric = current_metric_value
            if self.verbose:
                logger.info(f"EarlyStopper: Improvement! New best metric: {self.best_actual_metric:.6f}")
            self.counter = 0 
        else: 
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopper: No significant improvement for {self.counter}/{self.patience} epochs for stopping. "
                            f"Best: {self.best_actual_metric:.6f}, Current: {current_metric_value:.6f}")
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def reset(self) -> None: # <<< ADD THIS METHOD
        """Resets the EarlyStopper's state."""
        self.counter = 0
        self.best_actual_metric = None # Will be re-initialized on the next call
        self.early_stop = False
        if self.verbose:
            logger.info("EarlyStopper has been reset.")
    
    @property 
    def best_score_actual(self) -> Optional[float]:
        """Returns the best actual metric value seen so far (not transformed)."""
        return self.best_actual_metric


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, 
                    current_metric_value: float, checkpoint_dir: Path, filename_prefix: str = "probe",
                    is_best: bool = False):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{filename_prefix}_epoch{epoch}_metric{current_metric_value:.4f}.pt"
    
    state = {
        'epoch': epoch, 
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metric_value': current_metric_value, 
    }
    torch.save(state, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    if is_best:
        best_model_path = checkpoint_dir / f"{filename_prefix}_best.pt"
        try:
            shutil.copyfile(checkpoint_path, best_model_path)
            logger.info(f"Best model updated at {best_model_path}")
        except Exception as e:
            logger.error(f"Failed to copy best model checkpoint: {e}")


def load_checkpoint(checkpoint_path: Path, model: nn.Module, 
                    optimizer: Optional[optim.Optimizer] = None, 
                    device: Optional[torch.device] = None
                    ) -> Tuple[int, float]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if device is None:
        if torch.cuda.is_available(): device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built(): device = torch.device("mps")
        else: device = torch.device("cpu")
            
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device) 
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for state_group in optimizer.state.values():
            for k, v_tensor in state_group.items():
                if isinstance(v_tensor, torch.Tensor):
                    state_group[k] = v_tensor.to(device)
    
    start_epoch_to_resume = checkpoint.get('epoch', 0) # This is the epoch that was *completed*
    metric_value_of_checkpoint = checkpoint.get('metric_value', 
                                                float('-inf') if EarlyStopper(mode="max").mode == "max" else float('inf')) 
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}. Model from completed epoch {start_epoch_to_resume}. Metric: {metric_value_of_checkpoint:.4f}")
    return start_epoch_to_resume + 1, metric_value_of_checkpoint # Training should start from next epoch