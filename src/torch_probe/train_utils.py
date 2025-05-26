# src/torch_probe/train_utils.py
from typing import Iterator, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import shutil # For saving best model
import logging # For better logging

# Using OmegaConf for type hinting config, but can be Any if not strictly typed
# from omegaconf import DictConfig # Not strictly needed if just using as Any

logger = logging.getLogger(__name__) # For better logging practices


def get_optimizer(model_parameters: Iterator[nn.Parameter], cfg_optimizer: Any) -> optim.Optimizer: # Changed DictConfig to Any
    """Instantiates a PyTorch optimizer based on Hydra configuration."""
    name = cfg_optimizer.get("name", "Adam").lower()
    lr = cfg_optimizer.get("lr", 1e-3)
    weight_decay = cfg_optimizer.get("weight_decay", 0.0)

    if name == "adam":
        betas = tuple(cfg_optimizer.get("betas", (0.9, 0.999))) # Ensure it's a tuple
        eps = cfg_optimizer.get("eps", 1e-8)
        return optim.Adam(model_parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name == "adamw":
        betas = tuple(cfg_optimizer.get("betas", (0.9, 0.999))) # Ensure it's a tuple
        eps = cfg_optimizer.get("eps", 1e-8)
        return optim.AdamW(model_parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name == "sgd":
        momentum = cfg_optimizer.get("momentum", 0.0)
        return optim.SGD(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer name: {name}")


import logging
logger = logging.getLogger(__name__)

class EarlyStopper:
    def __init__(self, patience: int = 5, mode: str = "min", delta: float = 0.0, verbose: bool = False):
        if patience < 1:
            raise ValueError("Patience should be at least 1.")
        if mode not in ["min", "max"]:
            raise ValueError("Mode should be 'min' or 'max'.")

        self.patience = patience
        self.mode = mode
        self.delta = abs(delta) 
        self.verbose = verbose
        
        self.counter = 0
        # self.best_score_for_stopping will store the transformed score (-metric for min, metric for max)
        self.best_score_for_stopping: Optional[float] = None 
        # self.best_actual_metric will store the actual metric value that was best
        self.best_actual_metric: Optional[float] = None
        self.early_stop = False

    def __call__(self, current_metric_value: float) -> bool:
        score_to_compare = -current_metric_value if self.mode == "min" else current_metric_value

        if self.best_score_for_stopping is None: # First call
            self.best_score_for_stopping = score_to_compare
            self.best_actual_metric = current_metric_value # Initialize best_actual_metric
            if self.verbose:
                logger.info(f"EarlyStopping: Best score initialized to {self.best_actual_metric:.6f}")
            return False 

        improved = False
        if self.mode == "min":
            # For min mode, lower is better. current_metric_value < best_actual_metric - delta
            if current_metric_value < (self.best_actual_metric - self.delta): # type: ignore
                improved = True
        else: # mode == "max"
            # For max mode, higher is better. current_metric_value > best_actual_metric + delta
            if current_metric_value > (self.best_actual_metric + self.delta): # type: ignore
                improved = True
        
        if improved:
            self.best_score_for_stopping = score_to_compare # Not strictly needed if using best_actual_metric for comparison
            self.best_actual_metric = current_metric_value
            if self.verbose:
                logger.info(f"EarlyStopping: Improvement! New best score: {self.best_actual_metric:.6f}")
            self.counter = 0 
        else: # No improvement
            self.counter += 1
            if self.verbose:
                logger.info(f"EarlyStopping: No significant improvement for {self.counter}/{self.patience} epochs. "
                            f"Best: {self.best_actual_metric:.6f}, Current: {current_metric_value:.6f}")
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    @property
    def best_score_actual(self) -> Optional[float]:
        """Returns the best actual metric value seen so far (not transformed)."""
        return self.best_actual_metric


def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, 
                    best_metric_value: float, checkpoint_dir: Path, filename_prefix: str = "probe",
                    is_best: bool = False): # Added is_best flag
    """Saves model and optimizer state."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    # Save epoch-specific checkpoint
    checkpoint_path = checkpoint_dir / f"{filename_prefix}_epoch{epoch}_metric{best_metric_value:.4f}.pt"
    
    state = {
        'epoch': epoch, # The epoch just completed
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric_value': best_metric_value, # Metric of the model being saved this epoch
    }
    torch.save(state, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    if is_best:
        best_model_path = checkpoint_dir / f"{filename_prefix}_best.pt"
        shutil.copyfile(checkpoint_path, best_model_path)
        logger.info(f"Best model updated at {best_model_path}")


def load_checkpoint(checkpoint_path: Path, model: nn.Module, 
                    optimizer: Optional[optim.Optimizer] = None, 
                    device: Optional[torch.device] = None
                    ) -> Tuple[int, float]:
    """Loads model and optimizer state from a checkpoint."""
    if not checkpoint_path.is_file(): # Check if it's a file
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device) 
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Move optimizer states to device if model was moved (important for Adam/AdamW)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    start_epoch = checkpoint.get('epoch', 0) + 1 
    # Metric stored in checkpoint is the metric for *that* model, which becomes the current "best"
    best_metric_value = checkpoint.get('best_metric_value', float('-inf') if EarlyStopper(mode="max").mode == "max" else float('inf')) 
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}. Resuming from epoch {start_epoch}. Loaded best metric: {best_metric_value:.4f}")
    return start_epoch, best_metric_value