# tests/unit/torch_probe/test_train_utils.py
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np

from src.torch_probe.train_utils import get_optimizer, EarlyStopper, save_checkpoint, load_checkpoint # Adjust import

# --- Tests for get_optimizer ---
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    def forward(self, x): return self.linear(x)

def test_get_optimizer_adam():
    model = DummyModel()
    cfg_opt = OmegaConf.create({"name": "Adam", "lr": 0.01, "weight_decay": 0.001})
    optimizer = get_optimizer(model.parameters(), cfg_opt)
    assert isinstance(optimizer, optim.Adam)
    assert optimizer.defaults['lr'] == 0.01
    assert optimizer.defaults['weight_decay'] == 0.001

def test_get_optimizer_adamw():
    model = DummyModel()
    cfg_opt = OmegaConf.create({"name": "AdamW", "lr": 0.005}) # Test default weight_decay
    optimizer = get_optimizer(model.parameters(), cfg_opt)
    assert isinstance(optimizer, optim.AdamW)
    assert optimizer.defaults['lr'] == 0.005

def test_get_optimizer_sgd():
    model = DummyModel()
    cfg_opt = OmegaConf.create({"name": "SGD", "lr": 0.1, "momentum": 0.9})
    optimizer = get_optimizer(model.parameters(), cfg_opt)
    assert isinstance(optimizer, optim.SGD)
    assert optimizer.defaults['lr'] == 0.1
    assert optimizer.defaults['momentum'] == 0.9

def test_get_optimizer_unsupported():
    model = DummyModel()
    cfg_opt = OmegaConf.create({"name": "UnsupportedOpt", "lr": 0.01})
    with pytest.raises(ValueError):
        get_optimizer(model.parameters(), cfg_opt)

# --- Tests for EarlyStopper ---
def test_early_stopper_min_mode():
    stopper = EarlyStopper(patience=2, mode="min", delta=0.01)
    assert not stopper(10.0) # First call, best_score = 10.0
    assert not stopper(9.9)  # Improvement, best_score = 9.9, counter = 0
    assert not stopper(9.9)  # No improvement (not < 9.9 - 0.01), counter = 1
    assert stopper(9.905)# No improvement, counter = 2
    assert stopper(9.91)   # No improvement, counter = 3 -> early_stop = True
    assert stopper.early_stop
    assert stopper.best_score_actual == 9.9

def test_early_stopper_max_mode():
    stopper = EarlyStopper(patience=1, mode="max", delta=0.1)
    assert not stopper(0.5)  # Best = 0.5
    assert not stopper(0.7)  # Best = 0.7, counter = 0 (Improvement)
    assert stopper(0.65) # No improvement (not > 0.7 + 0.1), counter = 1 -> early_stop = True
    assert stopper.early_stop
    assert stopper.best_score_actual == 0.7
    
def test_early_stopper_no_improvement_stops():
    stopper = EarlyStopper(patience=1, mode="min")
    stopper(10)  # Initial best
    assert stopper(11)  # No improvement, counter = 1, stop.
    assert stopper.early_stop

# --- Tests for save_checkpoint and load_checkpoint ---
    def test_save_and_load_checkpoint(tmp_path: Path):
        model = DummyModel()
        original_weight = model.linear.weight.clone().detach()
        # Original optimizer with lr=0.001
        optimizer = optim.Adam(model.parameters(), lr=0.001) 
        epoch = 5
        best_metric = 0.85

        checkpoint_dir = tmp_path / "checkpoints"
        save_checkpoint(model, optimizer, epoch, best_metric, checkpoint_dir, "test_probe")

        assert (checkpoint_dir / "test_probe_best.pt").exists()
        
        new_model = DummyModel()
        # New optimizer initialized with different lr=0.01
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.01) 

        assert not torch.allclose(new_model.linear.weight, original_weight)

        expected_device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")

        start_epoch, loaded_best_metric = load_checkpoint(
            checkpoint_dir / "test_probe_best.pt", 
            new_model, 
            new_optimizer,
            device=expected_device 
        )

        assert start_epoch == epoch + 1
        assert np.isclose(loaded_best_metric, best_metric)
        assert torch.allclose(new_model.linear.weight, original_weight.to(expected_device))
        
        # MODIFIED ASSERTION for optimizer LR:
        # Check the LR in the first parameter group of the new_optimizer
        assert new_optimizer.param_groups[0]['lr'] == 0.001