# tests/smoke/test_probe_pipeline_smoke.py
import re
import subprocess
from pathlib import Path
from textwrap import dedent

import pytest
import h5py
import numpy as np

# Adjust imports to be relative from the src directory
from src.torch_probe.dataset import ProbeDataset, collate_probe_batch

# --- Constants ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TRAIN_SCRIPT_PATH = SCRIPTS_DIR / "train_probe.py"
MAIN_CONFIG_DIR = PROJECT_ROOT / "configs"

# Sample data paths relative to PROJECT_ROOT
CONLLU_FILE_REL_PATH_STR = "src/legacy/structural_probe/example/data/en_ewt-ud-sample/en_ewt-ud-dev.conllu"
HDF5_FILE_REL_PATH_STR = "src/legacy/structural_probe/example/data/en_ewt-ud-sample/en_ewt-ud-dev.elmo-layers.hdf5"

# Smoke test parameters
ELMO_LAYER_INDEX_SMOKE = 2
EMBEDDING_DIM_SMOKE = 1024
PROBE_RANK_SMOKE = 4

# --- Pytest Skip Marker ---
data_files_exist = (PROJECT_ROOT / CONLLU_FILE_REL_PATH_STR).exists() and (
    PROJECT_ROOT / HDF5_FILE_REL_PATH_STR
).exists()
skip_if_no_data = pytest.mark.skipif(
    not data_files_exist,
    reason="Sample data files for smoke test not found at expected legacy paths.",
)


@pytest.fixture
def smoke_test_temp_config_dir(tmp_path: Path) -> Path:
    """Creates a temporary directory for an overriding experiment config file."""
    # This structure mimics a small part of your main config dir
    exp_dir = tmp_path / "experiment" / "smoke"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    config_file_path = exp_dir / "test_config.yaml"

    config_content = f"""
    name: smoke/test_config

    # This file primarily contains overrides.
    # The main 'defaults' list will be loaded from the project's real config.yaml
    
    dataset:
      paths:
        conllu_train: "{CONLLU_FILE_REL_PATH_STR}"
        conllu_dev: "{CONLLU_FILE_REL_PATH_STR}"
        conllu_test: null # Ensure we don't run on the full test set
    
    embeddings:
      layer_index: {ELMO_LAYER_INDEX_SMOKE}
      dimension: {EMBEDDING_DIM_SMOKE}
      paths:
        train: "{HDF5_FILE_REL_PATH_STR}"
        dev: "{HDF5_FILE_REL_PATH_STR}"
        test: null
    
    # Minimal training for speed
    training:
      epochs: 1
      batch_size: 2
      eval_on_train_epoch_end: false
      limit_train_batches: 2
      limit_eval_batches: 2
    
    probe:
      rank: {PROBE_RANK_SMOKE}
    
    runtime:
      device: cpu

    logging:
      wandb:
        enable: false
      enable_plots: false
    """
    config_file_path.write_text(dedent(config_content))
    return tmp_path # Return the root of the temporary config tree

@skip_if_no_data
def test_full_training_pipeline_smoke(tmp_path: Path, smoke_test_temp_config_dir: Path):
    print("\n--- Smoke Test: Full Training Pipeline (1 epoch, ELMo sample) ---")
    
    # This command tells Hydra to search for configs in two places:
    # 1. The main project config directory (to find config.yaml)
    # 2. The temporary directory (to find our overriding experiment file)
    command = [
        "poetry", "run", "python", str(TRAIN_SCRIPT_PATH),
        f"--config-dir={str(MAIN_CONFIG_DIR)}",         # Path to primary config.yaml
        f"--config-dir={str(smoke_test_temp_config_dir)}", # Path to overriding experiment file
        "experiment=smoke/test_config"                  # Select the experiment to run
    ]

    print(f"Running command: {' '.join(command)}")

    try:
        process = subprocess.run(
            command, capture_output=True, text=True, check=False, cwd=PROJECT_ROOT
        )

        if process.returncode != 0:
            print("SMOKE TEST STDOUT:")
            print(process.stdout)
            print("SMOKE TEST STDERR:")
            print(process.stderr)
        
        process.check_returncode()

        run_output_dir = None
        for line in process.stdout.splitlines():
            if "Output directory for this run" in line:
                match = re.search(r"Output directory for this run \(from HydraConfig\): (.*)", line)
                if match:
                    run_output_dir = Path(match.group(1).strip())
                    break
        
        assert run_output_dir is not None, "Could not parse Hydra output directory from stdout."
        assert run_output_dir.is_dir(), f"Hydra output directory {run_output_dir} was not created."
        
        # Verify that an essential output file was created
        assert (run_output_dir / "metrics_summary.json").exists()

    except subprocess.CalledProcessError as e:
        pytest.fail(f"Smoke test failed: {e.stderr}")
    except Exception as e:
        import traceback
        pytest.fail(f"An unexpected error occurred: {e}\n{traceback.format_exc()}")