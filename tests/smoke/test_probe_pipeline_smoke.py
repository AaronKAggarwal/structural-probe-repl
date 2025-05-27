# tests/smoke/test_probe_pipeline_smoke.py
import subprocess
import pytest
from pathlib import Path
import shutil # Keep for now, might not be needed if tmp_path handles all
import os
import re # For parsing stdout to find the output directory
import subprocess

# Define Project and Script Directories consistently
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TRAIN_SCRIPT_PATH = SCRIPTS_DIR / "train_probe.py" # Absolute path to train_probe.py

# Define paths to known sample data (relative to PROJECT_ROOT)
# These will be passed as overrides and resolved by train_probe.py using original_cwd
CONLLU_FILE_REL_PATH_STR = "src/legacy/structural_probe/example/data/en_ewt-ud-sample/en_ewt-ud-dev.conllu"
HDF5_FILE_REL_PATH_STR = "src/legacy/structural_probe/example/data/en_ewt-ud-sample/en_ewt-ud-dev.elmo-layers.hdf5"

# Embedding details for the sample HDF5 file
ELMO_LAYER_INDEX_SMOKE = 2
EMBEDDING_DIM_SMOKE = 1024
PROBE_RANK_SMOKE = 4 # Keep small for speed
BATCH_SIZE_SMOKE = 2 # Keep small

# Check if data files exist for conditional skipping
data_files_exist = (PROJECT_ROOT / CONLLU_FILE_REL_PATH_STR).exists() and \
                   (PROJECT_ROOT / HDF5_FILE_REL_PATH_STR).exists()

skip_if_no_data = pytest.mark.skipif(not data_files_exist,
                                     reason="Sample data files for smoke test not found at expected legacy paths.")

@pytest.fixture
def smoke_test_config_file(tmp_path: Path) -> Path:
    hydra_output_base_for_smoke_test = tmp_path / "smoke_test_hydra_outputs"
    
    config_content = f"""
    # This config is self-contained and overrides everything from the main config.yaml
    # No 'defaults' list pointing to external group files needed here for a full override.
    
    hydra:
      run:
        dir: {str(hydra_output_base_for_smoke_test)}/${{now:%Y-%m-%d}}/${{now:%H-%M-%S}}
      job:
        chdir: true 
        name: smoke_test_full_pipeline
    
    dataset:
      name: "elmo_ewt_sample_smoke_override"
      paths:
        conllu_train: "{CONLLU_FILE_REL_PATH_STR}"
        conllu_dev: "{CONLLU_FILE_REL_PATH_STR}"
        conllu_test: null
    
    embeddings:
      source_model_name: "elmo_smoke_test_override"
      layer_index: {ELMO_LAYER_INDEX_SMOKE}
      paths:
        train: "{HDF5_FILE_REL_PATH_STR}"
        dev: "{HDF5_FILE_REL_PATH_STR}"
        test: null
      dimension: {EMBEDDING_DIM_SMOKE}

    probe:
      type: "distance"
      rank: {PROBE_RANK_SMOKE} 
    
    training:
      optimizer:
        name: "Adam"
        lr: 0.001
        weight_decay: 0.0
        betas: [0.9, 0.999]
        eps: 1.0e-8
      batch_size: {BATCH_SIZE_SMOKE}
      epochs: 1
      patience: 1 
      early_stopping_metric: "loss" 
      early_stopping_delta: 0.001 
      loss_function: "l1_squared_diff" 
      clip_grad_norm: null
      save_every_epoch_checkpoint: true 
      save_checkpoint_every_n_epochs: -1
      lr_scheduler_with_reset:
        enable: false
    
    evaluation: 
      metrics: ["spearmanr_hm", "uuas"] 
      spearman_min_len: 2 
      spearman_max_len: 100 

    runtime:
      device: "cpu" 
      seed: 42
      num_workers: 0
      resolve_paths: true 

    logging:
      experiment_name: "smoke_test_elmo_distance_pipeline_override"
      log_freq_batch: 1
      enable_plots: false # Keep plots disabled for smoke test speed
      wandb:
        enable: false
    """
    config_file_path = tmp_path / "smoke_test_override_config.yaml"
    config_file_path.write_text(config_content)
    return config_file_path


@skip_if_no_data
def test_full_training_pipeline_smoke(tmp_path: Path, smoke_test_config_file: Path):
    print("\n--- Smoke Test: Full Training Pipeline (1 epoch, ELMo sample) ---")
    print(f"Smoke test config file: {smoke_test_config_file}")
    print(f"Pytest tmp_path for this test: {tmp_path}")

    command = [
        "poetry", "run", "python", str(TRAIN_SCRIPT_PATH),
        # Use --config-dir instead of --config-path for directory containing config_name
        # This tells Hydra to look for 'smoke_test_override_config.yaml' inside tmp_path
        f"--config-dir={str(smoke_test_config_file.parent)}",
        f"--config-name={smoke_test_config_file.stem}", # Should be 'smoke_test_override_config'
        # No need for hydra.run.dir here as it's in smoke_test_config_file
    ]

    print(f"Running command: {' '.join(command)}")

    run_output_dir = None # Initialize
    try:
        # Run train_probe.py from PROJECT_ROOT. 
        # Paths in smoke_test_config_file are relative to PROJECT_ROOT.
        # Hydra's output (hydra.run.dir) is absolute, pointing into tmp_path.
        process = subprocess.run(command, capture_output=True, text=True, check=False, cwd=PROJECT_ROOT)
        
        # Always print stdout/stderr for debugging if something goes wrong
        if process.stdout:
            print("SMOKE TEST STDOUT:")
            print(process.stdout)
        if process.stderr:
            print("SMOKE TEST STDERR:")
            print(process.stderr)
        
        process.check_returncode() # Raise an exception if the process failed

        # --- Find the actual output directory created by Hydra from script's stdout ---
        for line in process.stdout.splitlines():
            if "Output directory for this run (from HydraConfig):" in line:
                # Example line: "[...][__main__][INFO] - Output directory for this run (from HydraConfig): /private/var/.../pytest-of-user/pytest-0/test_X0/smoke_outputs/2023-10-27/12-00-00"
                match = re.search(r"Output directory for this run \(from HydraConfig\): (.*)", line)
                if match:
                    path_str = match.group(1).strip()
                    run_output_dir = Path(path_str)
                    break
        
        assert run_output_dir is not None, "Could not parse Hydra output directory from script stdout."
        assert run_output_dir.is_dir(), f"Hydra output directory {run_output_dir} was not created or is not a directory."
        print(f"Asserting outputs in Hydra run directory: {run_output_dir}")

        # --- Assertions relative to the dynamic run_output_dir ---
        assert (run_output_dir / "metrics_summary.json").exists(), \
            f"metrics_summary.json not found in {run_output_dir}"
        
        checkpoint_dir = run_output_dir / "checkpoints"
        assert checkpoint_dir.is_dir(), \
            f"Checkpoints directory not found in {run_output_dir}"

        probe_type_smoke = "distance" # As per smoke_test_config_file
        probe_rank_smoke = PROBE_RANK_SMOKE 
        
        # Check for at least one epoch checkpoint (epoch1 due to config)
        # And the best checkpoint (which will be the same as epoch1 for a 1-epoch run)
        epoch1_checkpoints = list(checkpoint_dir.glob(f"{probe_type_smoke}_probe_rank{probe_rank_smoke}_epoch1_metric*.pt"))
        assert len(epoch1_checkpoints) > 0, \
            f"No epoch 1 checkpoint file found in {checkpoint_dir}. Files: {list(checkpoint_dir.iterdir())}"

        best_checkpoint = checkpoint_dir / f"{probe_type_smoke}_probe_rank{probe_rank_smoke}_best.pt"
        assert best_checkpoint.exists(), \
            f"Best checkpoint file not found in {checkpoint_dir}"

    except subprocess.CalledProcessError as e:
        # Pytest will capture stdout/stderr automatically on fail, but good to have here too
        pytest.fail(f"Smoke test training script failed with exit code {e.returncode}.\n"
                    f"Command: {' '.join(e.cmd)}\n"
                    f"Stdout:\n{e.stdout}\nStderr:\n{e.stderr}")
    except Exception as e:
        import traceback
        pytest.fail(f"Smoke test encountered an unexpected error: {e}\n{traceback.format_exc()}")
    finally:
        # pytest's tmp_path fixture handles cleanup of its own directory.
        # No need to manually clean PROJECT_ROOT if outputs are correctly isolated to tmp_path.
        if run_output_dir and run_output_dir.exists():
             print(f"Smoke test outputs were in: {run_output_dir}")
        pass

    print("--- Full Training Pipeline Smoke Test PASSED ---")