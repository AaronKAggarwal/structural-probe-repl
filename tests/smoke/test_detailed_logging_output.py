# tests/smoke/test_detailed_logging_output.py
import subprocess
import pytest
from pathlib import Path
import shutil
import os
import tempfile # Will use pytest's tmp_path fixture instead

# Define Project and Script Directories
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TRAIN_SCRIPT_PATH = SCRIPTS_DIR / "train_probe.py"

# Define paths to known sample data (consistent with test_probe_pipeline_smoke.py)
CONLLU_FILE_REL_PATH = "src/legacy/structural_probe/example/data/en_ewt-ud-sample/en_ewt-ud-dev.conllu"
HDF5_FILE_REL_PATH = "src/legacy/structural_probe/example/data/en_ewt-ud-sample/en_ewt-ud-dev.elmo-layers.hdf5"

# Match embedding details for the sample HDF5 file
ELMO_LAYER_INDEX = 2  # As used in the other smoke test for this HDF5 file
EMBEDDING_DIM = 1024 # As used in the other smoke test

# Check if data files exist for conditional skipping
data_files_exist = (PROJECT_ROOT / CONLLU_FILE_REL_PATH).exists() and \
                   (PROJECT_ROOT / HDF5_FILE_REL_PATH).exists()

skip_if_no_data = pytest.mark.skipif(not data_files_exist,
                                     reason="Sample data files for smoke test not found at expected legacy paths.")

@skip_if_no_data
def test_detailed_outputs_generation(tmp_path: Path):
    """
    Smoke test to verify the generation of detailed metric JSON files and plots
    when running train_probe.py. This test overrides Hydra's run directory
    to point directly to tmp_path.
    """
    print(f"\n--- Smoke Test: Detailed Logging Outputs Generation (1 epoch, ELMo sample, Depth Probe) ---")
    print(f"Using temporary output directory for Hydra run: {tmp_path}")

    conllu_path_for_override = CONLLU_FILE_REL_PATH
    hdf5_path_for_override = HDF5_FILE_REL_PATH
    
    probe_type = "depth" # Using depth probe for this test
    probe_rank = 4 
    num_epochs = 1
    # Determine the monitor_metric that will be used by train_probe.py with default config
    # Default from configs/config.yaml is 'loss' for training.early_stopping_metric
    monitor_metric_for_test = "loss" 

    cmd = [
        "poetry", "run", "python", str(TRAIN_SCRIPT_PATH),
        f"hydra.run.dir={str(tmp_path)}", # CRITICAL: Force Hydra outputs directly into tmp_path
        f"dataset.paths.conllu_train={conllu_path_for_override}",
        f"dataset.paths.conllu_dev={conllu_path_for_override}",
        "dataset.paths.conllu_test=null",
        f"embeddings.paths.train={hdf5_path_for_override}",
        f"embeddings.paths.dev={hdf5_path_for_override}",
        "embeddings.paths.test=null",
        f"embeddings.layer_index={ELMO_LAYER_INDEX}",
        f"embeddings.dimension={EMBEDDING_DIM}",
        f"probe.type={probe_type}",
        f"probe.rank={probe_rank}",
        f"training.epochs={num_epochs}",
        "training.batch_size=2",
        "training.optimizer.name=Adam",
        "training.optimizer.lr=0.001",
        "training.early_stopping_metric=loss", # Explicitly set for clarity in test
        "training.save_every_epoch_checkpoint=true", # Ensure epoch 1 checkpoint is saved
        "logging.wandb.enable=false",
        "logging.enable_plots=true", # Ensure plots are generated
        "runtime.num_workers=0",
        "runtime.device=cpu",
        "hydra.job.name=smoke_test_detailed_logging_direct_tmp" # Give a job name
        # No hydra.sweep.dir needed for single run
    ]

    print(f"Running command: {' '.join(cmd)}")

    try:
        # train_probe.py will use original_cwd (PROJECT_ROOT) to resolve dataset paths
        # but will save outputs to tmp_path due to hydra.run.dir override.
        process = subprocess.run(cmd, capture_output=True, text=True, check=False, cwd=PROJECT_ROOT)
        
        if process.returncode != 0:
            print("STDOUT (test_detailed_outputs_generation):")
            print(process.stdout)
            print("STDERR (test_detailed_outputs_generation):")
            print(process.stderr)
        process.check_returncode() # Raise an exception if the process failed

        # Assertions for expected files directly in tmp_path
        # (because hydra.run.dir points there, and hydra.job.chdir=true in main config
        # should make train_probe.py's CWD = tmp_path)
        
        # Verify CWD behavior based on train_probe.py's logging
        # This assumes train_probe.py logs "Process CWD after @hydra.main decorator: <path>"
        # and "Output directory for this run (from HydraConfig): <path>"
        # And that with hydra.run.dir=tmp_path and hydra.job.chdir=true, these should be tmp_path.
        
        # Actual files are saved relative to the `output_dir` determined by `train_probe.py`
        # which uses `HydraConfig.get().runtime.output_dir`.
        # Since we overrode `hydra.run.dir` to `tmp_path`, `HydraConfig.get().runtime.output_dir` will be `tmp_path`.
        
        output_dir_for_assertions = tmp_path 

        assert (output_dir_for_assertions / f"train_detailed_metrics_epoch{num_epochs}.json").exists(), \
            f"train_detailed_metrics_epoch{num_epochs}.json not found in {output_dir_for_assertions}."
        assert (output_dir_for_assertions / f"dev_detailed_metrics_epoch{num_epochs}.json").exists(), \
            f"dev_detailed_metrics_epoch{num_epochs}.json not found in {output_dir_for_assertions}."
        
        assert (output_dir_for_assertions / "loss_vs_epoch.png").exists(), \
            f"loss_vs_epoch.png not found in {output_dir_for_assertions}."
        
        expected_dev_metric_plot_filename = f"dev_{monitor_metric_for_test}_vs_epoch.png"
        assert (output_dir_for_assertions / expected_dev_metric_plot_filename).exists(), \
            f"{expected_dev_metric_plot_filename} not found in {output_dir_for_assertions}."

        assert (output_dir_for_assertions / "metrics_summary.json").exists(), \
            f"metrics_summary.json not found in {output_dir_for_assertions}."

        checkpoints_dir = output_dir_for_assertions / "checkpoints"
        assert checkpoints_dir.is_dir(), f"Checkpoints directory not found in {output_dir_for_assertions}."
        
        # Check for epoch 1 checkpoint (since save_every_epoch_checkpoint=true)
        epoch_files = list(checkpoints_dir.glob(f"{probe_type}_probe_rank{probe_rank}_epoch{num_epochs}_metric*.pt"))
        assert len(epoch_files) > 0, \
            f"No epoch {num_epochs} checkpoint files found in {checkpoints_dir}. Files: {list(checkpoints_dir.iterdir())}"
        
        # Best checkpoint should also exist (for 1 epoch, it's the same as epoch 1)
        best_file = checkpoints_dir / f"{probe_type}_probe_rank{probe_rank}_best.pt"
        assert best_file.exists(), \
            f"Best checkpoint file ({best_file.name}) not found in {checkpoints_dir}"

    except subprocess.CalledProcessError as e:
        pytest.fail(f"Smoke test (detailed_outputs) training script failed with exit code {e.returncode}.\n"
                    f"Command: {' '.join(e.cmd)}\n"
                    f"Stdout:\n{e.stdout}\nStderr:\n{e.stderr}")
    except Exception as e:
        import traceback
        pytest.fail(f"Smoke test (detailed_outputs) encountered an unexpected error: {e}\n{traceback.format_exc()}")

    print(f"--- Detailed Logging Outputs Smoke Test PASSED (Outputs in {tmp_path}) ---")
