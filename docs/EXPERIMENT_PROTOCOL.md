---
    # Experiment Protocol

    Last updated: 2025-05-26

    This document outlines how to run experiments using the modern structural probe implemented in `scripts/train_probe.py` with Hydra configuration.

    ## Prerequisites

    1.  Native macOS environment set up as per `docs/ENV_SETUP.md`.
    2.  Poetry environment activated (`poetry shell` or use `poetry run ...`).
    3.  Required datasets (e.g., CoNLLU files) and pre-computed, word-aligned embedding files (e.g., HDF5) available at paths specified in Hydra configs.

    ## Running an Experiment

    Experiments are launched using `scripts/train_probe.py` and are primarily controlled by Hydra configuration files located in the `configs/` directory.

    **Basic Command Structure:**
    ```bash
    python scripts/train_probe.py experiment=<experiment_config_name> [other_overrides...]
    ```
    Or with Poetry:
    ```bash
    poetry run python scripts/train_probe.py experiment=<experiment_config_name> [other_overrides...]
    ```

    **Key Components:**

    *   **`experiment=<experiment_config_name>`:** This is the primary way to select a pre-defined experiment. Hydra will look for `<experiment_config_name>.yaml` inside the `configs/experiment/` directory. This experiment file typically uses a `defaults` list to compose configurations from other groups (dataset, embeddings, probe, training).
    *   **`[other_overrides...]`:** You can override any configuration parameter from the command line.
        *   Example: `training.epochs=10`
        *   Example: `runtime.device=cpu`
        *   Example: `probe.rank=64`
        *   Example: `logging.wandb.enable=true`

    **Output Directory:**
    *   Hydra automatically creates an output directory for each run. By default, this is `outputs/YYYY-MM-DD/HH-MM-SS/`.
    *   Inside this directory, you will find:
        *   `.hydra/`: Contains the snapshot of the full configuration used for the run (`config.yaml`, `hydra.yaml`, `overrides.yaml`).
        *   `train_probe.log`: Log output from the script (if default Python logging is captured by Hydra).
        *   `checkpoints/`: Contains saved model checkpoints (e.g., `*_best.pt`, epoch-specific checkpoints).
        *   `metrics_summary.json`: A JSON file with the best development metric and final test metrics.
    *   You can control the output directory structure using Hydra command-line overrides like `hydra.run.dir` or `hydra.sweep.dir` (for multi-runs).

    **Example: Running the ELMo Distance Probe Sample Experiment**
    This assumes you have `configs/experiment/elmo_ewt_dist_phase1_MY_SAMPLE_train.yaml` configured to use your self-generated ELMo HDF5 files.

    ```bash
    poetry run python scripts/train_probe.py \
        experiment=elmo_ewt_dist_phase1_MY_SAMPLE_train \
        runtime.device=mps \
        training.epochs=30 \
        logging.wandb.enable=false \
        hydra.run.dir=outputs/manual_runs/elmo_dist_my_sample_$(date +%F_%H-%M-%S)
    ```

    **Example: Running the ELMo Depth Probe Sample Experiment**
    ```bash
    poetry run python scripts/train_probe.py \
        experiment=elmo_ewt_depth_phase1_MY_SAMPLE_train \ # Assuming you created this config
        runtime.device=mps \
        training.epochs=30 \
        logging.wandb.enable=false \
        hydra.run.dir=outputs/manual_runs/elmo_depth_my_sample_$(date +%F_%H-%M-%S)
    ```

    ## Adding New Datasets or Embeddings

    1.  **New Dataset (CoNLLU):**
        *   Place your CoNLLU files (train, dev, test) in an accessible location (e.g., `data/my_new_dataset/`).
        *   Create a new YAML file in `configs/dataset/`, e.g., `my_new_dataset.yaml`:
            ```yaml
            name: "my_new_dataset_id"
            paths:
              conllu_train: "data/my_new_dataset/train.conllu"
              conllu_dev: "data/my_new_dataset/dev.conllu"
              conllu_test: "data/my_new_dataset/test.conllu"
            ```
    2.  **New Embeddings:**
        *   Run the embedding extraction pipeline (Phase 2) to generate standardized embedding files (e.g., HDF5) for your new dataset and target LLM.
        *   Create a new YAML file in `configs/embeddings/`, e.g., `my_model_layerX_my_dataset.yaml`:
            ```yaml
            source_model_name: "name_of_my_llm"
            layer_index: X # or "all_avg", etc.
            paths:
              train: "embeddings/my_new_dataset/my_model_layerX/train_embeddings.h5"
              dev:   "embeddings/my_new_dataset/my_model_layerX/dev_embeddings.h5"
              test:  "embeddings/my_new_dataset/my_model_layerX/test_embeddings.h5"
            dimension: <embedding_dimension_of_my_model>
            ```
    3.  **New Experiment:**
        *   Create a new YAML file in `configs/experiment/`, e.g., `probe_my_model_on_my_dataset.yaml`:
            ```yaml
            defaults:
              - override /dataset: my_new_dataset
              - override /embeddings: my_model_layerX_my_dataset
              - override /probe: distance_rank128 # Or your desired probe config
              - override /training: default_adam   # Or your desired training config
              - _self_
            
            # Any specific overrides for this experiment
            logging:
              experiment_name: "probe_my_model_on_my_dataset_dist_L_X_r128"
            ```
    4.  **Run:**
        ```bash
        poetry run python scripts/train_probe.py experiment=probe_my_model_on_my_dataset
        ```
    ```

---