#!/bin/bash
set -e

echo "--- Running Legacy Probe/Demo ---"
echo "Current PWD before any operations: $(pwd)" # Should be /app/structural_probe_original

# Initialize variables
CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR=""
PYTHON_SCRIPT_TO_RUN="structural-probes/run_experiment.py" # Default to experiment runner
SEED_FOR_EXPERIMENT=1 # Define the seed to use for run_experiment.py

# Argument parsing to determine config file and which python script to run
if [ "$#" -eq 0 ]; then
    # No arguments passed directly, use default for run_experiment.py
    echo "No arguments passed, using default example config for run_experiment.py with seed ${SEED_FOR_EXPERIMENT}."
    CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR="example/config/prd_en_ewt-ud-sample.yaml"
    PYTHON_SCRIPT_TO_RUN="structural-probes/run_experiment.py"
elif [ "$1" == "example/demo-bert.yaml" ]; then
    # Specific check if the first argument IS the demo config
    echo "Config 'example/demo-bert.yaml' detected. Switching to run_demo.py."
    CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR="$1"
    PYTHON_SCRIPT_TO_RUN="structural-probes/run_demo.py"
elif [ "$1" == "--config_file" ] && [ -n "$2" ]; then
    # Handle explicit --config_file flag
    echo "Received --config_file flag, using provided path: $2"
    CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR="$2"
    if [ "$2" == "example/demo-bert.yaml" ]; then
        echo "'example/demo-bert.yaml' (via --config_file) detected. Switching to run_demo.py."
        PYTHON_SCRIPT_TO_RUN="structural-probes/run_demo.py"
    else
        echo "Using run_experiment.py with seed ${SEED_FOR_EXPERIMENT}."
        PYTHON_SCRIPT_TO_RUN="structural-probes/run_experiment.py"
    fi
else
    # Assume the first argument is the config file path.
    echo "Assuming first argument ('$1') is the config file path."
    CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR="$1"
    if [ "$1" == "example/demo-bert.yaml" ]; then
        echo "'example/demo-bert.yaml' detected as first arg. Switching to run_demo.py."
        PYTHON_SCRIPT_TO_RUN="structural-probes/run_demo.py"
    else
        echo "Using run_experiment.py with seed ${SEED_FOR_EXPERIMENT}."
        PYTHON_SCRIPT_TO_RUN="structural-probes/run_experiment.py"
    fi
fi

# Validate that a config file path was determined
if [ -z "$CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR" ]; then
    echo "ERROR: Config file path could not be determined from arguments: $@"
    exit 1
fi

# Validate paths (relative to current PWD, which should be WORKDIR /app/structural_probe_original)
if [ ! -f "$CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR" ]; then
    echo "ERROR: Config file not found at $(pwd)/${CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR}"
    exit 1
fi

if [ ! -f "$PYTHON_SCRIPT_TO_RUN" ]; then
    echo "ERROR: Target Python script not found at $(pwd)/${PYTHON_SCRIPT_TO_RUN}"
    exit 1
fi

# Announce and run
echo "Using config file: $(pwd)/${CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR}"
echo "Target Python script: ${PYTHON_SCRIPT_TO_RUN}"

if [ "$PYTHON_SCRIPT_TO_RUN" == "structural-probes/run_experiment.py" ]; then
    echo "Executing: python ${PYTHON_SCRIPT_TO_RUN} ${CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR} --seed ${SEED_FOR_EXPERIMENT}"
    python "$PYTHON_SCRIPT_TO_RUN" "$CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR" --seed "${SEED_FOR_EXPERIMENT}"
else
    # run_demo.py expects stdin for sentences if not given a file argument for sentences
    echo "Executing: python ${PYTHON_SCRIPT_TO_RUN} ${CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR}"
    echo "Piping stdin to ${PYTHON_SCRIPT_TO_RUN} if applicable..."
    python "$PYTHON_SCRIPT_TO_RUN" "$CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR"
fi

echo "--- Legacy Probe/Demo Run Finished ---"