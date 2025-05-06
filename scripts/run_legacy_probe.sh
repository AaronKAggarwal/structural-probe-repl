#!/bin/bash
set -e

echo "--- Running Legacy Probe ---"
echo "Current PWD before any operations: $(pwd)" # Should be /app/structural_probe_original

# Initialize CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR
CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR=""

# This script will be called by the ENTRYPOINT (check_legacy_env.sh), 
# and the arguments to this script will be the CMD from the Dockerfile.
# Example Dockerfile CMD: ["/scripts/run_legacy_probe.sh", "example/config/prd_en_ewt-ud-sample.yaml"]
# So, $1 will be "example/config/prd_en_ewt-ud-sample.yaml"

if [ "$#" -eq 0 ]; then
    # No arguments passed directly to run_legacy_probe.sh when docker run is invoked without overriding CMD
    # This case might not be hit if Dockerfile CMD always provides an argument.
    # But as a fallback, use a default.
    echo "No arguments passed to run_legacy_probe.sh, using default example config."
    CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR="example/config/prd_en_ewt-ud-sample.yaml"
elif [ "$1" == "--config_file" ] && [ -n "$2" ]; then
    # Handle if user explicitly uses --config_file flag with docker run
    # e.g., docker run ... probe:legacy_pt_cpu /scripts/run_legacy_probe.sh --config_file path/to/config.yaml
    echo "Received --config_file flag, using provided path: $2"
    CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR="$2"
else
    # Assume the first argument is the config file path (relative to WORKDIR)
    # This is the case for the default Dockerfile CMD
    echo "Assuming first argument ('$1') is the config file path (relative to WORKDIR)."
    CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR="$1"
fi

if [ -z "$CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR" ]; then
    echo "ERROR: Config file path could not be determined."
    exit 1
fi

# The path is now relative to WORKDIR (/app/structural_probe_original)
# No need to prepend /app/ or anything, as WORKDIR is already set.
# The python script itself is also relative to WORKDIR.
TARGET_PYTHON_SCRIPT="structural-probes/run_experiment.py"

# Check if the config file exists relative to the current PWD (which should be WORKDIR)
if [ ! -f "$CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR" ]; then
    echo "ERROR: Config file not found at $(pwd)/${CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR}"
    echo "Looking in directory: $(pwd)/$(dirname "$CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR")"
    ls -l "$(pwd)/$(dirname "$CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR")"
    exit 1
fi

# Check if the target python script exists
if [ ! -f "$TARGET_PYTHON_SCRIPT" ]; then
    echo "ERROR: Target Python script not found at $(pwd)/${TARGET_PYTHON_SCRIPT}"
    exit 1
fi

echo "Using config file: $(pwd)/${CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR}"
echo "Running: python ${TARGET_PYTHON_SCRIPT} ${CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR}"

python "$TARGET_PYTHON_SCRIPT" "$CONFIG_FILE_PATH_RELATIVE_TO_WORKDIR"

echo "--- Legacy Probe Run Finished ---"