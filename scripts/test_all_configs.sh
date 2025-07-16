#!/bin/bash
#
# Runs all smoke test suites sequentially.
#
set -e

# Get the directory of the current script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
TEST_SUITE_DIR="$SCRIPT_DIR/smoke_tests"

echo "Running all smoke test suites..."

# Find all test scripts in the smoke_tests directory and run them
for test_script in "$TEST_SUITE_DIR"/*.sh; do
  if [ -f "$test_script" ]; then
    echo -e "\n\n>>>>>>>>>> Running suite: $(basename "$test_script") <<<<<<<<<<"
    bash "$test_script"
  fi
done

echo -e "\n\n ALL SMOKE TEST SUITES PASSED! "