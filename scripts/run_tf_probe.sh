#!/usr/bin/env bash
set -e
cd /workspace/legacy/structural_probe
# use the tiny demo config shipped in the repo
printf "The chef that went to the stores was out of food" | \
  python run_demo.py example/demo-bert.yaml
