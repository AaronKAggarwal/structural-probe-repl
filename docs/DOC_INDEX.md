# Master Documentation Index

Last updated: 2025-05-06

| Document Title              | Path                               | Purpose                                                                             | Update Rule                                            |
|-----------------------------|------------------------------------|-------------------------------------------------------------------------------------|--------------------------------------------------------|
| README (Project Root)       | `README.md`                        | Project overview, goals, and primary entry point.                                   | On major project scope, usage, or layout changes.      |
| Environment Setup           | `docs/ENV_SETUP.md`                | Instructions for native macOS (MPS) & Dockerized (legacy CPU, future CUDA) setup. | When setup steps or core environment components change.  |
| Dependency Notes            | `docs/DEPENDENCIES.md`             | Details on key dependencies, version pins, and rationale across environments.       | When significant pins change or new constraints arise. |
| Frozen Native Requirements  | `requirements-mps.txt`             | Exact native macOS dependencies exported via `poetry export`.                       | Regenerate after any `poetry add/remove/update`.       |
| Repository Architecture     | `docs/ARCHITECTURE.md`             | Overview of project directory structure, legacy code, and new components.           | When major code modules or directories are added/changed. |
| Legacy CPU Dockerfile       | `env/Dockerfile.legacy_pt_cpu`     | Builds Py3.7 + PyTorch 1.3 CPU image for original Hewitt & Manning code.              | Update when base image or key dependencies change.     |
| Legacy Env Health Check     | `scripts/check_legacy_env.sh`      | Verifies critical dependencies inside the `probe:legacy_pt_cpu` container.          | Update if new critical dependencies are added/pinned.    |
| Legacy Probe Runner         | `scripts/run_legacy_probe.sh`      | Wrapper script to execute H&M's `run_experiment.py` in the container.               | Update if argument passing or target script changes.   |
| Build/Debug History         | `docs/HISTORY.md`                  | Chronological log of significant build/debug milestones & resolutions.              | Append after each significant debugging session/fix.     |
| Quirks & Workarounds        | `docs/QUIRKS.md`                   | Lists non-obvious issues, their solutions, or important notes.                    | Append when a new "quirk" is discovered/solved.        |
| *Contributing Guide*        | *`CONTRIBUTING.md`* (Placeholder)  | *Guidelines for contributing to the project (if applicable).*                     | *When collaboration guidelines change.*                |
| *Changelog*                 | *`CHANGELOG.md`* (Placeholder)     | *Record of notable changes for each version/tag.*                                 | *On every semantic version tag.*                       |
| *Code Style Guide*          | *`docs/CODE_STYLE.md`* (Placeholder)| *Coding conventions and style guidelines.*                                        | *When linting rules or style preferences change.*      |
| *Experiment Protocol*       | *`docs/EXPERIMENT_PROTOCOL.md`* (Placeholder)| *Standard procedures for running and logging new experiments.*                  | *Before starting new experimental phases.*             |