**`CONTRIBUTING.md`**

```markdown
# Contributing to Structural Probe Replication & Extensions

Thank you for your interest in contributing to this project! Whether you're fixing a bug, adding a new feature, improving documentation, or suggesting an idea, your input is valuable.

To ensure a smooth and effective collaboration, please take a moment to review these guidelines.

## Table of Contents

1.  [Code of Conduct](#code-of-conduct)
2.  [How Can I Contribute?](#how-can-i-contribute)
    *   [Reporting Bugs](#reporting-bugs)
    *   [Suggesting Enhancements or New Features](#suggesting-enhancements-or-new-features)
    *   [Pull Requests](#pull-requests)
3.  [Development Setup](#development-setup)
4.  [Git Workflow](#git-workflow)
    *   [Branching](#branching)
    *   [Commit Messages](#commit-messages)
5.  [Coding Standards](#coding-standards)
    *   [Python Style](#python-style)
    *   [Type Hinting](#type-hinting)
    *   [Docstrings](#docstrings)
6.  [Testing](#testing)
7.  [Documentation](#documentation)
8.  [Managing Dependencies (Poetry)](#managing-dependencies-poetry)
9.  [Configuration (Hydra)](#configuration-hydra)

## 1. Code of Conduct

This project and everyone participating in it is governed by a [Code of Conduct](CODE_OF_CONDUCT.md) (You'll need to create this file, often based on a standard template like the Contributor Covenant). Please adhere to it in all your interactions with the project.

## 2. How Can I Contribute?

### Reporting Bugs

*   Before submitting a bug report, please check the existing [GitHub Issues](https://github.com/AaronKAggarwal/structural-probe-repl/issues) to see if the bug has already been reported.
*   If you can't find an open issue addressing the problem, [open a new one](https://github.com/AaronKAggarwal/structural-probe-repl/issues/new).
*   Be sure to include a **title and clear description**, as much relevant information as possible, and a **code sample or an executable test case** demonstrating the expected behavior that is not occurring. Include details about your environment (OS, Python version, library versions).

### Suggesting Enhancements or New Features

*   Open an issue on GitHub and provide a clear description of the proposed enhancement or feature.
*   Explain why this enhancement would be useful and provide examples if possible.
*   Be open to discussion and refinement of your idea.

### Pull Requests

Pull Requests (PRs) are the primary way to contribute code or documentation changes.

1.  **Fork the Repository:** If you are an external contributor.
2.  **Create a Feature Branch:** Create a new branch from the `main` branch for your changes (see [Branching](#branching)).
3.  **Make Your Changes:** Implement your feature or bug fix.
    *   Ensure your code adheres to the [Coding Standards](#coding-standards).
    *   Add or update unit tests for your changes (see [Testing](#testing)).
    *   Update relevant documentation (see [Documentation](#documentation)).
4.  **Test Your Changes:** Run all tests locally (`poetry run pytest`) to ensure nothing is broken.
5.  **Commit Your Changes:** Follow the [Commit Message](#commit-messages) guidelines.
6.  **Push to Your Fork/Branch.**
7.  **Open a Pull Request:**
    *   Target the `main` branch of the upstream repository.
    *   Provide a clear title and description for your PR, linking to any relevant issues.
    *   Explain the changes you've made and why.
    *   Ensure all CI checks (if configured) pass.
8.  **Address Review Comments:** Be responsive to feedback and make necessary changes.

## 3. Development Setup

Please refer to `docs/ENV_SETUP.md` for detailed instructions on setting up the native macOS development environment using Poetry and the Dockerized environments for legacy code or CUDA operations.

## 4. Git Workflow

### Branching

*   **`main` branch:** This branch represents the stable, main line of development. Direct pushes to `main` are generally discouraged (except for maintainers after review or for CI-driven merges).
*   **Feature Branches:** All new features, bug fixes, or significant changes should be developed on separate feature branches.
    *   Create feature branches from the latest `main`.
    *   Naming convention: `feat/<short-description>`, `fix/<issue-number>-<short-description>`, `docs/<area>-update`, etc.
    *   Example: `git checkout -b feat/add-nonlinear-probe`
*   **Pull Requests:** Once a feature branch is complete, open a Pull Request to merge it into `main`.

### Commit Messages

Please follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. This helps create a more readable and maintainable Git history.

*   **Format:** `<type>(<scope>): <short summary>`
    *   `<type>`: `feat` (new feature), `fix` (bug fix), `docs` (documentation), `style` (formatting, no code change), `refactor`, `test`, `chore` (build process, etc.), `perf` (performance improvement).
    *   `(<scope>)`: Optional, e.g., `(dataset)`, `(probe_models)`, `(ci)`.
    *   `<short summary>`: Concise description in present tense. Start with a capital letter. No period at the end.
*   **Body (Optional):** Provide more context after a blank line.
*   **Footer (Optional):** For `BREAKING CHANGE:` notes or issue linking (e.g., `Fixes #123`).

**Example:**
```
feat(probe): Add support for Positive Semi-Definite constraint

Implemented the B^T B parameterization to ensure PSD matrix for the
distance probe, aligning more closely with Hewitt & Manning's description.

Refs #42 
```

## 5. Coding Standards

### Python Style

*   Follow **PEP 8** style guidelines.
*   Use an auto-formatter like **Black** (`poetry run black .`) and a linter like **Ruff** (`poetry run ruff check .`). Configure these as pre-commit hooks if possible.
*   Maximum line length: Aim for 88-100 characters (Black default is 88).

### Type Hinting

*   Use Python type hints for all function signatures and key variables (`from typing import ...`).
*   Run `mypy` (`poetry run mypy src scripts tests`) to check static types.

### Docstrings

*   Write clear and concise docstrings for all public modules, classes, and functions.
*   Follow a standard format (e.g., Google style, NumPy style, or reStructuredText).
    *   **Example (Google Style):**
        ```python
        def my_function(param1: int, param2: str) -> bool:
            """Does something interesting.

            Args:
                param1: The first parameter.
                param2: The second parameter.

            Returns:
                True if successful, False otherwise.
            
            Raises:
                ValueError: If param1 is negative.
            """
            # ... code ...
        ```
*   Refer to `docs/DOCSTRING_GUIDE.md` for more specific project conventions if it exists.

## 6. Testing

*   All new features and bug fixes **must** include corresponding unit tests.
*   Use the `pytest` framework.
*   Place unit tests for `src/torch_probe/module.py` in `tests/unit/torch_probe/test_module.py`.
*   Place integration/smoke tests in `tests/smoke/`.
*   Tests should be self-contained and not rely on external data files beyond small, fixture-generated ones where possible.
*   Run all tests before submitting a PR: `poetry run pytest`

## 7. Documentation

*   Keep documentation up-to-date with code changes.
*   If you add new modules, scripts, or significantly change architecture, update `docs/ARCHITECTURE.md`.
*   If you encounter new "quirks" or solve tricky issues, add them to `docs/QUIRKS.md`.
*   Log significant development steps or decisions in `docs/HISTORY.md`.
*   Update `docs/ENV_SETUP.md` or `docs/DEPENDENCIES.md` if environment requirements change.
*   Ensure `docs/DOC_INDEX.md` is current.
*   Write clear docstrings (see [Docstrings](#docstrings)).

## 8. Managing Dependencies (Poetry)

*   Add new dependencies: `poetry add <package_name>` (for main deps) or `poetry add <package_name> --group dev` (for dev deps).
*   Update dependencies: `poetry update <package_name>` or `poetry update`.
*   Always commit both `pyproject.toml` and `poetry.lock` after dependency changes.

## 9. Configuration (Hydra)

*   Experiments are configured using Hydra YAML files in the `configs/` directory.
*   Follow the existing structure (main `config.yaml`, group subdirectories, experiment files).
*   Refer to `docs/EXPERIMENT_PROTOCOL.md` for details on using Hydra.
*   New configurable components should have corresponding config group files.


By following these guidelines, we can maintain a high-quality, understandable, and collaborative research project. If you have questions, please open an issue or discuss with the project maintainer(s).

**Note to self on `CONTRIBUTING.md`:**

*   **Placeholders:** It mentions `CODE_OF_CONDUCT.md` and `docs/DOCSTRING_GUIDE.md`. Would need to create these (often using standard templates and then customising).
*   **Customisation:** You'll need to replace placeholders like `[Your Name <you@example.com>]` and the GitHub repo URL if it's different.
*   **Pre-commit Hooks:** It suggests pre-commit hooks for Black/Ruff. This is a very good practice to enforce style automatically. You would set this up separately (e.g., with a `.pre-commit-config.yaml` file).
*   **Evolution:** This document can evolve as the project grows and team practices are refined.