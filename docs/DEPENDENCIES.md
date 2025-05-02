
---

### 1.2 `docs/DEPENDENCIES.md` (**new file**)

```markdown
# Dependency Notes

| Package / Tool | Pin / Version | Rationale |
|----------------|---------------|-----------|
| **Python** | 3.11 | Torch wheels exist; 3.12/3.13 unsupported |
| **Torch** | 2.2.0 | Stable MPS backend |
| **NumPy** | < 2 (1.26.4) | Torch 2.2 compiled against NumPy 1 headers |
| **Transformers** | 4.40.0 | Matches tokenizers constraint |
| **tokenizers** | 0.19.1 (built from source) | Needs Rust; required by Transformers 4.40 |
| **Rust tool‑chain** | `brew install rust` | Compiles tokenizers 0.19 |
| **poetry-plugin-export** | ≥ 1.8 | Provides `poetry export` command |
| **Docker Desktop** | 28.x, Rosetta enabled | Runs Linux containers for TF1 / CUDA |
