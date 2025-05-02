# Master Documentation Index

Last updated: $(date +%F)

| Doc               | Path               | Purpose                         | Update rule                         |
|-------------------|--------------------|---------------------------------|-------------------------------------|
| README.md         | ./README.md        | Project overview                | On usage or layout changes          |
| Environment Setup | docs/ENV_SETUP.md  | How to install & configure env  | On any env or install step change   |
| Env Setup            | docs/ENV_SETUP.md        | Native + Docker install & freeze steps | Update on any env change |
| Dependency Notes     | docs/DEPENDENCIES.md     | Special pins, NumPy<2, Rust req        | Update when pins move    |
| Frozen requirements  | requirements-mps.txt     | Exact native deps via poetry export    | Regenerate after dep change |

