# antibody-md

OpenMM-based molecular dynamics pipeline for antibody structures with AMBER19/TIP3P-FB force fields, automated disulfide handling, and a multi-stage equilibration protocol.

## Features

- **AMBER19 + TIP3P-FB** with LJPME long-range LJ interactions
- **Automated disulfide detection** with SSBOND record fallback and inter-chain escalation
- **Hydrogen mass repartitioning (HMR)** for 4 fs production timesteps
- **Temperature ramping** (100K → 298K) to prevent hot spots
- **Multi-stage restraint ladder**: heavy atoms (100→10 kcal/Å²) → backbone (10→1→0.1) → unrestrained
- **Stage-aware checkpoint resume** — restart from any stage
- **Crash recovery** with automatic rollback and emergency stabilization

## Installation

```bash
pip install git+https://github.com/kingifashe/antibody-md.git
```

Or for development:

```bash
git clone https://github.com/kingifashe/antibody-md.git
cd antibody-md
pip install -e .
```

### Prerequisites

OpenMM must be installed (typically via conda):

```bash
conda install -c conda-forge openmm pdbfixer
pip install -e .
```

## Usage

### Command line

```bash
# Place your PDB file in the working directory (e.g., 1ezv.pdb)
antibody-md --pdb_code 1ezv

# With options
antibody-md --pdb_code 1ezv --production_ns 500 --timestep_fs 4 --disulfide_method auto
```

### Python API

```python
from antibody_md.config import SimConfig
from antibody_md.preparation import prepare_system
from antibody_md.system import build_system, create_simulation
from antibody_md.equilibration import run_equilibration
from antibody_md.production import run_production, save_final_state
from antibody_md.utils import init_status_log, stamp

cfg = SimConfig(pdb_code="1ezv", production_ns=100)
init_status_log(cfg.status_path, cfg.output_prefix)

modeller, ff = prepare_system(cfg)
system, posres_heavy, posres_bb, barostat = build_system(modeller, ff, cfg)
sim, integrator = create_simulation(modeller, system, cfg)

run_equilibration(sim, cfg, integrator, barostat, posres_heavy, posres_bb)
run_production(sim, cfg, barostat)
save_final_state(sim, cfg)
```

## Protocol

| Stage | Type | Duration | Restraints |
|-------|------|----------|------------|
| 0 | Energy minimization (3-phase) | — | Heavy 100→10→0 |
| 1 | Pre-heat settle (1 fs) | 20 ps | Heavy 50 |
| 2 | NVT heating 100→298K (2 fs) | 1 ns | Heavy 50 |
| 3 | Gentle NPT settle (2 fs) | 50 ps | Heavy 100 |
| 4–5 | NPT equilibration | 2 × 1 ns | Heavy 100→10 |
| 6 | Minimization | — | Backbone 10 |
| 7–9 | NPT equilibration | 3 × 1 ns | Backbone 10→1→0.1 |
| 10 | NPT unrestrained | 1 ns | None |
| 11+ | Production NPT (4 fs, HMR) | 300 ns | None |

## Output Files

| File | Description |
|------|-------------|
| `{code}_solvated.pdb` | Prepared solvated system |
| `{code}_minimized.pdb` | Post-minimization structure |
| `{code}_equilibrated.pdb` | Post-equilibration structure |
| `{code}_equil_trajectory.dcd` | Equilibration trajectory |
| `{code}_final.pdb` | Final structure |
| `{code}_trajectory.dcd` | Production trajectory |
| `{code}_data.csv` | Production thermodynamic data |
| `{code}_checkpoint.chk` | Restart checkpoint |
| `{code}_status.log` | Timestamped progress log |

## Project Structure

```
src/antibody_md/
├── __init__.py          # Package version
├── cli.py               # CLI entry point
├── config.py            # SimConfig dataclass (all tunables)
├── utils.py             # Logging, checkpoints, restraint helpers
├── disulfides.py        # S-S bond detection & modeling
├── preparation.py       # PDBFixer, solvation pipeline
├── system.py            # OpenMM System & Simulation creation
├── equilibration.py     # Multi-stage equilibration (Stages 0–11)
└── production.py        # Production MD & final output
```

## License

MIT