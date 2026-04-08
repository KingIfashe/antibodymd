"""Command-line entry point."""

import argparse

from .config import SimConfig
from .utils import init_status_log, stamp
from .preparation import prepare_system
from .system import build_system, create_simulation
from .equilibration import run_equilibration
from .production import run_production, save_final_state


def parse_args():
    p = argparse.ArgumentParser(
        description="Run antibody MD simulation (AMBER19 + TIP3P-FB)")
    p.add_argument("--pdb_code", required=True,
                   help="PDB code, e.g. 1ezv (expects {code}.pdb)")
    p.add_argument("--production_ns", type=float, default=300.0,
                   help="Production run length in ns (default: 300)")
    p.add_argument("--timestep_fs", type=float, default=4.0,
                   help="Production timestep in fs (default: 4)")
    p.add_argument("--disulfide_method", default="auto",
                   choices=["auto", "manual"],
                   help="Disulfide handling method (default: auto)")
    return p.parse_args()


def main():
    args = parse_args()

    cfg = SimConfig(
        pdb_code=args.pdb_code,
        production_ns=args.production_ns,
        timestep_fs=args.timestep_fs,
        disulfide_method=args.disulfide_method,
    )

    init_status_log(cfg.status_path, cfg.output_prefix)
    stamp("JOB START: structure preparation", cfg.status_path)

    # 1. Prepare structure
    modeller, ff = prepare_system(cfg)

    # 2. Build OpenMM system
    print("\n" + "=" * 60)
    print("SYSTEM SETUP")
    print("=" * 60)
    stamp("Building system and restraints", cfg.status_path)

    system, posres_heavy, posres_bb, barostat = build_system(
        modeller, ff, cfg)
    sim, integrator = create_simulation(modeller, system, cfg)

    # 3. Equilibrate
    run_equilibration(sim, cfg, integrator, barostat,
                      posres_heavy, posres_bb)

    # 4. Production
    run_production(sim, cfg, barostat)

    # 5. Final output
    save_final_state(sim, cfg)

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()