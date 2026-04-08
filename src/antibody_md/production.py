"""Production MD and final state output."""

import os
import time
from sys import stdout

from openmm.app import (
    PDBFile, StateDataReporter, DCDReporter, CheckpointReporter,
)

from .config import SimConfig
from .utils import (
    stamp, add_status_reporter, safe_remove_reporters,
    load_stage_progress, save_stage_progress,
    load_checkpoint, steps_from_ns, next_part_path,
)


def run_production(sim, cfg, barostat):
    """Run production NPT MD with checkpoint resume."""
    print("\n" + "=" * 60)
    print("PRODUCTION MD (NPT)")
    print("=" * 60)

    if os.path.exists(cfg.chk_path):
        load_checkpoint(sim, cfg.chk_path)

    sim.context.setParameter("k_posres_heavy", 0.0)
    sim.context.setParameter("k_posres_bb", 0.0)
    barostat.setFrequency(25)

    progress = load_stage_progress(cfg.stage_progress_path)
    if progress.get("production_start_step") is None:
        prod_start = sim.currentStep
        save_stage_progress(cfg.stage_progress_path, 11,
                            production_start_step=prod_start)
    else:
        prod_start = progress["production_start_step"]

    total = steps_from_ns(cfg.production_ns, cfg.timestep_fs)
    done = sim.currentStep - prod_start
    remaining = max(0, total - done)

    if remaining == 0:
        print("Production already completed.")
        return

    if done > 0:
        ns_done = done * cfg.timestep_fs / 1e6
        print(f"Resuming: {ns_done:.1f} ns done, "
              f"{remaining * cfg.timestep_fs / 1e6:.1f} ns left.")

    stamp(f"PRODUCTION START: {remaining} steps remaining",
          cfg.status_path)

    interval = max(1000, remaining // 200)
    total_rep = sim.currentStep + remaining

    rep_stat = add_status_reporter(sim, total_rep, interval,
                                   cfg.status_path)

    traj = f"{cfg.output_prefix}_trajectory.dcd"
    if os.path.exists(traj):
        traj = next_part_path(f"{cfg.output_prefix}_trajectory")
    print(f"Trajectory: {traj}")

    sim.reporters.append(DCDReporter(traj, cfg.report_interval))
    sim.reporters.append(StateDataReporter(
        f"{cfg.output_prefix}_data.csv", cfg.report_interval,
        step=True, time=True, potentialEnergy=True,
        kineticEnergy=True, totalEnergy=True,
        temperature=True, volume=True, density=True, speed=True))
    sim.reporters.append(
        CheckpointReporter(cfg.chk_path, cfg.checkpoint_interval))
    sim.reporters.append(StateDataReporter(
        stdout, cfg.report_interval,
        step=True, time=True, potentialEnergy=True,
        temperature=True, speed=True, progress=True,
        remainingTime=True, totalSteps=total_rep))

    t0 = time.time()
    sim.step(remaining)
    safe_remove_reporters(sim, rep_stat)
    stamp(f"PRODUCTION DONE | wall={(time.time()-t0)/3600:.2f} h",
          cfg.status_path)


def save_final_state(sim, cfg):
    """Write final PDB and state XML."""
    print("\nSaving final state...")
    pos = sim.context.getState(getPositions=True).getPositions()
    with open(f"{cfg.output_prefix}_final.pdb", 'w') as f:
        PDBFile.writeFile(sim.topology, pos, f)
    sim.saveState(f"{cfg.output_prefix}_final_state.xml")
    stamp("JOB COMPLETE", cfg.status_path)