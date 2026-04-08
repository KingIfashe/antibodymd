"""Multi-stage equilibration protocol (Stages 0–11)."""

import math
import os
import time
from sys import stdout

from openmm.app import (
    PDBFile, StateDataReporter, DCDReporter,
)
from openmm.unit import (
    kelvin, femtoseconds, picosecond, kilojoule_per_mole,
)

from .config import SimConfig
from .utils import (
    stamp, add_status_reporter, safe_remove_reporters,
    load_stage_progress, save_checkpoint_and_progress,
    load_checkpoint, update_restraint_positions,
    kcal_a2_to_kj_nm2, steps_from_ns, next_part_path,
)


def _run_minimization(sim, cfg, posres_heavy, posres_backbone, barostat):
    """Stage 0: three-phase energy minimization."""
    progress = load_stage_progress(cfg.stage_progress_path)
    if progress["last_completed_stage"] >= 0:
        print("Minimization already completed (resuming).")
        load_checkpoint(sim, cfg.chk_path)
        return

    stamp("STAGE 0 START: Energy minimization", cfg.status_path)
    barostat.setFrequency(0)

    # Phase 1
    sim.context.setParameter("k_posres_heavy",
                             kcal_a2_to_kj_nm2(100.0))
    sim.context.setParameter("k_posres_bb", 0.0)

    state = sim.context.getState(getEnergy=True)
    pe = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    print(f"Initial PE: {state.getPotentialEnergy()}")
    if not math.isfinite(pe):
        print("  WARNING: Initial energy non-finite; "
              "proceeding (OpenMM can resolve).")

    for phase, k, iters in [
        (1, 100.0, cfg.minimization_steps),
        (2, 10.0, cfg.minimization_steps // 2),
        (3, 0.0, cfg.minimization_steps // 2),
    ]:
        sim.context.setParameter("k_posres_heavy",
                                 kcal_a2_to_kj_nm2(k))
        if phase == 3:
            sim.context.setParameter("k_posres_bb", 0.0)
        t0 = time.time()
        sim.minimizeEnergy(
            tolerance=cfg.minimization_tolerance,
            maxIterations=iters)
        stamp(f"  Phase {phase} done (heavy={k}) | "
              f"wall={(time.time()-t0)/60:.1f} min",
              cfg.status_path)

    # Post-min check
    state = sim.context.getState(getEnergy=True, getPositions=True)
    pe = state.getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    pos = state.getPositions()
    n = sim.topology.getNumAtoms()

    if not math.isfinite(pe):
        raise SystemExit("FATAL: Post-min energy is non-finite.")
    pe_per = pe / n
    print(f"Post-min PE/atom: {pe_per:.1f} kJ/mol ({n} atoms)")
    if pe_per > cfg.max_postmin_energy_per_atom:
        raise SystemExit(
            f"FATAL: PE/atom ({pe_per:.1f}) exceeds threshold "
            f"({cfg.max_postmin_energy_per_atom}).")

    # Update restraint references
    print("Updating restraint references to minimized coords...")
    update_restraint_positions(posres_heavy, pos)
    update_restraint_positions(posres_backbone, pos)
    posres_heavy.updateParametersInContext(sim.context)
    posres_backbone.updateParametersInContext(sim.context)

    with open(f"{cfg.output_prefix}_minimized.pdb", 'w') as f:
        PDBFile.writeFile(sim.topology, pos, f)

    save_checkpoint_and_progress(sim, cfg.chk_path,
                                 cfg.stage_progress_path, 0)
    stamp("STAGE 0 DONE", cfg.status_path)


def _run_preheat(sim, cfg, integrator, barostat):
    """Stage 1: pre-heat settle (1 fs, 20k steps)."""
    progress = load_stage_progress(cfg.stage_progress_path)
    if progress["last_completed_stage"] >= 1:
        print("Pre-heat settle already completed.")
        if progress["last_completed_stage"] == 1:
            load_checkpoint(sim, cfg.chk_path)
        return

    stamp("STAGE 1 START: Pre-heat settle", cfg.status_path)
    barostat.setFrequency(0)

    dt_save = integrator.getStepSize()
    fric_save = integrator.getFriction()
    integrator.setStepSize(1.0 * femtoseconds)
    integrator.setFriction(5 / picosecond)

    sim.context.setParameter("k_posres_heavy",
                             kcal_a2_to_kj_nm2(50.0))
    sim.context.setParameter("k_posres_bb", 0.0)
    sim.context.setVelocitiesToTemperature(100 * kelvin)
    sim.step(20000)

    integrator.setStepSize(dt_save)
    integrator.setFriction(fric_save)

    save_checkpoint_and_progress(sim, cfg.chk_path,
                                 cfg.stage_progress_path, 1)
    stamp("STAGE 1 DONE", cfg.status_path)


def _run_nvt_heating(sim, cfg, integrator, barostat):
    """Stage 2: NVT temperature ramp 100K → target."""
    nvt_steps = steps_from_ns(cfg.equil_nvt_ns, cfg.nvt_timestep_fs)
    progress = load_stage_progress(cfg.stage_progress_path)
    if progress["last_completed_stage"] >= 2:
        print("NVT heating already completed.")
        if progress["last_completed_stage"] == 2:
            load_checkpoint(sim, cfg.chk_path)
        return

    T_end = cfg.temperature_k
    print(f"\n[Stage 2] NVT heating 100K → {T_end}K, "
          f"{cfg.equil_nvt_ns} ns at {cfg.nvt_timestep_fs} fs")
    stamp(f"STAGE 2 START: NVT heat", cfg.status_path)

    if progress["last_completed_stage"] == 1:
        load_checkpoint(sim, cfg.chk_path)

    barostat.setFrequency(0)
    dt_save = integrator.getStepSize()
    fric_save = integrator.getFriction()
    integrator.setStepSize(cfg.nvt_timestep_fs * femtoseconds)
    integrator.setFriction(5 / picosecond)

    sim.context.setParameter("k_posres_heavy",
                             kcal_a2_to_kj_nm2(50.0))
    sim.context.setParameter("k_posres_bb", 0.0)
    sim.context.setVelocitiesToTemperature(100 * kelvin)

    n_blocks = 200
    spb = max(1, nvt_steps // n_blocks)
    interval = max(1000, nvt_steps // 100)
    total_rep = sim.currentStep + nvt_steps

    rep_out = StateDataReporter(
        stdout, interval, step=True, temperature=True,
        potentialEnergy=True, speed=True, progress=True,
        remainingTime=True, totalSteps=total_rep)
    rep_stat = add_status_reporter(sim, total_rep, interval,
                                   cfg.status_path)
    sim.reporters.append(rep_out)

    block_chk = f"{cfg.output_prefix}_nvt_block_{os.getpid()}.chk"
    t0 = time.time()

    for block in range(n_blocks):
        with open(block_chk, "wb") as fh:
            sim.saveCheckpoint(fh)
        T = 100.0 + (T_end - 100.0) * block / max(1, n_blocks - 1)
        integrator.setTemperature(T * kelvin)
        try:
            sim.step(spb)
        except Exception as e:
            stamp(f"WARNING: NVT block {block} failed: {e}",
                  cfg.status_path)
            with open(block_chk, "rb") as fh:
                sim.loadCheckpoint(fh)
            integrator.setStepSize(1.0 * femtoseconds)
            integrator.setFriction(10 / picosecond)
            sim.context.setParameter("k_posres_heavy",
                                     kcal_a2_to_kj_nm2(200.0))
            sim.context.setVelocitiesToTemperature(
                max(50.0, 0.5 * T) * kelvin)
            sim.step(min(5000, spb))
            sim.minimizeEnergy(
                tolerance=cfg.minimization_tolerance,
                maxIterations=3000)
            sim.context.setParameter("k_posres_heavy",
                                     kcal_a2_to_kj_nm2(50.0))
            integrator.setStepSize(cfg.nvt_timestep_fs * femtoseconds)
            integrator.setFriction(5 / picosecond)
            sim.context.setVelocitiesToTemperature(T * kelvin)
            sim.step(spb)

    integrator.setStepSize(dt_save)
    integrator.setFriction(fric_save)
    safe_remove_reporters(sim, rep_out, rep_stat)

    if os.path.exists(block_chk):
        os.remove(block_chk)

    save_checkpoint_and_progress(sim, cfg.chk_path,
                                 cfg.stage_progress_path, 2)
    stamp(f"STAGE 2 DONE | wall={(time.time()-t0)/60:.1f} min",
          cfg.status_path)


def _run_gentle_npt(sim, cfg, integrator, barostat):
    """Stage 3: gentle NPT settle with cautious barostat."""
    progress = load_stage_progress(cfg.stage_progress_path)
    if progress["last_completed_stage"] >= 3:
        print("Gentle NPT settle already completed.")
        barostat.setFrequency(25)
        return

    print("\n[Stage 3] Gentle NPT settle (50 ps)")
    stamp("STAGE 3 START", cfg.status_path)

    if progress["last_completed_stage"] == 2:
        load_checkpoint(sim, cfg.chk_path)

    dt_save = integrator.getStepSize()
    fric_save = integrator.getFriction()
    integrator.setStepSize(2.0 * femtoseconds)
    integrator.setFriction(5 / picosecond)

    barostat.setFrequency(100)
    sim.context.setParameter("k_posres_heavy",
                             kcal_a2_to_kj_nm2(100.0))
    sim.context.setParameter("k_posres_bb", 0.0)
    sim.context.setVelocitiesToTemperature(cfg.temperature)
    sim.step(steps_from_ns(0.05, 2.0))

    barostat.setFrequency(25)
    integrator.setStepSize(dt_save)
    integrator.setFriction(fric_save)

    save_checkpoint_and_progress(sim, cfg.chk_path,
                                 cfg.stage_progress_path, 3)
    stamp("STAGE 3 DONE", cfg.status_path)


def _run_npt_stage(sim, cfg, integrator, barostat,
                   label, stage_idx, k_heavy, k_bb,
                   equil_traj_reporter_holder):
    """Run one NPT equilibration stage with crash recovery."""
    progress = load_stage_progress(cfg.stage_progress_path)
    if progress["last_completed_stage"] >= stage_idx:
        print(f"  [{label}] already completed. Skipping.")
        return

    prior = stage_idx - 1
    if (progress["last_completed_stage"] == prior
            and os.path.exists(cfg.chk_path)):
        load_checkpoint(sim, cfg.chk_path)

    stamp(f"STAGE {stage_idx} START: {label}", cfg.status_path)
    print(f"\n[Stage {stage_idx}] {label}")

    barostat.setFrequency(25)
    sim.context.setParameter("k_posres_heavy", k_heavy)
    sim.context.setParameter("k_posres_bb", k_bb)

    # Ensure equil trajectory reporter
    if equil_traj_reporter_holder[0] is None:
        path = f"{cfg.output_prefix}_equil_trajectory.dcd"
        if os.path.exists(path):
            path = next_part_path(
                f"{cfg.output_prefix}_equil_trajectory")
        rep = DCDReporter(path, cfg.equil_traj_interval)
        sim.reporters.append(rep)
        equil_traj_reporter_holder[0] = rep

    stage_steps = steps_from_ns(1.0, cfg.timestep_fs)
    interval = max(1000, stage_steps // 100)
    total_rep = sim.currentStep + stage_steps

    rep_out = StateDataReporter(
        stdout, interval, step=True, potentialEnergy=True,
        temperature=True, volume=True, density=True, speed=True,
        progress=True, remainingTime=True, totalSteps=total_rep)
    rep_stat = add_status_reporter(sim, total_rep, interval,
                                   cfg.status_path)
    sim.reporters.append(rep_out)

    t0 = time.time()
    try:
        sim.step(stage_steps)
    except Exception as e:
        stamp(f"NaN in {label}: {e}", cfg.status_path)
        safe_remove_reporters(sim, rep_out, rep_stat)
        load_checkpoint(sim, cfg.chk_path)

        dt_k = integrator.getStepSize()
        fric_k = integrator.getFriction()
        integrator.setStepSize(2.0 * femtoseconds)
        integrator.setFriction(5 / picosecond)
        barostat.setFrequency(100)
        sim.context.setParameter("k_posres_heavy",
                                 kcal_a2_to_kj_nm2(100.0))
        sim.context.setParameter("k_posres_bb", 0.0)
        sim.minimizeEnergy(tolerance=cfg.minimization_tolerance,
                           maxIterations=2000)
        sim.context.setVelocitiesToTemperature(cfg.temperature)
        sim.step(10000)

        integrator.setStepSize(dt_k)
        integrator.setFriction(fric_k)
        barostat.setFrequency(25)
        sim.context.setParameter("k_posres_heavy", k_heavy)
        sim.context.setParameter("k_posres_bb", k_bb)

        retry_total = sim.currentStep + stage_steps
        rep_out = StateDataReporter(
            stdout, interval, step=True, potentialEnergy=True,
            temperature=True, volume=True, density=True, speed=True,
            progress=True, remainingTime=True,
            totalSteps=retry_total)
        rep_stat = add_status_reporter(sim, retry_total, interval,
                                       cfg.status_path)
        sim.reporters.append(rep_out)
        try:
            sim.step(stage_steps)
        except Exception as e2:
            safe_remove_reporters(sim, rep_out, rep_stat)
            raise SystemExit(f"FATAL: {label} failed twice: {e2}")

    safe_remove_reporters(sim, rep_out, rep_stat)
    wall = (time.time() - t0) / 60.0
    stamp(f"STAGE {stage_idx} DONE: {label} | wall={wall:.1f} min",
          cfg.status_path)
    save_checkpoint_and_progress(sim, cfg.chk_path,
                                 cfg.stage_progress_path, stage_idx)


def run_equilibration(sim, cfg, integrator, barostat,
                      posres_heavy, posres_backbone):
    """Execute the full equilibration protocol (Stages 0–11)."""
    _run_minimization(sim, cfg, posres_heavy, posres_backbone,
                      barostat)
    _run_preheat(sim, cfg, integrator, barostat)
    _run_nvt_heating(sim, cfg, integrator, barostat)
    _run_gentle_npt(sim, cfg, integrator, barostat)

    k = kcal_a2_to_kj_nm2
    equil_traj_holder = [None]  # mutable holder for reporter ref

    stages = [
        ("NPT 1 ns (heavy=100)", 4, k(100.0), 0.0),
        ("NPT 1 ns (heavy=10)", 5, k(10.0), 0.0),
    ]
    for label, idx, kh, kb in stages:
        _run_npt_stage(sim, cfg, integrator, barostat,
                       label, idx, kh, kb, equil_traj_holder)

    # Stage 6: minimization with bb restraints
    progress = load_stage_progress(cfg.stage_progress_path)
    if progress["last_completed_stage"] < 6:
        if progress["last_completed_stage"] == 5:
            load_checkpoint(sim, cfg.chk_path)
        print("\n[Stage 6] Minimization (bb=10)")
        stamp("STAGE 6 START", cfg.status_path)
        sim.context.setParameter("k_posres_heavy", 0.0)
        sim.context.setParameter("k_posres_bb", k(10.0))
        sim.minimizeEnergy(tolerance=cfg.minimization_tolerance,
                           maxIterations=cfg.minimization_steps)
        save_checkpoint_and_progress(
            sim, cfg.chk_path, cfg.stage_progress_path, 6)
        stamp("STAGE 6 DONE", cfg.status_path)

    bb_stages = [
        ("NPT 1 ns (bb=10)", 7, 0.0, k(10.0)),
        ("NPT 1 ns (bb=1)", 8, 0.0, k(1.0)),
        ("NPT 1 ns (bb=0.1)", 9, 0.0, k(0.1)),
        ("NPT 1 ns (unrestrained)", 10, 0.0, 0.0),
    ]
    for label, idx, kh, kb in bb_stages:
        _run_npt_stage(sim, cfg, integrator, barostat,
                       label, idx, kh, kb, equil_traj_holder)

    # Detach equil traj reporter
    if equil_traj_holder[0] is not None:
        safe_remove_reporters(sim, equil_traj_holder[0])

    # Stage 11: mark complete
    progress = load_stage_progress(cfg.stage_progress_path)
    if progress["last_completed_stage"] < 11:
        if progress["last_completed_stage"] == 10:
            load_checkpoint(sim, cfg.chk_path)
        print("\nEquilibration complete. Saving structure...")
        pos = sim.context.getState(getPositions=True).getPositions()
        with open(f"{cfg.output_prefix}_equilibrated.pdb", 'w') as f:
            PDBFile.writeFile(sim.topology, pos, f)
        save_checkpoint_and_progress(
            sim, cfg.chk_path, cfg.stage_progress_path, 11)
        stamp("EQUILIBRATION COMPLETE", cfg.status_path)
    else:
        print("Equilibration already marked complete.")