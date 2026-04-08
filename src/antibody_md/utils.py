"""Shared helpers: logging, checkpoints, restraint utilities."""

import json
import math
import os
from datetime import datetime

from openmm import CustomExternalForce, MonteCarloBarostat
from openmm.app import StateDataReporter
from openmm.unit import nanometers

AA3 = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
    'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
    'TYR', 'VAL', 'CYX',
}


def is_protein_res(res):
    return res.name.upper() in AA3


def steps_from_ns(ns: float, timestep_fs: float) -> int:
    return max(1, int(round(ns * 1e6 / float(timestep_fs))))


def kcal_a2_to_kj_nm2(k: float) -> float:
    return k * 418.4


def next_part_path(base_no_ext: str) -> str:
    i = 1
    while True:
        cand = f"{base_no_ext}.part{i}.dcd"
        if not os.path.exists(cand):
            return cand
        i += 1


# ── Status logging ──────────────────────────────────────────────────

def stamp(msg: str, status_path: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(status_path, "a", encoding="utf-8", errors="replace") as f:
        f.write(f"[{ts}] {msg}\n")


def init_status_log(status_path: str, prefix: str):
    with open(status_path, "w", encoding="utf-8", errors="replace") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{ts}] LOG START for {prefix}\n")


def add_status_reporter(sim, total_steps, interval, status_path):
    rep = StateDataReporter(
        status_path, interval,
        step=True, time=True, progress=True,
        speed=True, remainingTime=True, totalSteps=total_steps,
    )
    sim.reporters.append(rep)
    return rep


def safe_remove_reporters(simulation, *reporters):
    for rep in reporters:
        try:
            simulation.reporters.remove(rep)
        except ValueError:
            pass


# ── Stage progress / checkpoints ────────────────────────────────────

def load_stage_progress(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"last_completed_stage": -1, "production_start_step": None}


def save_stage_progress(path: str, stage_idx: int,
                        production_start_step=None):
    data = {"last_completed_stage": stage_idx}
    if production_start_step is not None:
        data["production_start_step"] = production_start_step
    else:
        prev = load_stage_progress(path)
        if prev.get("production_start_step") is not None:
            data["production_start_step"] = prev["production_start_step"]
    with open(path, "w") as f:
        json.dump(data, f)


def save_checkpoint_and_progress(simulation, chk_path, progress_path,
                                 stage_idx, production_start_step=None):
    with open(chk_path, "wb") as fh:
        simulation.saveCheckpoint(fh)
    save_stage_progress(progress_path, stage_idx,
                        production_start_step=production_start_step)


def load_checkpoint(simulation, chk_path):
    with open(chk_path, "rb") as fh:
        simulation.loadCheckpoint(fh)


# ── Restraint forces ────────────────────────────────────────────────

def make_posres_force(global_name: str) -> CustomExternalForce:
    expr = (
        f"0.5*{global_name}"
        f"*periodicdistance(x, y, z, x0, y0, z0)^2"
    )
    f = CustomExternalForce(expr)
    f.addGlobalParameter(global_name, 0.0)
    f.addPerParticleParameter("x0")
    f.addPerParticleParameter("y0")
    f.addPerParticleParameter("z0")
    return f


def add_heavy_atom_restraints(force, topology, ref_positions):
    skip = ('HOH', 'WAT', 'Na+', 'Cl-', 'NA', 'CL')
    count = 0
    for atom in topology.atoms():
        if atom.residue.name in skip:
            continue
        if atom.element is not None and atom.element.symbol == 'H':
            continue
        xyz = ref_positions[atom.index]
        force.addParticle(atom.index, [xyz.x, xyz.y, xyz.z])
        count += 1
    return count


def add_backbone_restraints(force, topology, ref_positions):
    skip = ('HOH', 'WAT', 'Na+', 'Cl-', 'NA', 'CL')
    count = 0
    for atom in topology.atoms():
        if atom.residue.name in skip:
            continue
        if atom.name in ('N', 'CA', 'C'):
            xyz = ref_positions[atom.index]
            force.addParticle(atom.index, [xyz.x, xyz.y, xyz.z])
            count += 1
    return count


def update_restraint_positions(force, new_positions):
    """Update per-particle reference positions to new coordinates."""
    for i in range(force.getNumParticles()):
        atom_idx, _ = force.getParticleParameters(i)
        xyz = new_positions[atom_idx]
        force.setParticleParameters(i, atom_idx,
                                    [xyz.x, xyz.y, xyz.z])


# ── Barostat helpers ────────────────────────────────────────────────

def system_has_barostat(system) -> bool:
    for i in range(system.getNumForces()):
        if isinstance(system.getForce(i), MonteCarloBarostat):
            return True
    return False


def get_barostat(system):
    for i in range(system.getNumForces()):
        f = system.getForce(i)
        if isinstance(f, MonteCarloBarostat):
            return f
    return None