"""OpenMM System creation, restraints, and Simulation object setup."""

from openmm import (
    Platform, CMMotionRemover, LangevinMiddleIntegrator,
    MonteCarloBarostat,
)
from openmm.app import Simulation, LJPME, HBonds
from openmm.unit import nanometers, amu, femtoseconds

from .config import SimConfig
from .utils import (
    make_posres_force, add_heavy_atom_restraints,
    add_backbone_restraints,
)


def build_system(modeller, ff, cfg: SimConfig):
    """Create the OpenMM System with restraints and barostat.

    Returns (system, posres_heavy, posres_backbone, barostat).
    """
    timestep = cfg.timestep_fs * femtoseconds
    cmode = cfg.constraints_mode
    print(f"Using timestep={cfg.timestep_fs} fs, "
          f"HMR={cfg.hmr_mass_amu} amu, "
          f"constraints={'HBonds' if cmode is HBonds else 'AllBonds'}, "
          f"LJPME")

    system = ff.createSystem(
        modeller.topology,
        nonbondedMethod=LJPME,
        nonbondedCutoff=0.9 * nanometers,
        constraints=cmode,
        rigidWater=True,
        ewaldErrorTolerance=5e-4,
        hydrogenMass=cfg.hmr_mass_amu * amu,
    )
    system.addForce(CMMotionRemover())

    # Restraint forces
    posres_heavy = make_posres_force("k_posres_heavy")
    posres_backbone = make_posres_force("k_posres_bb")
    n_h = add_heavy_atom_restraints(
        posres_heavy, modeller.topology, modeller.positions)
    n_bb = add_backbone_restraints(
        posres_backbone, modeller.topology, modeller.positions)
    print(f"Restraints: heavy={n_h}, backbone={n_bb} atoms")

    system.addForce(posres_heavy)
    system.addForce(posres_backbone)

    # Barostat (disabled initially via frequency=0)
    barostat = MonteCarloBarostat(cfg.pressure, cfg.temperature, 0)
    system.addForce(barostat)
    print("Barostat added (frequency=0, disabled for NVT)")

    return system, posres_heavy, posres_backbone, barostat


def create_simulation(modeller, system, cfg: SimConfig):
    """Build integrator and Simulation; set initial positions."""
    timestep = cfg.timestep_fs * femtoseconds
    integrator = LangevinMiddleIntegrator(
        cfg.temperature, cfg.friction, timestep)
    integrator.setConstraintTolerance(1e-6)

    sim = None
    for name, kwargs in [
        ('CUDA', {"Precision": "mixed"}),
        ('OpenCL', {}),
        ('CPU', {}),
    ]:
        try:
            plat = Platform.getPlatformByName(name)
            sim = Simulation(modeller.topology, system, integrator,
                             plat, kwargs) if kwargs else \
                  Simulation(modeller.topology, system, integrator,
                             plat)
            print(f"Using platform: {name}")
            break
        except Exception:
            continue

    if sim is None:
        sim = Simulation(modeller.topology, system, integrator)
        print("Using platform: Reference (fallback)")

    sim.context.setPositions(modeller.positions)
    return sim, integrator