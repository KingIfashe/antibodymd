"""Simulation configuration with sensible defaults."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from openmm.app import HBonds, AllBonds
from openmm.unit import (
    kelvin, bar, picosecond, nanometers, molar,
    kilojoule_per_mole, nanometer, amu,
)


@dataclass
class SimConfig:
    """All tuneable parameters for an antibody MD run."""

    # --- Input ---
    pdb_code: str = ""
    pdb_file: str = ""  # derived from pdb_code if empty

    # --- Disulfide handling ---
    disulfide_method: str = "auto"  # 'auto' or 'manual'
    manual_disulfides: List[Tuple[str, int, str, int]] = field(
        default_factory=list
    )
    require_interchain_dsb: bool = False
    interchain_dsb_max_dist_nm: float = 0.35
    max_interchain_per_pair: int = 2

    # --- Integrator / thermostat ---
    temperature_k: float = 298.0
    pressure_bar: float = 1.0
    friction_per_ps: float = 1.0
    timestep_fs: float = 4.0
    nvt_timestep_fs: float = 2.0

    # --- HMR ---
    @property
    def hmr_mass_amu(self) -> float:
        return 1.5 if self.timestep_fs <= 4.0 else 3.0

    @property
    def constraints_mode(self):
        return HBonds if self.timestep_fs <= 4.0 else AllBonds

    # --- Durations ---
    equil_nvt_ns: float = 1.0
    equil_npt_ns: float = 6.0
    production_ns: float = 300.0

    # --- Solvation ---
    padding_nm: float = 1.2
    ionic_strength_m: float = 0.15

    # --- Minimization ---
    minimization_steps: int = 5000
    minimization_tolerance_kjmol_nm: float = 10.0
    max_postmin_energy_per_atom: float = 0.0

    # --- Output frequency (steps) ---
    report_interval: int = 15000
    checkpoint_interval: int = 25000
    equil_traj_interval: int = 50000

    # --- Derived paths (set in __post_init__) ---
    output_prefix: str = ""
    chk_path: str = ""
    stage_progress_path: str = ""
    status_path: str = ""

    def __post_init__(self):
        if not self.pdb_file and self.pdb_code:
            self.pdb_file = f"{self.pdb_code}.pdb"
        if not self.output_prefix:
            self.output_prefix = self.pdb_code
        if not self.chk_path:
            self.chk_path = f"{self.output_prefix}_checkpoint.chk"
        if not self.stage_progress_path:
            self.stage_progress_path = (
                f"{self.output_prefix}_stage_progress.json"
            )
        if not self.status_path:
            self.status_path = f"{self.output_prefix}_status.log"

    # --- Unit-bearing helpers ---
    @property
    def temperature(self):
        return self.temperature_k * kelvin

    @property
    def pressure(self):
        return self.pressure_bar * bar

    @property
    def friction(self):
        return self.friction_per_ps / picosecond

    @property
    def padding(self):
        return self.padding_nm * nanometers

    @property
    def ionic_strength(self):
        return self.ionic_strength_m * molar

    @property
    def minimization_tolerance(self):
        return self.minimization_tolerance_kjmol_nm * (
            kilojoule_per_mole / nanometer
        )