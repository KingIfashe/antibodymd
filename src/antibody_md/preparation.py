"""Structure preparation: PDB sanitization, fixing, disulfides, solvation."""

import os
import re
import tempfile

import pdbfixer
from openmm.app import ForceField, Modeller, PDBFile, HBonds

from .config import SimConfig
from .disulfides import (
    existing_ss_bonds, add_interchain_disulfides, add_ssbond_fallback,
    summarize_ss, nearest_interchain_sg_distance_nm, parse_ssbond_records,
    _protein_chains,
)


def sanitize_pdb(inpath: str, pdb_code: str) -> str:
    """Strip CONECT/LINK/SSBOND/CISPEP records. Returns temp path."""
    pid = os.getpid()
    basename = os.path.splitext(os.path.basename(inpath))[0]
    outpath = os.path.join(
        tempfile.gettempdir(),
        f"antibody_clean_{basename}_{pid}.pdb",
    )
    skip = re.compile(r'^(CONECT|LINK|LINKR|SSBOND|CISPEP)\b')
    with open(inpath) as fin, open(outpath, 'w') as fout:
        for line in fin:
            if not skip.match(line):
                fout.write(line)
    return outpath


def fix_structure(pdb_path: str):
    """Run PDBFixer: replace non-standard residues, add missing."""
    print("\nLoading PDB structure with PDBFixer...")
    fixer = pdbfixer.PDBFixer(filename=pdb_path)
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)

    fixer.findMissingResidues()
    if fixer.missingResidues:
        print(f"  Adding {len(fixer.missingResidues)} missing residues")
        fixer.addMissingResidues()

    fixer.findMissingAtoms()
    if fixer.missingAtoms:
        print(f"  Adding missing atoms in "
              f"{len(fixer.missingAtoms)} residues")
        fixer.addMissingAtoms()
    else:
        print("  No missing heavy atoms")

    return fixer


def _apply_disulfides_auto(modeller, cfg: SimConfig, ssbond_records):
    """Auto-detect disulfides with escalating cutoffs + SSBOND fallback."""
    print("-> Step 1: Topology.createDisulfideBonds (auto)")
    pre = len(existing_ss_bonds(modeller.topology))
    modeller.topology.createDisulfideBonds(modeller.positions)
    post = len(existing_ss_bonds(modeller.topology))
    print(f"   Auto added {post - pre} disulfide(s).")

    print(f"-> Step 2: Force inter-chain S-S under "
          f"{cfg.interchain_dsb_max_dist_nm:.2f} nm (greedy)")
    for cutoff in (cfg.interchain_dsb_max_dist_nm, 0.40, 0.45):
        forced = add_interchain_disulfides(
            modeller, max_nm=cutoff,
            max_per_pair=cfg.max_interchain_per_pair,
        )
        if forced:
            print(f"   + Added {len(forced)} inter-chain S-S "
                  f"at cutoff {cutoff:.2f} nm:")
            for d, r1, r2 in forced:
                print(f"     - {r1.chain.id}:{r1.id}-{r1.name}SG <-> "
                      f"{r2.chain.id}:{r2.id}-{r2.name}SG  "
                      f"(d={d:.3f} nm)")
            break
        else:
            nearest = nearest_interchain_sg_distance_nm(
                modeller.topology, modeller.positions)
            tag = (f"nearest={nearest:.3f} nm" if nearest is not None
                   else "no cross-chain SG found")
            print(f"   + No eligible pairs under {cutoff:.2f} nm "
                  f"({tag}).")

    if ssbond_records:
        print("-> Step 3: SSBOND record fallback")
        n = add_ssbond_fallback(modeller, ssbond_records)
        print(f"   + Added {n} disulfide(s) from SSBOND records."
              if n else "   + No additional bonds needed.")


def _apply_disulfides_manual(modeller, cfg: SimConfig):
    """Apply manually-specified disulfide bonds."""
    print("-> Using MANUAL disulfide specification")
    res_lookup = {}
    for res in modeller.topology.residues():
        key = (res.chain.id.strip(), res.id.strip())
        if res.name.upper() in ('CYS', 'CYX'):
            res_lookup[key] = res

    count = 0
    for (ch1, rid1, ch2, rid2) in cfg.manual_disulfides:
        r1 = res_lookup.get((ch1, str(rid1)))
        r2 = res_lookup.get((ch2, str(rid2)))
        if r1 is None or r2 is None:
            print(f"   WARNING: residues not found for "
                  f"{ch1}:{rid1} <-> {ch2}:{rid2}")
            continue
        try:
            modeller.addDisulfideBond(r1, r2)
            count += 1
            print(f"   Manual S-S: {ch1}:{rid1} <-> {ch2}:{rid2}")
        except Exception as e:
            print(f"   WARNING: Failed {ch1}:{rid1} <-> "
                  f"{ch2}:{rid2}: {e}")
    print(f"   Added {count} manual disulfide(s).")


def _add_hydrogens(modeller, ff, ssbond_records):
    """Add hydrogens; fall back to PDBFixer if FF addH fails."""
    pre_h_ss = set()
    for b in modeller.topology.bonds():
        a1, a2 = b[0], b[1]
        if a1.name.upper() == 'SG' and a2.name.upper() == 'SG':
            pre_h_ss.add(id(a1.residue))
            pre_h_ss.add(id(a2.residue))

    try:
        modeller.addHydrogens(ff, pH=7.0)
        print("  addHydrogens succeeded (primary path).")
        return modeller
    except Exception as e:
        print(f"  WARNING: addHydrogens failed: {e}")
        print("  Falling back to PDBFixer hydrogen addition...")

    with open("preH_noH.pdb", "w") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f,
                          keepIds=True)

    fx = pdbfixer.PDBFixer(filename="preH_noH.pdb")
    fx.addMissingHydrogens(pH=7.0)
    modeller = Modeller(fx.topology, fx.positions)
    modeller.topology.createDisulfideBonds(modeller.positions)
    if ssbond_records:
        add_ssbond_fallback(modeller, ssbond_records)

    n_cyx = sum(1 for r in modeller.topology.residues()
                if r.name.upper() == 'CYX')
    print(f"  After fallback: CYX residues = {n_cyx}")
    if n_cyx == 0 and pre_h_ss:
        raise SystemExit(
            "FATAL: CYX count is 0 but disulfides existed before "
            "H-addition.")
    return modeller


def _pdb_roundtrip(modeller, cfg, ssbond_records):
    """Normalize topology via PDB write/read cycle."""
    path = f"tmp_pre_solvate_{cfg.pdb_code}_{os.getpid()}.pdb"
    with open(path, "w") as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f,
                          keepIds=True)
    pdb = PDBFile(path)
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.topology.createDisulfideBonds(modeller.positions)

    total_ss, inter_ss = summarize_ss(modeller.topology)
    print(f"After PDB round-trip: S-S total={total_ss}, "
          f"inter-chain={inter_ss}")

    if (ssbond_records and cfg.require_interchain_dsb
            and inter_ss == 0):
        add_ssbond_fallback(modeller, ssbond_records)
        total_ss, inter_ss = summarize_ss(modeller.topology)
        if inter_ss == 0:
            raise SystemExit(
                "FAIL: Inter-chain S-S lost on PDB round-trip.")

    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass

    return modeller


def prepare_system(cfg: SimConfig):
    """Full preparation pipeline. Returns (modeller, ff)."""
    print("=" * 60)
    print("STRUCTURE PREPARATION")
    print("=" * 60)

    ssbond_records = parse_ssbond_records(cfg.pdb_file)
    if ssbond_records:
        print(f"Parsed {len(ssbond_records)} SSBOND record(s).")

    clean_path = sanitize_pdb(cfg.pdb_file, cfg.pdb_code)
    fixer = fix_structure(clean_path)

    print("\n" + "=" * 60)
    print("SOLVATION & DISULFIDES")
    print("=" * 60)

    modeller = Modeller(fixer.topology, fixer.positions)
    ff = ForceField('amber19/protein.ff19SB.xml',
                    'amber19/tip3pfb.xml')

    # Disulfides before hydrogens
    print("\n=== DISULFIDE MODELING ===")
    if cfg.disulfide_method == 'manual':
        _apply_disulfides_manual(modeller, cfg)
    else:
        _apply_disulfides_auto(modeller, cfg, ssbond_records)

    # Verify
    all_ss = existing_ss_bonds(modeller.topology)
    inter_ss = [b for b in all_ss
                if b[0][0].chain is not b[1][0].chain]
    print(f"-> Verification: total S-S={len(all_ss)}, "
          f"inter-chain={len(inter_ss)}")

    if cfg.require_interchain_dsb and not inter_ss:
        raise SystemExit("FAIL: No inter-chain disulfide modeled.")

    # Hydrogens
    print("\nAdding hydrogens at pH 7.0...")
    modeller = _add_hydrogens(modeller, ff, ssbond_records)
    total_ss, inter_count = summarize_ss(modeller.topology)

    if cfg.require_interchain_dsb and inter_count == 0:
        raise SystemExit("FAIL: No inter-chain S-S after H-add.")

    # PDB round-trip
    modeller = _pdb_roundtrip(modeller, cfg, ssbond_records)

    # Template precheck
    print("Template precheck...")
    try:
        _ = ff.createSystem(modeller.topology, constraints=HBonds,
                            rigidWater=False)
        print("  Protein templates OK.")
    except Exception as e:
        raise SystemExit(f"Template precheck failed: {e}")

    # Solvate
    print(f"\nAdding solvent (padding={cfg.padding}, "
          f"ionic_strength={cfg.ionic_strength})...")
    modeller.addSolvent(
        ff, model='tip3p', padding=cfg.padding,
        ionicStrength=cfg.ionic_strength, boxShape='octahedron',
    )
    print(f"Solvated system atoms: "
          f"{modeller.topology.getNumAtoms()}")

    with open(f"{cfg.output_prefix}_solvated.pdb", 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)

    # Cleanup temp files
    for p in [clean_path]:
        if os.path.exists(p) and p.startswith(tempfile.gettempdir()):
            try:
                os.remove(p)
            except OSError:
                pass

    return modeller, ff