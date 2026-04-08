"""Disulfide bond detection, creation, and verification."""

from openmm.unit import nanometers
from .utils import is_protein_res


def _sg_atom(res):
    for a in res.atoms():
        if a.name.upper() == 'SG':
            return a
    return None


def _protein_chains(top):
    return [ch for ch in top.chains()
            if any(is_protein_res(r) for r in ch.residues())]


def _collect_cys_by_chain(top, pos):
    by_chain = {}
    for ch in _protein_chains(top):
        lst = []
        for res in ch.residues():
            if res.name.upper() in ('CYS', 'CYX'):
                sg = _sg_atom(res)
                if sg is not None:
                    lst.append((res, sg.index, pos[sg.index]))
        if lst:
            by_chain[ch] = lst
    return by_chain


def _dist_nm(p, q):
    d = (p - q).value_in_unit(nanometers)
    return (d.x ** 2 + d.y ** 2 + d.z ** 2) ** 0.5


def nearest_interchain_sg_distance_nm(top, pos):
    cys = _collect_cys_by_chain(top, pos)
    chains = list(cys.keys())
    best = None
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            for (_, _, pi) in cys[chains[i]]:
                for (_, _, pj) in cys[chains[j]]:
                    d = _dist_nm(pi, pj)
                    if best is None or d < best:
                        best = d
    return best


def existing_ss_bonds(top):
    out = []
    for b in top.bonds():
        a1, a2 = b[0], b[1]
        if a1.name.upper() == 'SG' and a2.name.upper() == 'SG':
            out.append(((a1.residue, a1), (a2.residue, a2)))
    return out


def add_interchain_disulfides(modeller, max_nm=0.35, max_per_pair=2):
    top, pos = modeller.topology, modeller.positions
    cys = _collect_cys_by_chain(top, pos)

    cand = []
    chains = list(cys.keys())
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            for (ri, _, pi) in cys[chains[i]]:
                for (rj, _, pj) in cys[chains[j]]:
                    cand.append((_dist_nm(pi, pj), ri, rj))

    cand.sort(key=lambda x: x[0])
    used, per_pair, added = set(), {}, []

    for d, r1, r2 in cand:
        if d > max_nm:
            break
        key = tuple(sorted((id(r1.chain), id(r2.chain))))
        if per_pair.get(key, 0) >= max_per_pair:
            continue
        if id(r1) in used or id(r2) in used:
            continue
        try:
            modeller.addDisulfideBond(r1, r2)
        except Exception:
            continue
        added.append((d, r1, r2))
        used.update((id(r1), id(r2)))
        per_pair[key] = per_pair.get(key, 0) + 1

    return added


def add_ssbond_fallback(modeller, ssbond_records):
    top = modeller.topology
    res_lookup = {}
    for res in top.residues():
        key = (res.chain.id.strip(), res.id.strip())
        if res.name.upper() in ('CYS', 'CYX'):
            res_lookup[key] = res

    already_bonded = set()
    for b in top.bonds():
        a1, a2 = b[0], b[1]
        if a1.name.upper() == 'SG' and a2.name.upper() == 'SG':
            already_bonded.add(id(a1.residue))
            already_bonded.add(id(a2.residue))

    added = 0
    for (ch1, rid1, ch2, rid2) in ssbond_records:
        r1 = res_lookup.get((ch1, str(rid1)))
        r2 = res_lookup.get((ch2, str(rid2)))
        if r1 is None or r2 is None:
            continue
        if id(r1) in already_bonded or id(r2) in already_bonded:
            continue
        try:
            modeller.addDisulfideBond(r1, r2)
            already_bonded.update((id(r1), id(r2)))
            added += 1
            print(f"   SSBOND fallback: {ch1}:{rid1} <-> {ch2}:{rid2}")
        except Exception:
            continue
    return added


def summarize_ss(top):
    total = inter = 0
    per_pair = {}
    lbl = {ch: (ch.id.strip() or f"anon_{i}")
           for i, ch in enumerate(top.chains())}
    for b in top.bonds():
        a1, a2 = b[0], b[1]
        if a1.name.upper() == 'SG' and a2.name.upper() == 'SG':
            total += 1
            c1, c2 = a1.residue.chain, a2.residue.chain
            if c1 is not c2:
                inter += 1
                key = tuple(sorted((lbl[c1], lbl[c2])))
                per_pair[key] = per_pair.get(key, 0) + 1
    print(f"S-S bonds: total={total}, inter-chain={inter}, "
          f"by-pair={per_pair}")
    return total, inter


def parse_ssbond_records(inpath):
    ssbonds = []
    with open(inpath) as f:
        for line in f:
            if line.startswith("SSBOND"):
                try:
                    ch1 = line[15].strip()
                    rid1 = int(line[17:21].strip())
                    ch2 = line[29].strip()
                    rid2 = int(line[31:35].strip())
                    ssbonds.append((ch1, rid1, ch2, rid2))
                except (ValueError, IndexError):
                    continue
    return ssbonds