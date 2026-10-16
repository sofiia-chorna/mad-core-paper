from __future__ import annotations
import re
import numpy as np

_LAT = re.compile(r'Lattice="([^"]*)"')
_PBC = re.compile(r'pbc="([^"]*)"')
_PROPS = re.compile(r'Properties=(\S+)')
_ENERGY = re.compile(r'(?:^|\s)energy=(-?[\d.eE+]+)')
_ENTRY = re.compile(r'entry_id=(\S+)')
_STRUCTID = re.compile(r'structure_id=(\S+)')   # OPTIMADE xyz (mcloud/oqmd/...) native id


def parse_properties(props_str):
    toks = props_str.split(":")
    cols = {}
    start = 0
    for i in range(0, len(toks) - 2, 3):
        name, _dtype, count = toks[i], toks[i + 1], int(toks[i + 2])
        cols[name] = (start, count)
        start += count
    return cols


def parse_comment(comment):
    m = _LAT.search(comment)
    cell = (np.array(m.group(1).split(), dtype=float).reshape(3, 3)
            if m and m.group(1).strip() else None)
    mp = _PBC.search(comment)
    pbc = (np.array([t.upper().startswith("T") for t in mp.group(1).split()], dtype=bool)
           if mp else None)
    me = _ENERGY.search(comment)
    energy = float(me.group(1)) if me else None
    mpr = _PROPS.search(comment)
    props = parse_properties(mpr.group(1)) if mpr else {}
    men = _ENTRY.search(comment)
    msi = _STRUCTID.search(comment)
    native_id = men.group(1) if men else (msi.group(1) if msi else None)
    return cell, pbc, energy, props, native_id


def iter_frames(stream):
    while True:
        first = stream.readline()
        if not first:
            return
        first = first.strip()
        if not first:
            continue
        natoms = int(first)
        comment = stream.readline().rstrip("\n")
        block = [stream.readline() for _ in range(natoms)]
        yield natoms, comment, block


def parse_block_columns(block, natoms, col_specs):
    out = {k: (np.empty((natoms, n)) if k != "species" else [None] * natoms)
           for k, _s, n in col_specs}
    for i in range(natoms):
        t = block[i].split()
        for k, s, n in col_specs:
            if k == "species":
                out[k][i] = t[s]
            else:
                out[k][i] = [float(t[s + j]) for j in range(n)]
    return out
