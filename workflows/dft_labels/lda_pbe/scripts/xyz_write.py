import numpy as np

SYMBOLS = (
    "X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni "
    "Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe "
    "Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg "
    "Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg "
    "Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og"
).split()


def vacuum_axes(positions, cell, min_gap=6.0):
    frac = np.linalg.solve(cell.T, positions.T).T % 1.0
    volume = abs(np.linalg.det(cell))

    count = 0
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        height = volume / np.linalg.norm(np.cross(cell[j], cell[k]))
        s = np.sort(frac[:, i])
        gap = np.diff(s, append=s[0] + 1.0).max()
        if gap * height > min_gap:
            count += 1

    return count


def classify(positions, cell, pbc_row):
    n_periodic = int(pbc_row.sum())

    if n_periodic == 0:
        return "fff", -1
    if n_periodic < 3:
        return "surfaces", -1

    vac = vacuum_axes(np.asarray(positions, dtype=np.float64),
                      np.asarray(cell, dtype=np.float64))
    if vac == 0:
        return "ttt", 0
    if vac == 3:
        return "fff", 3
    return "surfaces", vac


def write_frame(fh, numbers, positions, cell, pbc_row, info):
    for value in info.values():
        assert '"' not in value and "\n" not in value

    tokens = [f'{key}="{value}"' for key, value in info.items()]

    if pbc_row.any():
        lattice = " ".join(f"{v:.8f}" for v in np.asarray(cell, dtype=np.float64).ravel())
        tokens.insert(0, f'Lattice="{lattice}"')

    pbc_str = " ".join("T" if flag else "F" for flag in pbc_row)
    tokens.append('Properties=species:S:1:pos:R:3')
    tokens.append(f'pbc="{pbc_str}"')

    fh.write(f"{len(numbers)}\n")
    fh.write(" ".join(tokens) + "\n")

    for z, (px, py, pz) in zip(numbers, positions):
        fh.write(f"{SYMBOLS[z]} {px:.8f} {py:.8f} {pz:.8f}\n")
