import sys
from pathlib import Path

from mad.madcore_vs_random_20k.xyz_write import SYMBOLS

TARGET_Z = set(range(1, 103))
Z_OF = {symbol: z for z, symbol in enumerate(SYMBOLS)}


def elements_of_dir(out_dir):
    present = set()

    for name in ("ttt", "fff", "surfaces"):
        with open(out_dir / f"{name}.extxyz") as fh:
            while True:
                count_line = fh.readline()
                if not count_line:
                    break
                n = int(count_line)
                fh.readline()
                for _ in range(n):
                    present.add(Z_OF[fh.readline().split()[0]])

    return present


def main(dirs):
    for out_dir in dirs:
        out_dir = Path(out_dir)
        present = elements_of_dir(out_dir)
        missing = sorted(TARGET_Z - present)
        extra = sorted(present - TARGET_Z)

        print(f"{out_dir.name}: {len(present)} elements", end="")
        if missing:
            print(f"  MISSING {len(missing)}: {[SYMBOLS[z] for z in missing]}", end="")
        if extra:
            print(f"  beyond Z=102: {[SYMBOLS[z] for z in extra]}", end="")
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
