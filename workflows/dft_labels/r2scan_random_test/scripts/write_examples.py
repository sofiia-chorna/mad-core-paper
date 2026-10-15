import json

from paths import BUNDLE, PBE_MANIFEST, PBE_XYZ, parse_frames, read_tsv

EXAMPLES = {"isolated_pbe_frame_00792": ("isolated_fff", "792"), "periodic_pbe_frame_00000": ("periodic_ttt", "0"),
            "sample_member_rank0015_isolated_pbe": ("isolated_fff", "5")}


def main():
    manifest = read_tsv(PBE_MANIFEST)
    cid_of = {(r["dft_bucket"], r["frame_id"]): r["configuration_id"] for r in manifest}
    wanted = {cid_of[v]: k for k, v in EXAMPLES.items()}
    with open(PBE_XYZ, "rb") as fh:
        for offset, info, numbers, positions, cell, pbc in parse_frames(PBE_XYZ):
            cid = info["configuration_id"]
            if cid not in wanted:
                continue
            fh.seek(offset)
            n = int(fh.readline())
            fh.readline()
            forces = [[float(v) for v in fh.readline().split()[4:7]] for _ in range(n)]
            expected = {"configuration_id": cid, "functional": "pbe", "natoms": n, "energy_free_eV": float(info["energy"]),
                        "forces_eV_A": forces, "source": "lda_pbe/labelled/random_test_6k/pbe.xyz (electronic free energy convention)"}
            (BUNDLE / "examples" / wanted[cid] / "expected.json").write_text(json.dumps(expected, indent=1))
            print(wanted[cid], cid, n, info["energy"])


if __name__ == "__main__":
    main()
