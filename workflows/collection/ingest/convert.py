from __future__ import annotations
import argparse
import importlib
import os

from .ids import mint_ids
from .normalize import ShardWriter
from .records import validate_record

DATASETS = {
    "OC20_S2EF_train_all": ("extxyz_xz", "OC20_S2EF_train_all"),
    "OMol25_train":        ("aselmdb", "OMol25_train"),
    "OMol25_val":          ("aselmdb", "OMol25_val"),
    "ODAC_train":          ("aselmdb", "ODAC_train"),
    "ODAC_val":            ("aselmdb", "ODAC_val"),
    "OC25_train":          ("aselmdb", "OC25_train"),
    "OC25_val":            ("aselmdb", "OC25_val"),
    "Alexandria_pbesol":   ("alexandria", "Alexandria_pbesol"),
    "Alexandria_pressure": ("alexandria", "Alexandria_pressure"),
    "nomad":               ("xyz", "nomad"),
    "mad_next":            ("mad_next", "mad_next"),
    "mad_sol":             ("mad_next", "mad_sol"),
    "mad_1.5":             ("mad15", "MAD-1.5"),
    "oc20neb":             ("traj", "oc20neb"),
    "mcloud__2dtopo":                   ("xyz", "mcloud__2dtopo"),
    "mcloud__autowannier":              ("xyz", "mcloud__autowannier"),
    "mcloud__curated-cofs":             ("xyz", "mcloud__curated-cofs"),
    "mcloud__mc2d":                     ("xyz", "mcloud__mc2d"),
    "mcloud__pyrene-mofs":              ("xyz", "mcloud__pyrene-mofs"),
    "mcloud__stoceriaitf":              ("xyz", "mcloud__stoceriaitf"),
    "mcloud__tin-antimony-sulfoiodide": ("xyz", "mcloud__tin-antimony-sulfoiodide"),
    "mcloudarchive__1z-pd":             ("xyz", "mcloudarchive__1z-pd"),
    "mcloudarchive__ed-g2":             ("xyz", "mcloudarchive__ed-g2"),
    "mcloudarchive__jk-9v":             ("xyz", "mcloudarchive__jk-9v"),
    "mcloudarchive__m0-zg":             ("xyz", "mcloudarchive__m0-zg"),
    "mcloudarchive__zt-z4":             ("xyz", "mcloudarchive__zt-z4"),
    "oqmd__oqmd":                       ("xyz", "oqmd__oqmd"),
    "gdb13":                            ("xyz", "gdb13"),
}

_MCLOUD_ARCHIVE_EXTRAS = [
    "1herw-7m923", "405an-d8183", "ann3c-0qq90", "zd984-2ja12", "nf76v-1eh14",
    "sd5dx-6yx66", "qyb0c-q8m61", "ffvcy-q4c20", "1j65w-wy733",
    "dyvt8-af249", "et7sa-3cv85", "0ewdw-fx861", "6txnj-70460", "e4fat-6ga07",
    "hx49x-ab323", "qe7x5-fbb10",
]
for _mc in _MCLOUD_ARCHIVE_EXTRAS:
    DATASETS[f"mcloudarchive__{_mc}"] = ("xyz", f"mcloudarchive__{_mc}")

_MCLOUD_TIERC = [
    "05j11-ky482", "0kxkb-nh787", "0x1nd-2pq96", "1fq3y-6yt09", "1fx58-pde31",
    "1z76q-e6n84", "238ad-1bg44", "2by8y-7yb79", "2eyh5-grx50", "342fz-5wx57",
    "3agdy-69a84", "3t20t-26564", "3yke8-xxf82", "45yfz-7xa87", "4j5g3-8wn28",
    "4m62e-np961", "55jga-d0447", "59y2w-rap91", "5j3ys-8b546", "5pdz3-vsq42",
    "5xcvw-k5073", "5y3ms-91j88", "61kg0-hy694", "6jc67-ars95", "6xm49-rz892",
    "74wkx-2ym98", "7m7rj-qcy74", "7qm51-anq57", "83e5g-ts228", "8sbz4-6ng07",
    "95z7z-zbr43", "9x8ac-2dg09", "a673y-gdt46", "afx8g-ra033", "anq8a-2ep59",
    "b4kgx-rtd78", "b9pbd-yb110", "bj9ac-bn156", "bmsyr-ysb48", "brdpj-j1e50",
    "bzch9-1xh79", "c00et-26a37", "c0hpy-pad70", "c6ngz-zrh49", "cdt4h-yvv48",
    "cedze-akd72", "crxpj-3gk87", "d5hgx-p9961", "d7k4k-xbw11", "e4wpj-x0f45",
    "eg3pn-1fw83", "emryk-yfa58", "es05d-9jc34", "ez9cv-dvd64", "f59hy-ben23",
    "f8k8b-6p870", "fcafh-h1367", "ffkqx-5ae17", "fmdsp-f5690", "fx32t-12t34",
    "gbgn3-00c60", "grmx8-xva78", "gxpcc-vq370", "h3b8e-ywx98", "h73gv-1q045",
    "hexw2-njc93", "hk0x5-xxq93", "hkbke-aje39", "hnjfh-nd118", "hpp9j-0g506",
    "hqa6g-z2386", "hry14-dk842", "hv7b1-71340", "hxcdd-t9527", "j144f-74t50",
    "j4v22-z2c33", "jcvjc-qar82", "jwwje-1nc89", "jxf1h-pe887", "jyjyw-fm563",
    "k5b15-1ne74", "kqv0y-v9k76", "kwfgp-94v27", "kxt46-z8911", "kyaqv-avg31",
    "m17qc-y7350", "nd7cc-cxx13", "ng82t-cj831", "nz8sk-6h010", "p8n7a-p5e75",
    "paxx3-a1k09", "pkhy1-ted45", "pqa4c-qk811", "qgggm-5bt60", "qp1ej-vst22",
    "r8e44-wf170", "razpa-n7327", "rntn6-t3h85", "rnve6-eas10", "s26za-yrg30",
    "s31sf-y9t40", "s48qh-yzy40", "sg18z-dav56", "sg5fz-amj89", "sjbf4-cfh83",
    "sqstg-vcd29", "tykd4-2aw12", "v57hk-6bm41", "vbd5n-pnp32", "vermx-08476",
    "vkhxg-f0f84", "vr2m8-p5570", "vtahk-cd853", "wxy8k-14m62", "xavbv-5h791",
    "xb7gc-shq49", "xc3bm-qc060", "xjz0c-tpe46", "xn8vs-6mb67", "xs5xv-k8r90",
    "ydn09-ngs56", "ym9h2-r9h78", "yssqm-36792", "z3g78-sq788", "z6g6d-mp618",
    "zxjdj-xme02", "zztez-8a187",
]
for _mc in _MCLOUD_TIERC:
    DATASETS[f"mcloudarchive__{_mc}"] = ("archive", f"mcloudarchive__{_mc}")

# MC archive records the title-keyword filter had skipped; re-scan (2026-06-19) found these contain structures.
_MCLOUD_RESCAN = [
    "00s2s-xre47", "02rcd-nc341", "06329-0n866", "07wfy-shs16", "08a12-p4a37", "08gsg-gpd20", "0db0e-t6d44",
    "0gg37-qvg37", "0hb0h-40r37", "0m9r0-g3486", "0p5ya-w1n45", "16jd7-0zz18", "1ap7q-cn115", "1bymd-by722",
    "1e98j-qj866", "1qfe8-htw65", "1zpzw-a7p58", "286v9-64865", "2aj5h-tzk43", "2cn5m-gek76", "2daf9-7pk21",
    "2e2md-b4070", "2ftdw-8cr18", "2gr2g-h8s29", "2kh6a-57d12", "2sa0a-49w88", "2y89r-m1q15", "32ra9-dhv25",
    "36x8p-hrv47", "3a62g-a7r35", "3aemm-p8g93", "3nqcz-g3h81", "3pcj9-df236", "3snez-rpq88", "3ydnb-ckz62",
    "3yf97-hzb15", "3zt0r-ewf78", "460v7-scx59", "4mdsp-xpm53", "4nfmg-88z80", "4t64a-r3d56", "4tgt6-1p378",
    "4vt5h-sjd96", "4xb37-dcy27", "57kc3-fhs52", "58a61-7te73", "58hhx-t4q48", "5dbg6-vj639", "5jsfj-cdg70",
    "5v0ea-za044", "5vh19-61y09", "643dg-51806", "645qm-6ss59", "6540w-gpb92", "6a0y1-57t73", "6eedj-3ey43",
    "6f23w-zqe90", "6ntbp-vbx54", "6t3hd-9v682", "6t5yd-kek31", "71861-xfe58", "7cqcz-ev630", "7ggkm-e8132",
    "7p3z3-t3q91", "7q0a5-wvr60", "7s8zb-zbc69", "7v6p9-92t31", "7v74k-sbq73", "813jn-zdg11", "88sbv-m7172",
    "8a0qm-ae178", "8dhnc-d7064", "8fdar-jpw03", "8fj15-e3949", "8h7y3-yrj49", "8q9cy-0e333", "8qvdy-t3662",
    "8z1k0-m0p53", "919tg-egj89", "935a7-wb166", "9b7fy-9zd08", "9hydd-z0v77", "9j1pn-77r98", "9qf6r-9fq18",
    "a68hf-tt080", "a8r4c-whw21", "a9a17-tdt45", "a9xjd-s5s43", "abrrf-1qs35", "ac701-y1a73", "achjy-9mq04",
    "ae9tf-vq616", "as20t-exy47", "as73a-ja910", "at71v-gdh04", "b0733-0bz44", "b0ved-e1358", "b69b2-wd449",
    "b9k22-a4856", "bc27j-y6f82", "bgtn9-gd151", "bj15t-p3415", "bmrfs-z1892", "bnxsn-7w947", "c3px7-y0k21",
    "c40dt-dg536", "c9ksd-mbb05", "c9peh-kt938", "caz77-rka20", "cdh59-rfp34", "cmh66-t8h09", "cnc75-96z93",
    "cpypm-rar80", "cr6yv-kmm28", "cw8c1-cr538", "czba5-18919", "d2hm0-qsk96", "d6pmr-tty23", "dby42-mrn61",
    "dca0t-q7m31", "de3ds-jdd49", "desfh-3ma25", "detxn-0ga89", "dfnmr-0r807", "dgn52-bfq64", "dn3b0-h9470",
    "dr2cc-f2047", "drzx4-bh631", "dxr5p-qjh86", "ejv7m-q7928", "f20vs-e0v74", "f4cjq-y4987", "fb785-kq109",
    "fmrcj-08z10", "frvyv-s0e55", "fwtgd-ahr14", "g5rqp-2aw69", "g8jhr-t8x36", "gnbx5-ahw69", "granj-nhv94",
    "grpwm-0mn79", "gwem9-1z940", "gwxqt-erf27", "gyg6f-c0m39", "h1y3j-qnc51", "hacw9-yzc96", "hbpvk-pfp22",
    "hdb97-me114", "hh8re-ram66", "hhx76-4a941", "hmdsb-k3h43", "hp176-eq805", "hyjhn-8em96", "j2fka-sda13",
    "j2myj-5zd34", "j9ss5-v8y68", "jd7mr-4x249", "jd7xq-3b856", "jg7n7-6zf07", "jh6ff-x5518", "jj64f-dqy33",
    "jsrvx-8yp49", "jvr52-4ff25", "jwt9m-dtt37", "k1hjd-y6e90", "k45dq-ebf35", "k6ak4-4a882", "k79n4-gaj04",
    "k8aqa-34m62", "kacdj-ssv28", "kbsns-73a82", "kc275-mc539", "kd6ct-85z54", "kf8d0-27289", "kf9nw-eqr51",
    "kg706-sbx37", "kp5c5-40408", "kpnmb-1gh68", "kqtmm-08x21", "m3scp-g5b30", "m78xf-q6561", "m7d8t-ktv16",
    "m9gfa-xr118", "mazj9-rbz26", "mfryb-9xc43", "my4yn-3nj21", "n271h-74987", "nb3a2-hx706", "nc4x8-yme87",
    "nepw0-kn826", "ng460-sjb05", "njdbv-4yr45", "nk9kn-w0b37", "nqd1f-q1z28", "nr4an-byf21", "nsp7v-2dk91",
    "ntjyq-kmp74", "nv386-6m953", "p76zk-aax05", "p939g-t5y42", "pbd17-m0b04", "pfng9-t9j62", "pg1n3-b0061",
    "pke3c-65m49", "pkn2d-h8190", "pp60c-spt74", "pp9m5-dfr36", "ptryq-skn46", "q2smg-en715", "qbdsr-k2m78",
    "qc3y4-t0292", "qc8ra-r2a80", "qffdm-h9w98", "qs0kz-6t805", "qwg2b-ncm42", "qyk4y-c7h83", "r0ysk-tx858",
    "r4n3a-sa170", "rdm6f-8qp56", "rgpf5-1wh29", "rj828-h9687", "rs6a8-47972", "rs736-v1c41", "s0dhd-dph38",
    "s3nkw-qv745", "s74c9-2bv33", "s7ww9-v2664", "s99ps-4cb80", "sa8qw-v7j53", "sbvxf-73e81", "sh9jq-9zz50",
    "skbnr-9wx61", "skdzy-kyx54", "snw8t-ke894", "spga5-j9p32", "sppr5-92n09", "sqrdx-ebj42", "t5nvc-rah21",
    "t87ct-v1804", "t8qks-9s556", "tab6j-y9a89", "tbphj-bsk15", "tcrks-ymp88", "teqne-zmd89", "tvpaj-16g70",
    "txxzk-5mp05", "v2xaa-5pa22", "v5ahk-pk833", "vb1sd-k1088", "vf417-5pb37", "vkzwd-0bm25", "vmvs5-0tv03",
    "vpyk8-xaa86", "vs5fd-dbv98", "vsv00-8vy33", "vv6sy-98m49", "vzzfg-68n98", "w02kr-n5b13", "w6yzf-72989",
    "w7hen-cme26", "wbyh1-0p659", "wnqcv-06d96", "wpz35-q2y88", "wwrkx-77334", "x1n3f-vwz18", "x4str-apb06",
    "x5jkz-5g721", "x65fq-dhw66", "xahpx-dkk66", "xfsv2-2fq31", "xhzqk-b7f96", "xrtkn-tzm42", "xv1qx-85189",
    "y0fdd-7bq66", "y6ca2-5bn20", "y84fb-t4h71", "y90bb-bpd55", "yaqw0-vww55", "yayv3-eha98", "ykwg5-vb523",
    "yna62-s7749", "ynf18-k0x34", "ywe3w-nwg25", "yxefy-5b695", "yzhmw-vje16", "z3w68-7sy20", "zhab2-1tp79",
    "zne6q-a4h48", "zqxj1-45w58", "zrd92-wh846", "ztd8f-vj464", "zw0nd-wx180",
]
for _mc in _MCLOUD_RESCAN:
    DATASETS[f"mcloudarchive__{_mc}"] = ("archive", f"mcloudarchive__{_mc}")


def get_reader(dataset):
    if dataset not in DATASETS:
        raise SystemExit(f"unknown --dataset {dataset!r}; known: {list(DATASETS)}")
    mod_name, dataset_id = DATASETS[dataset]
    mod = importlib.import_module(f"inference.nc_ingest.readers.{mod_name}")
    return mod.read, dataset_id


def convert(dataset, inputs, out, *, chunk_rows=20000, strict=False):
    if os.path.exists(out):
        print(f"[convert] output exists, skip: {out}")
        return -1
    read, dataset_id = get_reader(dataset)
    w = ShardWriter(out, dataset_id, chunk_rows=chunk_rows)
    n = 0
    for src in inputs:
        for rec in read(src):
            if strict:
                validate_record(rec)
            ids = mint_ids(rec["numbers"], rec["positions"], rec["cell"], rec["pbc"],
                           dataset=dataset_id, src_file=rec["src_file"],
                           frame_idx=rec["frame_idx"], energy=rec["energy"],
                           source_id=rec["source_id"])
            names = [f"{dataset_id}__{rec['src_file']}__frame{rec['frame_idx']}"]
            if rec["source_id"]:
                names.append(rec["source_id"])
            names.extend(rec.get("extra_names", []))
            w.add(rec, ids, names)
            n += 1
            if n % 100000 == 0:
                print(f"[convert] {n:,} rows ...", flush=True)
    written = w.close()
    print(f"[convert] done: {written:,} rows -> {out}")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--inputs", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk-rows", type=int, default=20000)
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    convert(args.dataset, args.inputs, args.out,
            chunk_rows=args.chunk_rows, strict=args.strict)


if __name__ == "__main__":
    main()
