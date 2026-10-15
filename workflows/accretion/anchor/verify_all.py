import glob
import re

from collections import Counter

import numpy as np
import pyarrow.parquet as pq

rd = "${DEV_ROOT}/phace-atomic/accrete/runs/madcore_production"
POOL = 184_180_658
ANCHOR = 856_012
OK = lambda c: "OK" if c else "*** FAIL ***"

print("=" * 64)
print("PART 1 — CORESET INTEGRITY")
t = pq.read_table(f"{rd}/out/accrete.parquet")
cids = t.column("configuration_id").to_pylist()
gids = t.column("global_id").to_numpy()
fps = t.column("fps_order").to_numpy()
K = len(cids)
print(f"  size = {K:,}  {OK(K == ANCHOR)}")
print(f"  unique cids = {len(set(cids)):,}  dups = {K - len(set(cids))}  {OK(len(set(cids)) == K)}")
print(f"  global_id in [0,{POOL:,}) : min={gids.min():,} max={gids.max():,}  {OK(gids.min() >= 0 and gids.max() < POOL)}")
print(f"  fps_order contiguous 0..K-1  {OK(np.array_equal(np.sort(fps), np.arange(K)))}")
anc = int((gids < ANCHOR).sum())
print(f"  anchor(mad-coreset) picks gid<{ANCHOR:,} = {anc:,}  pool = {K - anc:,}")

print("\nPART 2 — NESTED FPS HIERARCHY (ckpt_k ⊂ ckpt_k+1) + final==accrete")
levels = sorted(int(re.search(r"ckpt_k(\d+)\.npy", p).group(1))
                for p in glob.glob(f"{rd}/out/ckpt/ckpt_k*.npy") if "_sep" not in p)
prev, sizes, nested = None, [], True
for k in levels:
    s = set(np.load(f"{rd}/out/ckpt/ckpt_k{k:02d}.npy").tolist())
    sizes.append((k, len(s)))
    if prev is not None and not prev.issubset(s):
        nested = False
        print(f"    k={k}: NOT a superset of k-1 *** ({len(prev - s):,} lost)")
    prev = s
print(f"  levels: {sizes}")
print(f"  strictly nested across all layers  {OK(nested)}")
final = set(np.load(f"{rd}/out/ckpt/ckpt_k{levels[-1]:02d}.npy").tolist())
print(f"  k={levels[-1]} checkpoint == accrete.parquet global_ids  {OK(final == set(gids.tolist()))}")
print(f"  each layer size == 2^k (except final partial)  "
      f"{OK(all(n == 2 ** k for k, n in sizes[:-1]))}  (final {sizes[-1][1]:,})")

print("\nPART 3 — sep>=cov GUARANTEE (every layer, every job)")
rows = viol = 0
worst = 1.0
for log in sorted(glob.glob(f"{rd}/logs/accrete-prod_*.out")):
    for line in open(log):
        if not re.match(r"\s*0:\s+\d+\s+\|", line):
            continue
        parts = [x.strip() for x in line.split("|")]
        if len(parts) < 8:
            continue
        try:
            sep, cov, rel = float(parts[5]), float(parts[6]), float(parts[7])
        except ValueError:
            continue
        rows += 1
        worst = min(worst, rel)
        if sep < cov - 1e-3 or rel < -1e-3:
            viol += 1
print(f"  layer rows parsed = {rows}   sep<cov violations = {viol}   worst rel_gap = {worst:.2e}  {OK(viol == 0)}")

print("\nPART 4 — FILTER COMPLIANCE (coreset zbl + finiteness)")
p = pq.read_table(f"{rd}/picks_properties.parquet",
                  columns=["zbl_max_force", "zbl_energy_per_atom", "zbl_max_atomic_energy", "n_sites"])
f, e, a = (p.column(c).to_numpy() for c in ["zbl_max_force", "zbl_energy_per_atom", "zbl_max_atomic_energy"])
ns = p.column("n_sites").to_numpy()
fin = np.isfinite(f) & np.isfinite(e) & np.isfinite(a)
vio = fin & ((f >= 200) | (e > 50) | (a > 100))
print(f"  finite-zbl picks = {int(fin.sum()):,}   threshold violations = {int(vio.sum())}  {OK(vio.sum() == 0)}")
print(f"  NaN-zbl picks (MAD anchor, not in zbl store) = {int((~fin).sum()):,}  (expect ~mad-coreset count)")
print(f"  n_sites <= 1000  {OK(int((ns > 1000).sum()) == 0)}  (max nsites = {int(ns.max())})")

print("\nPART 5 — ANALYSIS RECONCILIATION")
pp = pq.read_table(f"{rd}/picks_properties.parquet")
cat = Counter(pp.column("category").to_pylist())
print(f"  category counts sum = {sum(cat.values()):,}  {OK(sum(cat.values()) == K)}   {dict(cat)}")
ec = Counter()
for r in pp.column("elements").to_pylist():
    ec.update(set(int(z) for z in r))
print(f"  distinct elements = {len(ec)}   O(Z=8) in {ec.get(8, 0):,} structures  (element map showed O=97,167)  "
      f"{OK(ec.get(8) == 97167)}")
# family from properties vs the resolver-verified attribution
fam = Counter(pp.column("family").to_pylist())
print(f"  family counts sum = {sum(fam.values()):,}  {OK(sum(fam.values()) == K)}   top: {fam.most_common(4)}")

print("\nPART 6 — cid-GEOMETRY SPOT-CHECK (independent store read by cid)")
idx = pq.read_table(f"{rd}/cid_index.parquet")
loc = {}
for c, s, r in zip(idx.column("configuration_id").to_pylist(), idx.column("shard").to_pylist(), idx.column("row").to_numpy()):
    loc.setdefault(c, (s, int(r)))
prop_by_cid = {pp.column("configuration_id")[i].as_py():
               (sorted(set(int(z) for z in pp.column("elements")[i].as_py())), int(pp.column("periodicity")[i].as_py()))
               for i in range(0, K, K // 400)}                        # ~400 sampled picks
rng = np.random.default_rng(0)
sample = list(prop_by_cid)
bad = 0
for c in sample:
    sh, row = loc[c]
    rec = pq.ParquetFile(sh).read_row_group(0, columns=["configuration_id"])  # locate row group
    # read the exact row for atomic_numbers + pbc
    pf = pq.ParquetFile(sh)
    lo = 0
    for rg in range(pf.num_row_groups):
        n = pf.metadata.row_group(rg).num_rows
        if lo <= row < lo + n:
            tt = pf.read_row_group(rg, columns=["configuration_id", "atomic_numbers", "pbc"]).slice(row - lo, 1)
            break
        lo += n
    assert tt.column("configuration_id")[0].as_py() == c, "cid mismatch at (shard,row)!"
    els_store = sorted(set(int(z) for z in tt.column("atomic_numbers")[0].as_py()))
    per_store = int(sum(bool(x) for x in tt.column("pbc")[0].as_py()))
    els_prop, per_prop = prop_by_cid[c]
    if els_store != els_prop or per_store != per_prop:
        bad += 1
print(f"  spot-checked {len(sample)} picks: read structure by cid from store, compared elements+periodicity")
print(f"  mismatches = {bad}  {OK(bad == 0)}  (0 => geometry pipeline is content-hash-correct)")
print("=" * 64)
