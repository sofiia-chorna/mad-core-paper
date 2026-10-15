import numpy as np
import pyarrow.parquet as pq

project_dir = "${DEV_ROOT}/phace-atomic"
coreset_10M_path = f"{project_dir}/accrete/runs/madcore_10M/out/coreset_10M_full.parquet"
coreset_224_path = ("${STORE_ROOT}/mad-core/accrete/phace_atomic_ep92"
                    "/coreset224/out/coreset_224_full.parquet")
bor_walk_limit = 122_378
n_check = 200_000


def ordered_configuration_ids(path):
    table = pq.read_table(path, columns=["fps_order", "configuration_id"])
    order = np.argsort(table.column("fps_order").to_numpy(), kind="stable")
    return np.asarray(table.column("configuration_id").to_pylist(), dtype=object)[order]


coreset_10M_ids = ordered_configuration_ids(coreset_10M_path)
print(f"coreset_10M: {len(coreset_10M_ids):,} rows")

coreset_224_ids = ordered_configuration_ids(coreset_224_path)
print(f"coreset224 : {len(coreset_224_ids):,} rows")

n_compared = min(n_check, len(coreset_10M_ids))
n_identical = int((coreset_10M_ids[:n_compared] == coreset_224_ids[:n_compared]).sum())

print(f"\nfirst {n_compared:,} entries identical in order: {n_identical:,} "
      f"({100 * n_identical / n_compared:.4f}%)")

if n_identical == n_compared:
    print("  coreset_10M is the prefix of coreset224 over this range; "
          f"the fps_order <= {bor_walk_limit:,} cut is valid")
else:
    first_difference = int(np.argmax(
        coreset_10M_ids[:n_compared] != coreset_224_ids[:n_compared]))
    verdict = "invalid" if first_difference <= bor_walk_limit else "still valid"

    print(f"  orderings diverge at fps_order {first_difference:,}")
    print(f"      coreset_10M: {coreset_10M_ids[first_difference]}")
    print(f"      coreset224 : {coreset_224_ids[first_difference]}")
    print(f"      the BoR selection result is {verdict}")

n_walk = bor_walk_limit + 1
n_walk_identical = int((coreset_10M_ids[:n_walk] == coreset_224_ids[:n_walk]).sum())

print(f"\nover the BoR walk range (0..{bor_walk_limit:,}): "
      f"{n_walk_identical:,} / {n_walk:,} identical")
