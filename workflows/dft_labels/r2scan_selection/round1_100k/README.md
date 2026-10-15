# BoR — "Best of the Rest"

The 100k non-MAD structures picked out of the coreset for DFT recalculation, plus the check
of whether any failed flashmd frames were skipped in the process.

MAD-2.0 is composed as `MAD-1.5 + BoR + MAD-NEXT`. BoR is the non-MAD part.

## How BoR was created — `01_create_bor.py`

Walk the coreset in `fps_order` and keep the first 100,000 structures that are

1. **not in the MAD anchor** (`mad_union/anchor.parquet`), and
2. **under 300 atoms** (`n_sites < 300`).

```python
is_mad = pc.is_in(configuration_id, value_set=mad_configuration_ids)
qualifying = (~is_mad) & (n_sites < 300)
selected = qualifying[argsort(fps_order)][:100_000]
```

Result (`madcore_10M/recalc_100k/first100k_nonMAD_lt300atoms.{xyz,parquet}`):

| | |
|---|---|
| structures | 100,000 |
| `fps_rank` range | 1 … **122,378** |
| n_atoms | 1 … 256 |
| MAD structures | 0 |
| top families | Alexandria 42,302 · JARVIS_QE_TB 20,234 · MP-ALOE 12,825 · OMat24 8,054 |

The shared file `cscs-madcore_first100k_non-mad_max300.xyz` is byte-identical to this output.

**122,378 ranks were consumed to yield 100,000 structures**, so **22,378 were skipped** —
each either MAD or ≥300 atoms. That gap is what the check below is about.

### Which coreset

BoR was built from `madcore_10M/out/coreset_10M_full.parquet`, **not** from the production
`coreset224` parquet directly. `00_verify_coreset10M_prefix.py` confirms the two orderings
are identical over the relevant range, so `fps_order` means the same thing in both:

```
first 200,000 entries identical in order : 200,000 (100.0000%)
over the BoR walk range (0..122,378)     : 122,379 / 122,379 identical
```

## The failed-flashmd check — `02_` and `03_`

**The question.** The `flashmd_3d` structures were recomputed and 5,230 of 156,208 did
not converge. FPS suppresses candidates near whatever it has already selected,
so a flashmd structure sitting in the walked range **held an FPS slot and was then dropped
from BoR** by the `~is_mad` filter. If it now leaves MAD-2.0, no BoR structure covers that
region — a coverage hole.

**The obstacle.** `flashmd_failed_frames.xyz` is keyed by `frame_id` + `subset`; the coreset
is keyed by `configuration_id`. There is no mapping anywhere (`mad_next`'s
`component_configuration_ids` maps gdb13 *dimer* components, not flashmd frames).

**The workaround.** Match on structure content — an exact hash of
`(atomic_numbers, positions, cell)` with positions rounded to 0.001 Å. This matched
**5,230 / 5,230 (100%)**, so the geometries are identical between the two sources and the
correspondence is unambiguous.

### Result

```
5,230 failed frames          ->  100% matched to mad_next configuration_ids
  4,063  are in coreset_10M     (fps_order 1,052 … 9,988,503)
    723  are in the BoR walk    (fps_order <= 122,378)   <-- these cost a BoR slot
    723  of those are <300 atoms -> every one would have qualified
      0  were >=300 atoms        -> none excluded by the size filter anyway
```

They start early (first at `fps_order` 1,052) and are small: **482 are 3-atom, 179 are
2-atom**. Small molecules sit in sparse corners of feature space, which is why FPS took
them so early — and why losing their only representative would leave a real hole rather
than one a neighbour covers.

**`failed_frame_ids_BOR_detail.tsv`** — the 723, as `fps_order`, `n_atoms`,
`configuration_id`, `frame_id`, sorted by rank. This is the list handed back for the recalculation.

**Resolution.** The plan is to keep the failed structures without stress. That closes
the issue entirely: nothing leaves MAD-2.0, so no region loses its representative and the
723 distinction stops mattering. The list is only needed if a subset is kept.

## Files

| file | what |
|---|---|
| `00_verify_coreset10M_prefix.py` | confirms coreset_10M is the prefix of coreset224 |
| `01_create_bor.py` | builds BoR (copy of `madcore_10M/extract_recalc_100k.py`, which is where it was run) |
| `02_check_failed_in_bor.py` | hashes failed frames -> configuration_ids -> fps_order; writes `failed_in_coreset_cids.txt` |
| `03_map_failed_to_frame_id.py` | maps those back to `frame_id`; writes the detail TSV |
| `failed_frame_ids_BOR_detail.tsv` | **the 723** |
| `submit_*.sh` | slurm wrappers |

Run order: `02` then `03` (03 reads 02's output). The bulky inputs/outputs
(`.xyz`, `.parquet`, `.npy`) stay in `madcore_10M/` and are gitignored.

## Restore from the release + store copy (2026-08-21)

`coreset_10M_full.parquet` no longer exists, so round 1 was rebuilt from the frozen
`release/mad-core` and checked against the file that was sent to the collaborators
(archived here as `out/as_sent/cscs-madcore_first100k_non-mad_max300.xyz`).

- `08_restore_round1_from_release.py` — the restore. Asserts the walk reproduces the
  original index parquet rank-for-rank (ranks 1…122,378 and n_atoms identical), then
  writes the xyz + a parquet index carrying **both** id eras.
- `09_verify_restored_vs_shared.py` — frame-by-frame comparison with the as-sent file:
  ranks, species and pbc identical in all 100,000 frames; **99,995 identical
  atom-for-atom** (max |Δposition| = max |Δcell| = 0); writes `out/round1_id_map.tsv`.
- `10_check_folded_five.py` — the 5 differing frames are the nomad far-from-origin
  rows (60,919 · 63,706 · 96,880 · 100,156 · 100,985), where the as-sent file holds
  unwrapped coordinates at |x| ≈ 1e10 Å and the release holds them folded.
  Reconstructing `x + image_flags @ cell` matches the as-sent file to 2.8–4.8e-8
  relative — below the float32 ULP (512–1024 Å) at that magnitude, i.e. as close as
  the as-sent file's own float32 coordinates can represent. Same structures.

Data + scripts copied to
`${STORE_ROOT}/mad-core/botr/round1_100k/` (with `SHA256SUMS`).
