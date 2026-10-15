import json

from collections import Counter

import numpy as np
import pyarrow.parquet as pq


run_dir = "${DEV_ROOT}/phace-atomic/accrete/runs/madcore_production"

family_category = {
    "Alexandria": "3D bulk", "OMat24": "3D bulk", "Materials_Project": "3D bulk", "MP-ALOE": "3D bulk",
    "MatPES-2025.1": "3D bulk", "UNEP_v1_2023": "3D bulk",
    "JARVIS_QE_TB": "3D bulk", "JARVIS_CFID_OQMD": "3D bulk", "JARVIS_OQMD_no_CFID": "3D bulk",
    "JARVIS_ALIGNN_FF": "3D bulk", "JARVIS_SNUMAT": "3D bulk",
    "WBe_PRB2019": "3D bulk", "Mg_edmonds_2022": "3D bulk", "Zr_Sn_JNM_2024": "3D bulk",
    "MoNbTaVW_PRB2021": "3D bulk", "AlNiTi_CMS_2019": "3D bulk", "TiO2_CMS2016": "3D bulk",
    "Carolina_Materials": "3D bulk", "Cu_FHI-aims_NPJCM_2021": "3D bulk", "Si_Al_Ti_Seko_PRB_2019": "3D bulk",
    "Mo_PRM2019": "3D bulk", "V_PRM2019": "3D bulk", "NNIP_FeH_PRM_2021": "3D bulk",
    "pure_magnesium_DFT_PRM2020": "3D bulk", "H_nature_2022": "3D bulk",
    "AENET_amorphous_LiSi_JCP2021": "3D bulk", "Carbon_GAP_JCP_2020": "3D bulk",
    "cathode_materials_for_rechargeable_Ca_batteries_CM2021": "3D bulk",
    "OMol25": "molecules", "ANI-1": "molecules", "ANI-2x": "molecules", "gdb13": "molecules",
    "QM-22": "molecules", "Vector-QM24": "molecules",
    "OC20-IS2RES": "surfaces/2D", "OC20_S2EF_train_all": "surfaces/2D", "OC22-S2EF": "surfaces/2D",
    "OC22-IS2RE": "surfaces/2D", "ODAC": "surfaces/2D", "oc20neb": "surfaces/2D", "OC25": "surfaces/2D",
    "CGM-MLP_natcomm2023": "surfaces/2D",
    "Open_Molecular_Crystals_2025_OMC25": "molecular crystals",
}
anchor_category = json.load(open("${STORE_ROOT}/mad-core/mad_union/category_map.json"))

p = pq.read_table(f"{run_dir}/picks_properties.parquet", columns=["configuration_id", "family", "periodicity", "n_sites", "elements"])
cids = p.column("configuration_id").to_pylist()
fams = p.column("family").to_pylist()
nper = p.column("periodicity").to_numpy()
nsit = p.column("n_sites").to_numpy()
elems = p.column("elements").to_pylist()


NONMETAL_Z = {1, 2, 5, 6, 7, 8, 9, 10, 14, 15, 16, 17, 18, 32, 33, 34, 35, 36, 51, 52, 53, 54, 85, 86}


def dataset_category(cid, fam):
    if cid in anchor_category:
        return anchor_category[cid]
    return family_category.get(fam, "3D bulk")


cid_to_category = {}
by_pbc = by_dataset = 0
for cid, fam, n, ns, els in zip(cids, fams, nper, nsit, elems):
    if ns == 1 and n == 0:
        cid_to_category[cid] = "atom"; by_pbc += 1
    elif n == 0:
        has_metal = any(int(z) not in NONMETAL_Z for z in els)
        cid_to_category[cid] = "clusters" if has_metal else "molecules"; by_pbc += 1
    elif n in (1, 2):
        cid_to_category[cid] = "surfaces/2D"; by_pbc += 1
    else:
        cid_to_category[cid] = dataset_category(cid, fam); by_dataset += 1

json.dump(cid_to_category, open(f"{run_dir}/picks_category_map.json", "w"))
colors = json.load(open("${STORE_ROOT}/mad-core/mad_union/category_colors.json"))
colors["atom"] = "#d62728"
colors["clusters"] = "#17becf"
json.dump(colors, open(f"{run_dir}/picks_category_colors.json", "w"), indent=2)

print("category composition:", dict(Counter(cid_to_category.values())))
print(f"assigned by pbc (0/1/2 dims): {by_pbc:,}   by dataset (3-periodic): {by_dataset:,}")
