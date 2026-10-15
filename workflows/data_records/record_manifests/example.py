import h5py
import numpy as np
from ase import Atoms
from ase.io import write

data = h5py.File("madcosmo_cells_0-2pow18_data.h5")
features = h5py.File("madcosmo_cells_0-2pow18_features.h5")

# let's take structure 1 in the MAD-CORE ordering
madcore_row = 1

# here we retrieve MAD-COSMO structures associated with it
first = data["index/cell_offsets"][madcore_row]
last = data["index/cell_offsets"][madcore_row + 1]
print("MAD-CORE row", madcore_row, "is", data["index/center_configuration_id"][madcore_row].decode())
print("the structures assigned to it are rows", first, "to", last - 1)

# let's take the first MAD-COSMO structure in the cell
row = first
a = data["geometry/atom_offsets"][row]
b = data["geometry/atom_offsets"][row + 1]
print("numbers", data["geometry/numbers"][a:b])
print("positions", data["geometry/positions"][a:b])
print("cell", data["geometry/cell"][row], "pbc", data["geometry/pbc"][row])
print("ecumetric predicted forces", data["predictions/ecumetric_nc_forces"][a:b])
print("ecumetric predicted  energy", data["predictions/ecumetric_energy"][row])

# the dataset of the row, a position in the name list dictionaries/dataset_id
code = data["identity/dataset_id_code"][row]
print("dataset", data["dictionaries/dataset_id"][code].decode())
print("fps rank within the cell", data["ordering/fps_order_in_cell"][row])
print("filter code", data["diagnostics/filter_code"][row], "duplicate", data["diagnostics/is_duplicate"][row])

configuration_id = data["identity/configuration_id"][row].decode()
print("configuration id", configuration_id)

# the same structure as an ase Atoms object
atoms = Atoms(numbers=data["geometry/numbers"][a:b],
              positions=data["geometry/positions"][a:b],
              cell=data["geometry/cell"][row],
              pbc=data["geometry/pbc"][row])
atoms.info["configuration_id"] = configuration_id
atoms.info["ecumetric_energy"] = data["predictions/ecumetric_energy"][row]
atoms.arrays["ecumetric_nc_forces"] = data["predictions/ecumetric_nc_forces"][a:b]
write("structure.extxyz", atoms)

# here we read the features of the structure and scaled to the metric used for the selection
z = (features["ecumetric_features"][row] - features["scaler/mean"][:]) / features["scaler/scale"][()]
print("features", z.shape, z[:4])

# how many structures of the cell passed the filters, filter code 0 and not a duplicate
code = data["diagnostics/filter_code"][first:last]
duplicate = data["diagnostics/is_duplicate"][first:last]
keep = (code == 0) & (duplicate == False)
print(keep.sum(), "of", last - first, "structures passed")

# all structures of one dataset
names = [name.decode() for name in data["dictionaries/dataset_id"][:]]
code = names.index("ODAC_train")
codes = data["identity/dataset_id_code"][:]
rows = np.where(codes == code)[0]
print(len(rows), "structures from ODAC_train in this file, first rows", rows[:5])
print("energies of all of them", data["predictions/ecumetric_energy"][rows])

# how many structures each dataset has in this file
counts = np.bincount(codes, minlength=len(names))
for code in np.argsort(counts)[::-1][:5]:
    print(names[code], counts[code])
