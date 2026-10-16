import argparse
import multiprocessing
import os
import sys
import time
from rdkit import Chem
from rdkit.Chem import AllChem

def process_smiles_to_exyz(line_data):
    if not line_data:
        return None

    # handle SMILES or SMILES ID input format
    parts = line_data.strip().split()
    if not parts:
        return None
    
    smiles = parts[0]
    # if there is an ID, you might want to save it too
    
    try:
        # 1. RDKit Geometry Generation
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None

        mol = Chem.AddHs(mol)

        # ETKDGv3 Embedding (High quality conformers)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xf00d
        res = AllChem.EmbedMolecule(mol, params)
        
        # fallback to random coordinates if embedding fails
        if res == -1:
            res = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if res == -1: return None 

        # force Field Optimization
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except:
            pass 

        # 2. Extended XYZ Formatting
        xyz_block = Chem.MolToXYZBlock(mol)
        lines = xyz_block.split('\n')
        
        # lines 0 is the atom count (N)
        
        # the 'Properties' flag defines the columns
        header = f"Properties=species:S:1:pos:R:3 entry_id={smiles}"
        
        lines[1] = header
        
        # rejoin and ensure distinct newline at the end
        return "\n".join(lines).strip() + "\n"

    except Exception:
        return None

def main():
    # argument Parsing
    parser = argparse.ArgumentParser(
        description="Convert SMILES to Extended XYZ format (compatible with ASE/quippy)."
    )
    parser.add_argument("input_file", help="Path to input SMILES file.")
    parser.add_argument("-o", "--output", help="Output filename. Defaults to input_name.xyz", default=None)
    parser.add_argument("-j", "--jobs", type=int, default=max(1, multiprocessing.cpu_count() - 1), 
                        help="Number of cores to use.")

    args = parser.parse_args()

    # setup Output Filename
    if args.output is None:
        base = os.path.splitext(args.input_file)[0]
        output_file = f"{base}.xyz"
    else:
        output_file = args.output

    if not os.path.isfile(args.input_file):
        print(f"Error: File {args.input_file} not found.")
        sys.exit(1)

    # processing
    print(f"--- Processing GDB11 to Extended XYZ ---")
    print(f"Input: {args.input_file}")
    print(f"Output: {output_file}")
    print(f"Workers: {args.jobs}")

    start = time.time()
    count = 0
    success = 0

    def line_reader(path):
        with open(path, 'r') as f:
            for line in f:
                yield line

    with open(output_file, 'w') as out:
        with multiprocessing.Pool(processes=args.jobs) as pool:
            # imap with chunksize reduces IPC overhead significantly
            for res in pool.imap(process_smiles_to_exyz, line_reader(args.input_file), chunksize=500):
                count += 1
                if res:
                    out.write(res)
                    success += 1
                
                if count % 10000 == 0:
                    elapsed = time.time() - start
                    print(f"\rProcessed: {count:,} | Success: {success:,} | {count/elapsed:.0f} mol/s", end="")

    print(f"\n\n--- Complete ---")
    print(f"Total Lines: {count:,}")
    print(f"Structures Written: {success:,}")
    print(f"File saved to: {output_file}")

if __name__ == "__main__":
    main()
