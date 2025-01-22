import sys
from Bio.PDB import PDBParser, NeighborSearch, PDBIO, Structure, Model, Chain

def extract_pocket(pdb_file, center, threshold):
    parser = PDBParser()
    structure = parser.get_structure('protein', pdb_file)
    atoms = [atom for atom in structure.get_atoms()]
    
    ns = NeighborSearch(atoms)
    nearby_atoms = ns.search(center, threshold)
    
    pocket_residues = set()
    for atom in nearby_atoms:
        pocket_residues.add(atom.get_parent())
    
    return pocket_residues

def save_pocket_to_pdb(pocket_residues, output_file):
    pocket_structure = Structure.Structure("Pocket")
    pocket_model = Model.Model(0)
    
    chains = {}
    for residue in pocket_residues:
        chain_id = residue.get_parent().id
        if chain_id not in chains:
            chains[chain_id] = Chain.Chain(chain_id)
        chains[chain_id].add(residue.copy())
    
    for chain_id, chain in chains.items():
        sorted_residues = sorted(chain.get_list(), key=lambda res: res.id[1])
        new_chain = Chain.Chain(chain_id)
        for residue in sorted_residues:
            new_chain.add(residue)
        pocket_model.add(new_chain)
    
    pocket_structure.add(pocket_model)
    
    io = PDBIO()
    io.set_structure(pocket_structure)
    io.save(output_file)

def main():
    pdb_file = sys.argv[1]
    center = tuple(map(float, sys.argv[2:5]))
    threshold = float(sys.argv[5])
    output_file = sys.argv[6]
    
    pocket_residues = extract_pocket(pdb_file, center, threshold)
    
    save_pocket_to_pdb(pocket_residues, output_file)

if __name__ == "__main__":
    main()
