# attach.py
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
import numpy as np
from TM_catalyst_framework.GeoOpt import get_conf_positions, set_conf_positions, rotation_matrix_align_vectors

def attach_ligand_to_site(metal_mol, metal_idx, ligand_rdkit_mol, donor_atom_idx_in_ligand, donor_site_coord, bond_type=Chem.BondType.DATIVE, align=True):
    """
    Align ligand so donor atom sits at donor_site_coord, then merge into metal_mol and add M-L bond.
    Returns merged RDKit Mol.
    """
    # Ensure conformers exist
    if ligand_rdkit_mol.GetNumConformers() == 0:
        raise ValueError("Ligand must have 3D conformer.")

    if metal_mol.GetNumConformers() == 0:
        raise ValueError("Metal mol must have conformer with metal at origin.")

    # 1) extract coords
    lig_coords = get_conf_positions(ligand_rdkit_mol)
    donor_pos = lig_coords[donor_atom_idx_in_ligand]

    # 2) Optional alignment: rotate so donor->some neighbor points toward metal
    if align:
        # choose a neighbor of donor to define ligand direction (if exists)
        donor_atom = ligand_rdkit_mol.GetAtomWithIdx(donor_atom_idx_in_ligand)
        nbrs = [n.GetIdx() for n in donor_atom.GetNeighbors() if n.GetIdx() != donor_atom_idx_in_ligand]
        if len(nbrs) > 0:
            neighbor_idx = nbrs[0]
            vec_from = lig_coords[neighbor_idx] - donor_pos
            # desired vector: point from donor site toward metal center (origin)
            desired_vec = np.array([0.0, 0.0, 0.0]) - np.array(donor_site_coord)
            R = rotation_matrix_align_vectors(vec_from, desired_vec)
            # rotate all ligand coords around donor_pos
            lig_coords = (R @ (lig_coords - donor_pos).T).T + donor_pos

    # 3) translate ligand so donor sits at donor_site_coord
    translation = np.array(donor_site_coord) - lig_coords[donor_atom_idx_in_ligand]
    lig_coords = lig_coords + translation

    # update ligand conformer positions
    set_conf_positions(ligand_rdkit_mol, lig_coords)

    # 4) merge molecules
    combined = Chem.CombineMols(metal_mol, ligand_rdkit_mol)
    editable = Chem.EditableMol(combined)

    offset = metal_mol.GetNumAtoms()
    new_donor_idx = donor_atom_idx_in_ligand + offset

    # Add bond between metal_idx and new_donor_idx
    editable.AddBond(metal_idx, new_donor_idx, bond_type)
    merged = editable.GetMol()

    # (Optional) copy conformers — RDKit CombineMols copies conformers already
    return merged

from rdkit import Chem

def replace_atom_with_fragment(mol: Chem.Mol, atom_idx: int, fragment_smiles: str) -> Chem.Mol:
    """
    Replace an atom at atom_idx with a new fragment defined by fragment_smiles.

    Returns a new RDKit Mol object.
    """
    fragment = Chem.MolFromSmiles(fragment_smiles)
    if fragment is None:
        raise ValueError(f"Invalid SMILES: {fragment_smiles}")

    # Add Hs to both mol and fragment
    mol_h = Chem.AddHs(mol)
    frag_h = Chem.AddHs(fragment)

    # Get the editable molecule
    emol = Chem.RWMol(mol_h)

    # Remove the placeholder atom
    emol.RemoveAtom(atom_idx)

    # Combine the fragment
    combined = Chem.CombineMols(emol, frag_h)
    combined = Chem.AddHs(combined)
    Chem.SanitizeMol(combined)
    return combined

from TM_catalyst_framework.utils_atoms import get_placeholder_indices
from rdkit.Chem import AllChem

def attach_fragment(mol_template, symbol, fragment_smiles, attach_idx=0, bond_order=Chem.rdchem.BondType.SINGLE):
    """
    Attach a fragment to a molecule at a specific placeholder atom index.
    """
    fragment = Chem.MolFromSmiles(fragment_smiles)
    fragment = Chem.AddHs(fragment)
    
    emol = Chem.RWMol(mol_template)
    offset = emol.GetNumAtoms()
    
    # Add fragment atoms
    for atom in fragment.GetAtoms():
        emol.AddAtom(atom)
    
    # Add fragment bonds
    for bond in fragment.GetBonds():
        emol.AddBond(bond.GetBeginAtomIdx() + offset,
                     bond.GetEndAtomIdx() + offset,
                     bond.GetBondType())
    placeholder_idx = get_placeholder_indices(emol, symbol)
    print ("Attaching fragment at placeholder index:", placeholder_idx)
    # Connect placeholder to fragment
    emol.AddBond(placeholder_idx, attach_idx + offset, bond_order)
    
    # Remove placeholder atom
    emol.RemoveAtom(placeholder_idx)
    
    return emol.GetMol()
from TM_catalyst_framework.metal_template import load_ti_template_sdf
def build_geometry(R1="*CC(=O)O", 
                   R2="*CC(=O)O", 
                   R3="*CC(=O)O", template_name="ti_ureate_tetra_template.sdf") -> Chem.Mol:
    """
    Build a titanium-ureate complex from SDF template and attach substituents at placeholders.
    """
    mol = load_ti_template_sdf(template_name)
    
    replacements = {"F": R1, "Cl": R2, "Br": R3}

    # Attach fragments one by one
    # for symbol, idx_list in placeholder_indices.items():
    #     for idx in idx_list:
    #         mol = attach_fragment(mol, idx, replacements[symbol])
    for symbol, fragment_smiles in replacements.items():
        fragment_mol = Chem.MolFromSmiles(fragment_smiles)
        main_truncate_mol = Chem.DeleteSubstructs(mol, Chem.MolFromSmiles(symbol))
        combo = Chem.CombineMols(main_truncate_mol, fragment_mol)

        # mol = attach_fragment(mol, symbol, fragment_smiles)
    
    # Embed conformer if not present
    if mol.GetNumConformers() == 0:
        AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
        AllChem.UFFOptimizeMolecule(mol)
    
    return mol