"""
ligands.py
===========

Defines the Ligand class for handling organic/inorganic ligands in
the TM_catalyst_framework. Supports SMILES parsing via RDKit and
geometry optimization using Open Babel.
"""
# ligands.py (updated optimize_openbabel)
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel
import numpy as np

import itertools
from itertools import product


def permutations_with_repetition(iterable, length):
    """
    Generates all permutations with repetition of a given iterable.

    Args:
        iterable: The input iterable (e.g., a list, string, or tuple).
        length: The desired length of each permutation.

    Returns:
        An iterator yielding tuples representing the permutations.
    """
    return product(iterable, repeat=length)

def octahedral_N1N2_dictionary(iso_num):
    elements = ['N1', 'N2']
    perm_length = 2

    if iso_num in [1, 2]:
        N1_N2_combo = itertools.combinations_with_replacement(elements, 2)
    elif iso_num in range(3, 9):
        N1_N2_combo = permutations_with_repetition(elements, perm_length)

    print (f"N1_N2_combo: {N1_N2_combo}")
    for p in N1_N2_combo:
        print (p)
        Br, I = p[0], p[1]
        N1N2_map = {}
        if Br == "N1":
            N1N2_map = {"N1":[35], "N2":[9]}
        elif Br == "N2":
            N1N2_map = {"N1":[9], "N2":[35]}
        if I == "N1":
            N1N2_map["N1"].append(53)
            N1N2_map["N2"].append(17)
        elif I == "N2":
            N1N2_map["N1"].append(17)
            N1N2_map["N2"].append(53)
        print (f"N1N2_map:{N1N2_map}")
    return N1N2_map

class Ligand:
    def __init__(
            self, 
            name, 
            smiles, 
            charge = 0, 
            denticity=1, 
            donor_atoms=None
            ):
        self.name = name
        self.smiles = smiles
        self.charge = charge
        self.denticity = denticity
        self.donor_atoms = donor_atoms or []
        self.mol = Chem.MolFromSmiles(smiles)
        self.mol_3D = None
        self.obmol = None
        self.mol_3D_xyz = None

        if self.mol is None:
            raise ValueError(f"Invalid SMILES for ligand: {smiles}")

    def generate_3D(self, ff="UFF", maxIters=500):
        """Generate a 3D conformer and optimize it with RDKit."""
        mol = Chem.AddHs(self.mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol, maxIters=maxIters)
        self.mol_3D = mol
        return mol
    
    import numpy as np

    def attach_to_site(self, metal_mol, metal_idx, donor_coords, bond_type=Chem.BondType.DATIVE):
        """
        Align this ligand so that its donor atom sits at the specified coordinate,
        then merge it into the metal complex molecule.

        Parameters
        ----------
        metal_mol : rdkit.Chem.Mol
            The molecule containing the metal center.
        metal_idx : int
            The atom index of the metal in that molecule.
        donor_coords : tuple of float
            (x, y, z) coordinates where the donor atom should be placed.
        bond_type : rdkit.Chem.BondType
            Type of bond connecting ligand to metal.
        """

        if self.mol_3D is None:
            self.generate_3D()

        # --- Find donor atom (first matching atom in donor_atoms list) ---
        conf = self.mol_3D.GetConformer()
        donor_idx = None
        for atom in self.mol_3D.GetAtoms():
            if atom.GetSymbol() in self.donor_atoms:
                donor_idx = atom.GetIdx()
                break
        if donor_idx is None:
            raise ValueError(f"No donor atom found in ligand {self.name}")

        # --- Compute translation vector ---
        donor_pos = np.array(conf.GetAtomPosition(donor_idx))
        donor_coords = np.array(donor_coords)
        translation = donor_coords - donor_pos

        # --- Apply translation to all atoms ---
        for i in range(self.mol_3D.GetNumAtoms()):
            pos = np.array(conf.GetAtomPosition(i))
            new_pos = pos + translation
            conf.SetAtomPosition(i, tuple(new_pos))

        # --- Merge ligand into metal_mol ---
        combined = Chem.CombineMols(metal_mol, self.mol_3D)
        editable = Chem.EditableMol(combined)

        # Metal atom index remains metal_idx
        # Ligand atoms are offset by number of atoms in metal_mol
        offset = metal_mol.GetNumAtoms()
        new_donor_idx = donor_idx + offset

        # Add the M–L bond
        editable.AddBond(metal_idx, new_donor_idx, bond_type)

        merged_mol = editable.GetMol()
        return merged_mol


    def optimize_openbabel(self, method="MMFF94", steps=250):
        """
        Optimize ligand geometry using Open Babel force fields.
        Uses pybel.readstring to avoid file-based readfile issues.
        """
        if self.mol_3D is None:
            self.generate_3D()

        # Convert RDKit Mol to MolBlock string
        molblock_str = Chem.MolToMolBlock(self.mol_3D)

        # Create a Pybel molecule from MolBlock string
        obmol = pybel.readstring("mol", molblock_str)

        # Perform force-field optimization
        obmol.localopt(forcefield=method, steps=steps)

        # Store the optimized molecule and XYZ string
        self.obmol = obmol
        self.mol_3D_xyz = obmol.write("xyz")

        return obmol

    def save_xyz(self, filename=None):
        """Save ligand structure in XYZ format."""
        if self.mol_3D_xyz is None:
            self.optimize_openbabel()

        filename = filename or f"{self.name}.xyz"
        with open(filename, "w") as f:
            f.write(self.mol_3D_xyz)
        return filename
