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
from rdkit import Chem
from rdkit.Chem.Draw import MolToImage, MolDraw2DCairo
from PIL import Image
from io import BytesIO

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize

element_dict = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109, 'Ds': 110,
    'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}
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

def show_mol(d2d,mol,legend='',highlightAtoms=[]):
    d2d.DrawMolecule(mol,legend=legend, highlightAtoms=highlightAtoms)
    d2d.FinishDrawing()
    bio = BytesIO(d2d.GetDrawingText())
    return Image.open(bio)

class Ligand:
    def __init__(
            self, 
            smiles: str = "C", 
            charge: int = 1,
            denticity: int = 1, 
            mol_3D = True
            ):
        self.smiles = smiles
        self.charge = charge
        self.denticity = denticity
        self.ligand = pybel.readstring("smi", self.smiles)
        self.mol_3D = mol_3D
        if self.mol_3D == True:
            self.ligand.make3D()
        # self.obmol = None
        # self.mol_3D_xyz = None
        
    # def find_ligand_attachement_point(self, donor_atoms):
    #     attach_atom = None
    #     connect_atom = None
        
    #     if donor_atoms not in list(element_dict.keys()):
    #         raise ValueError(f"The connecting atom label not recognized; connect_label provided is: {donor_atoms}")
        
    #     connect_atomic_num = element_dict[donor_atoms]
    #     for atom in self.ligand:
    #         # find dummy atom with label *, which has atomic number == 0
    #         if atom.OBAtom.GetAtomicNum() == 0:
    #             # find the atom that is bonded to the dummy atom
    #             for nbr in pybel.ob.OBAtomAtomIter(atom.OBAtom):
    #                 break
    #             if nbr is None:
    #                 raise ValueError("Fragment * has no neighbor")
    #             # check if the connect atom matches connect_label
    #             # i.e. O, N1 or N2
    #             # skip to the next * atom if the nbr doesn't match the connect atomic number
    #             if nbr.GetAtomicNum() != connect_atomic_num:
    #                 continue
    #             else:
    #                 attach_atom = atom
    #                 connect_atom = nbr
    #                 break
    #     if attach_atom is None:
    #         raise ValueError("No * in fragment")
    #     if connect_atom is None:
    #         raise ValueError("Fragment * has no matching neighbor")  
    #     return attach_atom, connect_atom
    
    def add_num_to_dummy(self):
        dummy_idx_ls = []
        
        for idx, character in enumerate(self.smiles):
            if character == "*":
                dummy_idx_ls.append(idx+1)
        self.denticity = int(len(dummy_idx_ls))
        if self.denticity == 1:
            return self.smiles
        elif self.denticity == 2:
            smiles_draw = self.smiles[:dummy_idx_ls[0]] + f":1" + self.smiles[dummy_idx_ls[0]:dummy_idx_ls[1]] 
            smiles_draw = smiles_draw + f":2" + self.smiles[dummy_idx_ls[1]:]
            return smiles_draw
        else:            
            smiles_draw = self.smiles[:dummy_idx_ls[0]] + f":1"
            
            for idx, dummy_idx in enumerate(dummy_idx_ls[1:]):
                print (f"idx: {idx}")
                print (f"dummy_idx: {dummy_idx}")
                smiles_draw += self.smiles[dummy_idx_ls[idx]:dummy_idx] + f":{idx+2}"
            smiles_draw += self.smiles[dummy_idx_ls[-1]:]
            return smiles_draw
            

    def draw_ligand(self):
        for idx, character in enumerate(self.smiles):
            if character == "*":
                if self.smiles[idx+1] == ":":
                    draw_smiles = self.smiles
                else:
                    draw_smiles = self.add_num_to_dummy()
        if self.denticity > 0:
            if self.denticity == 1:
                mol = Chem.MolFromSmiles(draw_smiles, sanitize = False)
            else:
                print (f"draw_smiles: {draw_smiles}")
                mol = Chem.MolFromSmiles(draw_smiles, sanitize = False)
            d2d = MolDraw2DCairo(350,300)
            return show_mol(d2d, mol)
            

        # Draw the ureate ligand
        # Chem.MolFromSmiles(ligand_marked, sanitize = False)

    def attach_R_group(self, 
                       core_smiles: str,
                       R_smiles_ls: list,
                       return_smiles: bool = True
                       ):
        
        full_smiles = core_smiles
        for idx, sub_str_i in enumerate(R_smiles_ls):
            full_smiles += f".{sub_str_i}"
        print (full_smiles)
        
        full_mol = Chem.MolFromSmiles(full_smiles, sanitize=False)
        
        self.ligand = Chem.molzip(full_mol)
        if return_smiles == True:
            return Chem.MolToSmiles(self.ligand)
             
    
    def find_ligand_attachement_point(self, donor_atoms_idx):
        
        attach_atom = None
        connect_atom = None
        
        # if donor_atoms not in list(element_dict.keys()):
        #     raise ValueError(f"The connecting atom label not recognized; connect_label provided is: {donor_atoms}")
        target_dummy_count = 0

        for idx, character in enumerate(self.smiles):
            if character == "*":
                print (character)
                if self.smiles[idx+1] == ":":
                    smiles = self.smiles
                else:
                    smiles = self.add_num_to_dummy()
        print (f"smiles:{smiles}")
        print (f"donor_atoms_idx: {donor_atoms_idx}")
        for idx, character in enumerate(smiles):
            if character == "*":
                print (character)
                dummy_idx = int(smiles[idx+2])
                # print (f"dummy_idx: {dummy_idx}")
                if dummy_idx == donor_atoms_idx:
                    # print (f"dummy_idx: {dummy_idx}")
                    break
                else:
                    target_dummy_count += 1
        
        print (f"target_dummy_count: {target_dummy_count}")
        dummy_count = 0
        for atom in self.ligand:
            # find dummy atom with label *, which has atomic number == 0
            if atom.OBAtom.GetAtomicNum() == 0:
                # find the atom that is bonded to the dummy atom
                for nbr in pybel.ob.OBAtomAtomIter(atom.OBAtom):
                    break
                if nbr is None:
                    raise ValueError("Fragment * has no neighbor")
                # check if the connect atom matches connect_label
                # i.e. O, N1 or N2
                # skip to the next * atom if the nbr doesn't match the connect atomic number
                if dummy_count == target_dummy_count:
                    attach_atom = atom
                    connect_atom = nbr
                    break
                else:
                    dummy_count += 1
        if attach_atom is None:
            raise ValueError("No * in fragment")
        if connect_atom is None:
            raise ValueError("Fragment * has no matching neighbor")  
        return attach_atom, connect_atom
    
    # def add_substituients(self,
    #                       fragment_smiles_dict: dict):
        
    
    def save_xyz(self, filename=None):
        """Save ligand structure in XYZ format."""
        if self.mol_3D_xyz is None:
            self.optimize_openbabel()

        filename = filename or f"{self.name}.xyz"
        with open(filename, "w") as f:
            f.write(self.mol_3D_xyz)
        return filename
