from rdkit import Chem
from rdkit.Chem import AllChem
import openbabel
import pybel
from pathlib import Path

class ComplexBuilder:
    """
    Building transition metal complexes.
    Parameters
    ----------


    just a placement
    """

    def __init__(self, name, smiles, denticity=1, donor_atoms=None):
        self.name = name
        self.smiles = smiles
        self.denticity = denticity
        self.donor_atoms = donor_atoms or []
        self.mol = Chem.MolFromSmiles(smiles)
        self.mol_3D = None

        if self.mol is None:
            raise ValueError(f"Invalid SMILES for ligand: {smiles}")
