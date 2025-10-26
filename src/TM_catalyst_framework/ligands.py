"""
ligands.py
===========

Defines the Ligand class for handling organic/inorganic ligands in
the TM_catalyst_framework. Supports SMILES parsing via RDKit and
geometry optimization using Open Babel.
"""

from rdkit import Chem
from rdkit.Chem import AllChem
import openbabel
import pybel
from pathlib import Path


class Ligand:
    """
    Represents a ligand fragment.

    Parameters
    ----------
    name : str
        Common name of the ligand (e.g., 'ureate', 'NMe2').
    smiles : str
        SMILES string representing the ligand.
    denticity : int, optional
        Number of donor atoms (default = 1).
    donor_atoms : list[str], optional
        List of donor atom symbols (e.g., ['O', 'N']).
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

    # --------------------------------------------------------------
    # RDKit-based geometry generation
    # --------------------------------------------------------------
    def generate_3D(self, ff="UFF", maxIters=500):
        """Generate a 3D conformer and optimize it with RDKit."""
        mol = Chem.AddHs(self.mol)
        AllChem.EmbedMolecule(mol)
        AllChem.UFFOptimizeMolecule(mol, maxIters=maxIters)
        self.mol_3D = mol
        return mol

    # --------------------------------------------------------------
    # Open Babel optimization (optional)
    # --------------------------------------------------------------
    def optimize_openbabel(self, method="MMFF94", steps=250):
        """
        Optimize ligand geometry using Open Babel force fields.
        """
        if self.mol_3D is None:
            self.generate_3D()

        tmp_file = Path(f"{self.name}_rdkit.mol")
        Chem.MolToMolFile(self.mol_3D, str(tmp_file))

        obmol = next(pybel.readfile("mol", str(tmp_file)))
        obmol.localopt(forcefield=method, steps=steps)
        self.obmol = obmol
        self.mol_3D_xyz = obmol.write("xyz")
        tmp_file.unlink(missing_ok=True)
        return obmol

    # --------------------------------------------------------------
    # Export utilities
    # --------------------------------------------------------------
    def save_xyz(self, filename=None):
        """Save ligand structure in XYZ format."""
        if not hasattr(self, "mol_3D_xyz"):
            self.optimize_openbabel()

        filename = filename or f"{self.name}.xyz"
        with open(filename, "w") as f:
            f.write(self.mol_3D_xyz)
        return filename

    def save_mol(self, filename=None):
        """Save ligand structure in MOL format."""
        if self.mol_3D is None:
            self.generate_3D()
        filename = filename or f"{self.name}.mol"
        Chem.MolToMolFile(self.mol_3D, filename)
        return filename

    def __repr__(self):
        return f"Ligand(name={self.name}, denticity={self.denticity}, donor_atoms={self.donor_atoms})"
