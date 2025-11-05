from rdkit import Chem
from rdkit.Chem import AllChem
import openbabel
import pybel
from pathlib import Path
from ase import Atoms
import networkx as nx
import json

class MetalCenter:
    """
    Represents a transition metal center.
    Parameters
    ----------


    just a placement
    """

    def __init__(
            self, 
            element = "Ti", 
            coordination_number = 6
            ):
        self.element = element
        self.coordination_number = coordination_number
        self.template = self.load_template()
    
    def load_template(self):
        """
        Load a template structure for the metal center.
        Returns
        -------
        pybel.Molecule
            The template structure as a Pybel molecule.
        """
        if self.element == "Ti":
            coordination_ls = [4, 5, 6]
            if self.coordination_number not in coordination_ls:
                raise ValueError(f"Unsupported coordination number for Ti: {self.coordination_number}")
            else:       
                return Atoms(self.element, # define the metal center
                             positions = [[0,0,0]] # place the metal at the origin
                                )
    
    def attach_ligand(self, ligand, position_index: int):
        """
        Attach a ligand to the metal center at a specified position.
        Parameters
        ----------
        ligand : Atoms
            The ligand to attach.
        position_index : int
            The index of the position on the metal center to attach the ligand.
        Returns
        -------
        Atoms
            The combined structure of the metal center and ligand.
        """
        pass
        
    