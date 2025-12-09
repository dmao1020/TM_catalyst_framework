import numpy as np
import os
from importlib import resources
from rdkit import Chem

class QChem:
    def __init__(self,
                 software: str = "ORCA",
                 ):
        self.software = software
        self.supported_software = ["ORCA"]
        if self.software not in self.supported_software:
            raise ValueError(f"Unsupported software: {self.software}. Supported software: {self.supported_software}")
    
    def return_supported_software(self):
        """Return a list of supported quantum chemistry software."""
        return self.supported_software
    
    def generate_input(self, **kwargs):
        """Generate input file for the specified quantum chemistry software."""
        if self.software == "ORCA":
            return self.generate_orca_input(**kwargs)
        else:
            raise NotImplementedError(f"Input generation for {self.software} is not implemented yet.")

    def generate_orca_input(self,
                       method: str = "DFT",
                       basis_set: str = "def2-SVP",
                       task: str = "opt",
                       # solvent_model: str = "CPCM",
                       # solvent: str = "water",
                       # dispersion_correction: str = "D3BJ",
                       functional: str = None,
                       charge : int = 0,
                       multiplicity: int = 1,
                       mol = None,
                       atoms_coords: list = None,
                       filename: str = "geometry_optimization.inp",
                       additional_blocks: dict = None
                       ) -> None:
        """
        Generate a input file from Mol object.

        Parameters:
        - method (str): Quantum chemistry method (e.g., 'DFT', 'HF').
        - basis_set (str): Basis set to be used (e.g., '6-31G', 'cc-pVDZ').
        - task (str): Type of calculation (e.g., 'opt', 'freq', 'single-point').
        - charge (int): Overall charge of the complex. (default: 0)
        - multiplicity (int): Spin multiplicity of the complex. (default: 1)
        - atoms_coords (list of tuples): List of (atom_symbol, x, y, z) for Cartesian coordinates.
        - filename (str): Output filename.
        - additional_blocks (dict): Optional dictionary of additional ORCA blocks, e.g.,
                                    {'pal': 'nprocs 4\n end', 'gbasis': 'NewGBS ... end'}
        
        Example usage:
        atoms = [('O', 0.0, 0.0, 0.0), ('H', 0.96, 0.0, 0.0), ('H', -0.96, 0.0, 0.0)]
        generate_orca_input(atoms_coords=atoms)
        """

        if mol is not None:
            if type(mol) == Chem.Mol:
                # print("The complex is an RDKit Mol object.")
                # TODO this should be tested
                atoms_coords = []
                conf = mol.GetConformer()
                for atom in mol.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    atoms_coords.append((atom.GetSymbol(), pos.x, pos.y, pos.z))
            elif type(mol).__name__ == "Molecule":  # Open Babel Molecule
                tmp_xyz_path = resources.files("TM_catalyst_framework.tmp").joinpath("tmp_mol.xyz").__fspath__()
                # print("The complex is an Open Babel Molecule object.")
                mol.write("xyz", tmp_xyz_path, overwrite=True)
                with open(tmp_xyz_path, 'r') as f:
                    lines = f.readlines()
                num_atoms = int(lines[0].strip())
                atoms_coords = []
                for line in lines[2:2+num_atoms]:
                    parts = line.split()
                    atom_symbol = parts[0]
                    x, y, z = map(float, parts[1:4])
                    atoms_coords.append((atom_symbol, x, y, z))
            else:
                raise TypeError("Unsupported mol type. Provide an RDKit Mol or Open Babel Molecule object.")
        elif atoms_coords is None:
            raise ValueError("Either 'mol' or 'atoms_coords' must be provided.")
        else:
            print("Using provided atoms_coords for input generation.")
        
        print (atoms_coords)

        # Start writing the file
        task_lower = task.lower()
        if task_lower == "opt":
            task_label = "Opt"
        elif task_lower == "freq":
            task_label = "Freq"
        elif task_lower in ["single-point", "sp"]:
            task_label = "SP"
        else:
            raise ValueError(f"Unsupported task: {task_label}. Supported tasks: opt, freq, single-point")
        with open(filename, 'w') as f:
            # Method line
            if method == "DFT":
                f.write(f'! {functional} {basis_set} {task_label}\n\n')
            else:
                f.write(f'! {method} {basis_set} {task_label}\n\n')
            
            # Optional additional blocks (e.g., %pal, %maxcore)
            if additional_blocks:
                for block_name, block_content in additional_blocks.items():
                    f.write(f'%{block_name}\n{block_content}\nend\n\n')
            
            # Geometry block
            f.write(f'* xyz {charge} {multiplicity}\n')
            for atom, x, y, z in atoms_coords:
                f.write(f'{atom:2s} {x:>10.6f} {y:>10.6f} {z:>10.6f}\n')
            f.write('*\n')
        
        print(f"ORCA input file '{filename}' generated successfully.")
        print("You can now run: orca {filename} > output.out")