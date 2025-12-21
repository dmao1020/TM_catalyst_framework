import numpy as np
import os
from importlib import resources
from rdkit import Chem
from pathlib import Path


class QChem:
    def __init__(self,
                 software: str = "ORCA",
                 ):
        self.software = software
        self.software_options = ["ORCA", "Gaussian"]
        if self.software not in self.software_options:
            raise ValueError(f"Unsupported software: {self.software}. Supported software: {self.software_options}")
    
    def return_supported_software(self):
        """Return a list of supported quantum chemistry software."""
        return self.software_options
    
    def generate_input(self, **kwargs):
        """Generate input file for the specified quantum chemistry software."""
        if self.software == "ORCA":
            return self.generate_orca_input(**kwargs)
        elif self.software == "Gaussian":
            raise NotImplementedError("Input generation for Gaussian is not implemented yet.")
        else:
            raise NotImplementedError(f"Input generation for {self.software} is not implemented yet.")

    def generate_orca_input(self,
                       method: str = "DFT",
                       basis_set: str = "def2-SVP",
                       dispersion_correction: str = None,
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
                       memory_allocated: bool = True,
                       nprocs: int = 8,
                       maxcore: int = 14000,
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
            elif type(mol) == str:
                raise FileNotFoundError(f"mol input: {mol} is provided as a filepath, but code is still under construction.")
                # TODO: finish this part, allow SDF and XYZ file reading
                # Assume it's a file path
                # if os.path.isfile(mol):
                #     with open(mol, 'r') as f:
                #         lines = f.readlines()
                #     num_atoms = int(lines[0].strip())
                #     atoms_coords = []
                #     for line in lines[2:2+num_atoms]:
                #         parts = line.split()
                #         atom_symbol = parts[0]
                #         x, y, z = map(float, parts[1:4])
                #         atoms_coords.append((atom_symbol, x, y, z))
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
                if dispersion_correction is not None:
                    f.write(f'! {functional} {basis_set} {dispersion_correction} {task_label}\n\n')
                else:
                    f.write(f'! {functional} {basis_set} {task_label}\n\n')
            else:
                f.write(f'! {method} {basis_set} {task_label}\n\n')
            if memory_allocated:
                f.write(f'%pal\n    nprocs {nprocs}\nend\n\n')
                f.write(f'%maxcore {maxcore}\n\n')
            
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
    
    def generate_slurm_script(self,
            account_name: str = "def-rkrems", # Replace with your actual SLURM account
            molecule_name: str = "ti_d2d2_iso1", # Name of your molecule/system (used in file names)
            ntasks: int = 8, # Number of CPUs (should match %pal nprocs in ORCA input)
            memory: int = 16000,  # Memory per CPU in MB (e.g., 3800M)
            days: int = 00,  # Days for walltime (format as two digits if needed)
            hours: int = 10,   # Hours for walltime (00-23, format as two digits)
            minutes: int = 00, # Minutes for walltime (00-59, format as two digits)
            orca_inp_filename: str = "orca", # ORCA input filename
            script_filename: str = f"run_orca.sh" # Optional: customize the output .sh filename
        ): 
        """
        Generate a SLURM submission script for an ORCA job.
        Parameters:
        - account_name (str): SLURM account name.
        - molecule_name (str): Name of the molecule/system (used in file names).
        - ntasks (int): Number of CPUs (should match nprocs in ORCA input).
        - memory (int): Memory per CPU in MB.
        - days (int): Days for walltime.
        - hours (int): Hours for walltime.
        - orca_inp_filename (str): ORCA input filename (without extension).
        - script_filename (str): Output SLURM script filename.
        """
        lines = (
            f"#!/bin/bash\n"
            f"#SBATCH --job-name={molecule_name}\n"
            f"#SBATCH --account={account_name}\n"
            f"#SBATCH --ntasks={ntasks}               # cpus, should match nprocs in ORCA input\n"
            f"#SBATCH --mem-per-cpu={memory}M         # memory per cpu\n"
            f"#SBATCH --time={days}-{hours}:{minutes}         # time limit (DD-HH:MM)\n"
            f"#SBATCH --output={molecule_name}.log    # standard output\n"
            f"#SBATCH --error={molecule_name}.err     # standard error\n\n"
            f"module load StdEnv/2023 gcc/12.3 openmpi/4.1.5 orca/6.0.1\n\n"
            f"$EBROOTORCA/orca {orca_inp_filename}.inp > {orca_inp_filename}.out\n"
        )

        # Write the script to file
        with open(script_filename, "w") as f:
                for line in lines:
                    f.write(line)
        print(f"SLURM script successfully written to: {script_filename}")
        print("You can now submit it with: sbatch " + script_filename)