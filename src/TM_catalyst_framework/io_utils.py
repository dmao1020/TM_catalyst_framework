# io_util.py
from pathlib import Path
import subprocess
from rdkit import Chem
import py3Dmol
import os

class mol_visual:
    def visualize_sdf_molecule(filename: str):
        """
        Reads an SDF file and displays the 3D molecular structure using py3Dmol.

        Args:
            sdf_filepath (str): The path to the SDF file.
        """
        try:
            # Read the molecule from the SDF file
            mol = Chem.MolFromMolFile(filename)

            if mol is None:
                print(f"Error: Could not read molecule from {filename}. "
                    "Ensure the file is valid and contains 3D coordinates.")
                return

            # Convert the RDKit molecule to a MolBlock string in SDF format
            mol_block = Chem.MolToMolBlock(mol)

            # Create a py3Dmol viewer
            view = py3Dmol.view(width=600, height=400)

            # Add the molecule to the viewer
            view.addModel(mol_block, "sdf")

            # Set the style for visualization (e.g., sticks, spheres)
            view.setStyle({'stick': {}})  # Or {'sphere': {}} or {'cartoon': {}} for proteins

            # Zoom to fit the molecule in the view
            view.zoomTo()

            # Display the viewer
            view.show()

        except FileNotFoundError:
            print(f"Error: SDF file not found at {filename}")
        except Exception as e:
            print(f"An error occurred: {e}")
    def visualize_mol_molecule(mol):
        """
        Reads an SDF file and displays the 3D molecular structure using py3Dmol.

        Args:
            sdf_filepath (str): The path to the SDF file.
        """
        try:
            # Read the molecule from the SDF file
            mol_sdf_string = mol.write("sdf")

            # if mol is None:
            #     print(f"Error: Could not read molecule from {filename}. "
            #         "Ensure the file is valid and contains 3D coordinates.")
            #     return

            # # Convert the RDKit molecule to a MolBlock string in SDF format
            # Create a py3Dmol viewer
            view = py3Dmol.view(width=600, height=400)

            # Add the molecule to the viewer
            view.addModel(mol_sdf_string, "sdf")

            # Set the style for visualization (e.g., sticks, spheres)
            view.setStyle({'stick': {}})  # Or {'sphere': {}} or {'cartoon': {}} for proteins

            # Zoom to fit the molecule in the view
            view.zoomTo()

            # Display the viewer
            view.show()
        except Exception as e:
            print(f"An error occurred: {e}")

class ORCA:
    def __init__(self, orca_path: str = "orca"):
        """Initialize ORCA runner."""
        self.orca_path = orca_path

    def check_xyz_format(self, xyz_string: str) -> bool:
        """
        Check if the provided string is in
        valid XYZ format for ORCA input.
        """
        lines = xyz_string.strip().splitlines()
        # print ("lines[0]:", lines[0])
        # print ("len(lines[0]):", len(lines[0].split()))
        if len(lines[0].split()) == 1:
            return xyz_string[2:]
        elif len(lines[0].split()) == 4:
            return xyz_string
        

    def write_input(
        self,
        xyz_coordinates: str,
        total_charge: int = 0,
        multiplicity: int = 1,
        functional: str = "B3LYP",
        basis_set: str = "def2-SVP",
        task: str = "Opt",
        nprocs: int = 16,
        maxcore: int = 16000
    ) -> str:
        """
        Generate ORCA input file content for a given molecule.

        Parameters
        ----------
        xyz_coordinates : str
            The XYZ coordinates of the molecule (excluding header line).
        functional : str, optional
            DFT functional to use (default: B3LYP).
        basis_set : str, optional
            Basis set to use (default: def2-SVP).
        task : str, optional
            Type of calculation (default: Opt).
        nprocs : int, optional
            Number of CPU cores.
        maxcore : int, optional
            Memory per core in MB.
        """
        checked_xyz = self.check_xyz_format(xyz_coordinates)
        # print ("checked_xyz:", checked_xyz )
        orca_input = f"""! {functional} {basis_set} {task}

%pal
  nprocs {nprocs}
end

%maxcore {maxcore}

%scf
  MaxIter 10000
end

* xyz {total_charge} {multiplicity}
{checked_xyz.strip()}
*
"""
        return orca_input

    def run(self, input_file: Path):
        """Run ORCA on a given input file."""
        result = subprocess.run(
            [self.orca_path, str(input_file)],
            capture_output=True,
            text=True
        )
        return result


# utils_io.py
from openbabel import pybel
from rdkit import Chem
import tempfile

def rdkit_mol_from_xyz_file(xyz_path: str):
    """Read XYZ file and return RDKit Mol."""
    obmol = next(pybel.readfile("xyz", xyz_path))
    return Chem.MolFromSmiles(obmol.write("can").strip())

def rdkit_mol_from_xyz_str(xyz_str: str):
    """Convert XYZ string into RDKit Mol (temporary file method)."""
    with tempfile.NamedTemporaryFile(suffix=".xyz", mode="w", delete=False) as tmp:
        tmp.write(xyz_str)
        tmp.flush()
        obmol = next(pybel.readfile("xyz", tmp.name))
    return Chem.MolFromSmiles(obmol.write("can").strip())

