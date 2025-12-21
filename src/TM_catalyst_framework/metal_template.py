# metal_template.py
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

import os
from importlib import resources
from TM_catalyst_framework.io_utils import rdkit_mol_from_xyz_str

from rdkit import Chem
from importlib import resources

from rdkit import Chem
from importlib import resources

from openbabel import pybel
from openbabel import openbabel as ob

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

class Template:
    def __init__(self,
                template_dir: str = "TM_catalyst_framework.template",
                package: str = "rdkit",
                metal_center: str = "Ti"):
        self.template_dir = "TM_catalyst_framework.template"
        self.package = package
        self.metal_center = metal_center

    def template_dir(self):
        return self.template_dir
    
    def add_template(self,
                     template = None,
                     template_name: str = "custom_template", 
                     template_path: str = "TM_catalyst_framework.template"):
        """Add a new template to the template directory.
        
        Parameters
        ----------
        template : Chem.Mol or pybel.Molecule or SDF file path
            The molecule object representing the template.
        template_name : str
            The name to save the template as.
        template_path : str
            The path to save the template file.
        Raises
        ------
        NotImplementedError 
            This function is not yet implemented. # TODO change this after implementation
        """

        # Check if template is provided
        if template is None:
            raise ValueError("Please provide a valid template molecule or SDF file path.")
        elif isinstance(template, str) and os.path.isfile(template):
            # Check if template is sdf file:
            if not template.lower().endswith('.sdf'):
                raise ValueError("Template file must be an SDF file.")
            # If template is a file path, save the SDF file to the template directory
            sdf_path = resources.files(template_path).joinpath(f"{template_name}.sdf").__fspath__()
            if os.path.exists(sdf_path):
                print(f"Warning: Overwriting existing template at {sdf_path}")
                user_input = input("Type 'yes' to confirm overwriting: ")
                if user_input.lower() == 'yes':
                    template_file = open(sdf_path, "w")
                    template_file.write(template)
                    template_file.close()
            else:
                print(f"Saving new template at {sdf_path}")
                template_file = open(sdf_path, "w")
                template_file.write(template)
                template_file.close()
            
        elif isinstance(template, Chem.Mol):
            mol = template
            raise ValueError("RDkit format not yet implemented.")
        elif isinstance(template, pybel.Molecule):
            mol = template
            raise ValueError("pybel format not yet implemented.")
        elif isinstance(template, ob.OBMol):
            conv = ob.OBConversion()
            conv.SetOutFormat("sdf")
            sdf_path = resources.files(template_path).joinpath(f"{template_name}.sdf").__fspath__()
            
            if os.path.exists(sdf_path):
                print(f"Warning: Overwriting existing template at {sdf_path}")
                user_input = input("Type 'yes' to confirm overwriting: ")
                if user_input.lower() == 'yes':
                    conv.WriteFile(template, sdf_path)
            else:
                print(f"Saving new template at {sdf_path}")
                conv.WriteFile(template, sdf_path)
        else:
            raise ValueError("Invalid template type provided.")
        

    def options(self):
        resource_root = resources.files('TM_catalyst_framework.template')
        # Iterate through the contents
        for item in resource_root.iterdir():
            filename = item.name
            if "sdf" in filename:
                print (filename[:-4])

    def load_ti_template_sdf(self,
            template_name = "ti_ureate_tetra_template.sdf", 
            removeHs_statement = False
                ) -> Chem.Mol:
        """
        Load tetrahedral Ti-ureate template from SDF (bonded structure).
        Returns an RDKit Mol object with hydrogens preserved.
        """
        # Get the filesystem path of the SDF file inside the package
        print(resources.files(self.template_dir).joinpath(template_name).is_file())
        sdf_path = resources.files(self.template_dir).joinpath(template_name).__fspath__()

        if self.package == "rdkit":
            # RDKit reads SDF files from path
            suppl = Chem.SDMolSupplier(sdf_path, removeHs=removeHs_statement)
            for mol in suppl:
                if mol is not None:
                    print(f"Molecule with {mol.GetNumAtoms()} atoms")
                else:
                    raise ValueError(f"Failed to read SDF template from {sdf_path}")
        elif self.package == "openbabel":
            from openbabel import pybel
            mol = next(pybel.readfile("sdf", sdf_path))
            if mol is not None:
                print(f"Molecule with {len(mol.atoms)} atoms")
            if self.metal_center != "Ti":
                print ("Replace Ti with {self.metal_center}.")
                for atom in mol:
                    if atom.OBAtom.GetAtomicNum() == element_dict["Ti"]:  # * = dummy, 9 = fluorine
                        atom.OBAtom.SetAtomicNum(element_dict[str(self.metal_center)])
        return mol



    def load_ti_template_xyz(template_name="ti_ureate_template.xyz"):
        """Load the Ti-ureate template as RDKit Mol."""
        with resources.files("TM_catalyst_framework.template").joinpath(template_name).open("r") as f:
            xyz_text = f.read()
        return rdkit_mol_from_xyz_str(xyz_text)

    def metal_template_rdkit(metal_symbol="Ti", geometry="tetrahedral", bond_length=1.95):
        """
        Return an RDKit Mol with one metal atom (at origin) and a list of site coordinates.
        bond_length is the approximate metal-donor distance in Angstroms.
        """
        # Create RDKit single-atom mol
        rwm = Chem.RWMol()
        metal_idx = rwm.AddAtom(Chem.Atom(metal_symbol))
        mol = rwm.GetMol()

        # create Conformer with metal at origin
        conf = Chem.Conformer(mol.GetNumAtoms())
        conf.SetAtomPosition(metal_idx, (0.0, 0.0, 0.0))
        mol.AddConformer(conf, assignId=True)

        # define ideal site vectors
        if geometry == "tetrahedral":
            vecs = np.array([
                [ 1,  1,  1],
                [-1, -1,  1],
                [-1,  1, -1],
                [ 1, -1, -1],
            ]) / np.sqrt(3)
        elif geometry == "trigonal_bipyramidal":
            vecs = np.array([
                [0, 0,  1],
                [0, 0, -1],
                [ np.cos(0), np.sin(0), 0],
                [ np.cos(2*np.pi/3), np.sin(2*np.pi/3), 0],
                [ np.cos(4*np.pi/3), np.sin(4*np.pi/3), 0],
            ])
        else:
            raise ValueError("Unsupported geometry")

        sites = (vecs * bond_length).tolist()
        return mol, metal_idx, sites
