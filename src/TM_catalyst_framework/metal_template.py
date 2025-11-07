# metal_template.py
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


from importlib import resources
from TM_catalyst_framework.io_utils import rdkit_mol_from_xyz_str

from rdkit import Chem
from importlib import resources

from rdkit import Chem
from importlib import resources

from openbabel import pybel

def load_ti_template_sdf(template_name="ti_ureate_tetra_template.sdf", 
                         template_dir = "TM_catalyst_framework.template",
                         removeHs_statement = False,
                         package: str = "rdkit"
                         ) -> Chem.Mol:
    """
    Load tetrahedral Ti-ureate template from SDF (bonded structure).
    Returns an RDKit Mol object with hydrogens preserved.
    """
    # Get the filesystem path of the SDF file inside the package
    print(resources.files(template_dir).joinpath(template_name).is_file())
    sdf_path = resources.files(template_dir).joinpath(template_name).__fspath__()

    if package == "rdkit":
        # RDKit reads SDF files from path
        suppl = Chem.SDMolSupplier(sdf_path, removeHs=removeHs_statement)
        for mol in suppl:
            if mol is not None:
                print(f"Molecule with {mol.GetNumAtoms()} atoms")
            else:
                raise ValueError(f"Failed to read SDF template from {sdf_path}")
    elif package == "openbabel":
        from openbabel import pybel
        mol = next(pybel.readfile("sdf", sdf_path))
        if mol is not None:
            print(f"Molecule with {len(mol.atoms)} atoms")
    # mol = next(suppl)
    # if mol is None:
    #     raise ValueError(f"Failed to read SDF template from {sdf_path}")
    # Iterate and process molecules
    
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
