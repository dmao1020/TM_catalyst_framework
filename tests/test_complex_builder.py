#%%

from TM_catalyst_framework.ligands import Ligand
from TM_catalyst_framework.complex import Complex

from TM_catalyst_framework.ligands import Ligand
# workflow_example.py
from TM_catalyst_framework.io_utils import rdkit_mol_from_xyz_str
from TM_catalyst_framework.utils_atoms import atom_indices_by_symbol, first_atom_by_symbols
from TM_catalyst_framework.metal_template import metal_template_rdkit
from TM_catalyst_framework.attach import attach_ligand_to_site
from rdkit import Chem
from rdkit.Chem import AllChem

# 1) Read your template (ureate + Ti + placeholders)
from importlib import resources
# Load template file from package
with resources.files("TM_catalyst_framework.template").joinpath("u1_d1_iso1.xyz").open("r") as f:
    xyz_text = f.read()

# Convert XYZ string to RDKit Mol
ureate_mol = rdkit_mol_from_xyz_str(xyz_text)

# 2) Identify donor atom (O) and placeholders F/Cl/Br
ureate_O_idx = first_atom_by_symbols(ureate_mol, ["O"])
placeholder_indices = {
    "F": atom_indices_by_symbol(ureate_mol, "F"),
    "Cl": atom_indices_by_symbol(ureate_mol, "Cl"),
    "Br": atom_indices_by_symbol(ureate_mol, "Br"),
}
print("Donor O idx:", ureate_O_idx)
print("Placeholders:", placeholder_indices)
#%%
# 3) Prepare metal template (tetrahedral)
metal_mol, metal_idx, sites = metal_template_rdkit(metal_symbol="Ti", geometry="tetrahedral", bond_length=1.95)

# 4) Make sure ligand and metal have conformers (ureate_mol from pybel should have one)
# If ureate_mol lacks conformer, call AllChem.EmbedMolecule and UFFOptimize
if ureate_mol.GetNumConformers() == 0:
    ureate_mol = Chem.AddHs(ureate_mol)
    AllChem.EmbedMolecule(ureate_mol)
    AllChem.UFFOptimizeMolecule(ureate_mol)

# 5) Attach ureate O to the first site (for the tetrahedral case)
donor_site = sites[0]  # choose site for ureate O (you can pick others)
merged1 = attach_ligand_to_site(metal_mol, metal_idx, ureate_mol, ureate_O_idx, donor_site)

# 6) Generate NMe2 ligand (donor is N)
nme2_smiles = "CN(C)C"
nme2 = Chem.AddHs(Chem.MolFromSmiles(nme2_smiles))
AllChem.EmbedMolecule(nme2)
AllChem.UFFOptimizeMolecule(nme2)
nme2_N_idx = first_atom_by_symbols(nme2, ["N"])

# 7) Attach 3 NMe2 ligands to the remaining three sites (for tetrahedral)
merged2 = merged1
for site in sites[1:4]:
    merged2 = attach_ligand_to_site(merged2, metal_idx, nme2, nme2_N_idx, site)

# 8) merged2 now contains the combined complex; write to XYZ
from rdkit.Chem import rdmolfiles
xyz_text = rdmolfiles.MolToXYZBlock(merged2)
with open("ti_ureate_assembled.xyz", "w") as f:
    f.write(xyz_text)
print("Wrote ti_ureate_assembled.xyz")


# %%
