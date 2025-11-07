#%%
from rdkit import Chem
from TM_catalyst_framework.utils_atoms import get_placeholder_indices
from TM_catalyst_framework.metal_template import load_ti_template_sdf
from rdkit.Chem import AllChem
from rdkit.Chem import AllChem, rdChemReactions
# Example: Create a main molecule and a fragment
template = load_ti_template_sdf("u1_d1_iso1.sdf", package= "openbabel") # Ethanol

# %%

###### TEST PACKAGE CODE ######
from TM_catalyst_framework.complex import Complex
import os
# --------------------------------------------------------------
# INPUTS
# --------------------------------------------------------------
template_sdf = "u1_d1_iso1.sdf"          # <-- your SDF file
fragment_smiles_dict = {"R1":"*c1ccccc1",
                        "R2":"*C(=O)O",
                        "R3":"*C(C)C"
                        }            # <-- fragments with * attachment point

# --------------------------------------------------------------
# Define output directory and filename
# --------------------------------------------------------------
output_save_dir = "./output_sdf/"
if os.path.isdir(output_save_dir) != True:
    os.mkdir(output_save_dir)               # <-- where to save
output_sdf    = output_save_dir+"product.sdf"     


# --------------------------------------------------------------
# CREATE COMPLEX: Add fragments to template
# --------------------------------------------------------------
create_complex = Complex(metal_center="Ti", )
create_complex.add_fragments(template_sdf = template_sdf,
        fragment_smiles_dict = fragment_smiles_dict,
        output_sdf = output_sdf,
        force_field= "uff",
        force_field_steps= 5000
                      )

# %%

#%%

# %%
# ligand attachement unit test
from openbabel import pybel
from TM_catalyst_framework.complex import Complex
import os

dummy_connect_check = "CF"
template = pybel.readstring("smi", dummy_connect_check)

create_complex = Complex(metal_center="Ti", )
ligand_test_smi = "C/[N]([*])=C([N](C)([*])C)/O[*]"
ligand = pybel.readstring("smi", ligand_test_smi)

connect_label_test = "2"
attach_atom, connect_atom = create_complex.find_ligand_attachement_point(ligand, 
                                                                         connect_label=connect_label_test)

connect_idx = connect_atom.GetIdx()
template.OBMol += ligand.OBMol

ligand_start_idx = template.OBMol.NumAtoms() - ligand.OBMol.NumAtoms() + 1
connect_idx_in_merged = ligand_start_idx + connect_idx - 1

# Add bond (1-based indices)
template.OBMol.AddBond(1, connect_idx_in_merged, 1)  # 1 = single bond
#%%
template
# %%



# %%

###### TEST PACKAGE CODE ######
from TM_catalyst_framework.complex import Complex
import os
# --------------------------------------------------------------
# INPUTS
# --------------------------------------------------------------
"""
mono_ureate
u1_d1
u1_d2_ON1:
u1_d2_ON2:

bis-ureate
u2_d1
u2_d1_d2_ON1
u2_d1_d2_ON2
u2_d2d2_iso7
"""
general_fn = "u2_d1_d2_ON1"
template_sdf = f"{general_fn}.sdf"          # <-- your SDF file
ligand_test_smi = "C/[N]([*])=C([N](C)([*])C)/O[*]" # <----- ligand smile with attachement dummies *
# --------------------------------------------------------------
# Define output directory and filename
# --------------------------------------------------------------
output_save_dir = "./output_sdf/"
if os.path.isdir(output_save_dir) != True:
    os.mkdir(output_save_dir)               # <-- where to save
output_sdf    = output_save_dir+f"{general_fn}_product.sdf"     


#unit test
create_complex = Complex(metal_center="Ti", )
create_complex.add_ligands(template_sdf = template_sdf,
                           ligand_smiles = ligand_test_smi,                      
        output_sdf = output_sdf,
        force_field= "uff",
        force_field_steps= 5000
                      )


# # --------------------------------------------------------------
# # CREATE COMPLEX: Add fragments to template
# # --------------------------------------------------------------
# create_complex = Complex(metal_center="Ti", )
# create_complex.add_ligands(template_sdf = template_sdf,
#                            ligand_smiles = ligand_test_smi,                      
#         output_sdf = output_sdf,
#         force_field= "uff",
#         force_field_steps= 5000
#                       )
# %%
