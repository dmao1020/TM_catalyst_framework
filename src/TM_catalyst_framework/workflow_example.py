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
