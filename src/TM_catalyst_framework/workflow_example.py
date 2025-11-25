#%%
from rdkit import Chem
from TM_catalyst_framework import Complex
from TM_catalyst_framework.utils_atoms import get_placeholder_indices
from TM_catalyst_framework.metal_template import load_ti_template_sdf
from rdkit.Chem import AllChem
from rdkit.Chem import AllChem, rdChemReactions
from openbabel import pybel
import numpy as np
import openbabel.openbabel as ob

#%%

# %%

###### TEST PACKAGE CODE: none octahedral complexes ######
from TM_catalyst_framework.complex import Complex
import os
# --------------------------------------------------------------
# INPUTS
# --------------------------------------------------------------
"""
mono_ureate
u1_d1
u1_d2

bis-ureate
u2_d1
u2_d1d2_iso1
u2_d1d2_iso2

u2_d2d2_iso1 - u2_d2d2_iso8
"""
general_fn = "u2_d2d2_iso1"
template_sdf = f"{general_fn}.sdf"          # <-- your SDF file
ligand_test_smi = "C/[N]([*])=C([N](C)([*])C)/O[*]" # <----- ligand smile with attachement dummies *

precursor_ligand = "*N(C)C"
# --------------------------------------------------------------
# Define output directory and filename
# --------------------------------------------------------------
output_save_dir = "./output_sdf/"
if os.path.isdir(output_save_dir) != True:
    os.mkdir(output_save_dir)               # <-- where to save
output_sdf    = output_save_dir+f"{general_fn}_product_N2.sdf"     

connect_N_dict = {10: "N2", 18: "N1"}
#unit test
create_complex = Complex(metal_center="Ti", )
create_complex.add_ligands(
    connect_N_dict = connect_N_dict,
    template_sdf = template_sdf,
    ligand_smiles = ligand_test_smi,                              
    output_sdf = output_sdf,
    force_field= "uff",#MMFF94, Ghemical. uff
    force_field_steps= 10000
                )

complex_mol = create_complex.complex
create_complex.attach_monodentate_ligand(complex_mol,
                                         precursor_ligand,
                                         2
                                         )


complex = create_complex.complex
complex.write("sdf", output_sdf, overwrite=True)
#%%
create_complex.complex
# %%
#%%
###### TEST PACKAGE CODE: none octahedral complexes ######
from TM_catalyst_framework.complex import Complex, Ti_ureate_Complex
import os
# --------------------------------------------------------------
# INPUTS
# --------------------------------------------------------------
"""
tetrahedral
TBP
octahedral
"""
general_fn = "u2_d2d2_iso2"
geometry_fn = "octahedral"
template_sdf = f"{geometry_fn}.sdf"          # <-- your SDF file
ligand_test_smi = "C[*]/N=C(O[*])\[N]([*])(CC1=CC=CC=C1)C(C)(C)C"
#"C/[N]([*])=C([N](C)([*])C)/O[*]" # <----- ligand smile with attachement dummies *

precursor_ligand = "*N(C)C"
# --------------------------------------------------------------
# Define output directory and filename
# --------------------------------------------------------------
output_save_dir = "./geometry_output_sdf/"
if os.path.isdir(output_save_dir) != True:
    os.mkdir(output_save_dir)               # <-- where to save
output_sdf    = output_save_dir+f"{general_fn}_product_N2.sdf"     

connect_N_identity_dict = {3: "N2", 4: "N1"}
metal_coord_dummy_map = {"O1":2, "N1":3, "O2":5, "N2":4}
#unit test
create_complex = Ti_ureate_Complex(metal_center="Ti", )
create_complex.add_ligands_wt_geometry(
    connect_N_identity_dict = connect_N_identity_dict,
    template_sdf = template_sdf,
    ligand_smiles = ligand_test_smi,                              
    output_sdf = output_sdf,
    force_field= "uff",#MMFF94, Ghemical. uff
    force_field_steps= 10000,
    metal_coord_dummy_map = metal_coord_dummy_map
                )

# complex_mol = create_complex.complex
# create_complex.attach_monodentate_ligand(complex_mol,
#                                          precursor_ligand,
#                                          1, 
#                                          halogen_mapping = True
#                                          )
# create_complex.attach_monodentate_ligand(complex_mol,
#                                          precursor_ligand,
#                                          6,
#                                          halogen_mapping = True
#                                          )


complex = create_complex.complex
complex.write("sdf", output_sdf, overwrite=True)
#%%
create_complex.complex


#%%
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

# %%

# %%

###### TEST PACKAGE CODE: octahedral complexes ######
from TM_catalyst_framework.complex import Complex
from TM_catalyst_framework.ligands import *
import os
# --------------------------------------------------------------
# INPUTS
# --------------------------------------------------------------
"""
u2_d2d2_iso1
u2_d2d2_iso2
u2_d2d2_iso3
u2_d2d2_iso4
u2_d2d2_iso5
u2_d2d2_iso6
u2_d2d2_iso7
u2_d2d2_iso8
"""
for iso_num in range(1,2):
    general_fn = f"u2_d2d2_iso{iso_num}"
    template_sdf = f"{general_fn}.sdf"          # <-- your SDF file

    # fragment_smiles_dict = {"R1":"*c1ccccc1",
    #                         "R2":"*C(=O)O",
    #                         "R3":"*C(C)C"
    #                         } 

    fragment_smiles_dict = {"R1":"*c1ccccc1",
                            "R2":"*C(=O)O",
                            "R3":"*C(C)C"
                            } 
    
    N1N2_map = octahedral_N1N2_dictionary(iso_num)
    print (f"N1N2_map: {N1N2_map}")
    # --------------------------------------------------------------
    # Define output directory and filename
    # --------------------------------------------------------------
    output_save_dir = "./output_sdf/"
    if os.path.isdir(output_save_dir) != True:
        os.mkdir(output_save_dir)               # <-- where to save
    output_sdf = output_save_dir+f"{general_fn}_product.sdf"     


    #unit test
    create_complex = Complex(metal_center="Ti")
    create_complex.add_fragments(template_sdf,
                                fragment_smiles_dict,
                                N1N2_map,
                                output_sdf,
                                force_field= "uff",#MMFF94, Ghemical. uff
                                force_field_steps= 10000
                                )
#%%
# %%
# %%

# %%

# %%
