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
# Example: Create a main molecule and a fragment
general_fn = "u2_d2d2_iso1"
template_sdf = f"{general_fn}.sdf" 
template = load_ti_template_sdf(template_sdf, 
                                        package = "openbabel", 
                                        template_dir = "TM_catalyst_framework.template")
create_complex = Complex(metal_center="Ti", )
dummy_atomic_num = 9
dummy_atom, anchor_atom = create_complex.find_dummy_and_anchor(template, dummy_atomic_num)
print (dummy_atom)
template.OBMol.DeleteAtom(dummy_atom.OBAtom)

dummy_coord = dummy_atom.OBAtom.GetVector()
print(dummy_coord)
print (dummy_atom.OBAtom.GetX(), dummy_atom.OBAtom.GetY(), dummy_atom.OBAtom.GetZ())

frag = pybel.readstring("smi", "*C(=O)O")
frag.make3D()

attach_atom, connect_atom = create_complex.find_fragment_attachment_point(frag)
connect_idx = connect_atom.GetIdx()
connect_atom.SetVector(dummy_coord)

template.OBMol += frag.OBMol

frag_start_idx = template.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
connect_idx_in_merged = frag_start_idx + connect_idx - 1

template.OBMol.AddBond(anchor_atom.idx, connect_idx_in_merged, 1)  # 1 = single bond

for atom in template:
    # print (f"New molecule Atom idx: {atom.idx}, atomic num: {atom.OBAtom.GetAtomicNum()}")
    if atom.OBAtom.GetAtomicNum() == 0:  # *
        print (f"Found leftover * at index {atom.idx}, marking for deletion")
        template.OBMol.DeleteAtom(atom.OBAtom)
        
constraints = ob.OBFFConstraints()
metal_atomic_num = 22
metal_nbr_atom_ls = []
for atom in template:
    if atom.OBAtom.GetAtomicNum() == metal_atomic_num:  # * = dummy, 9 = fluorine
        metal_atom = atom
        # Find neighbor (the atom that was bonded to *)
        for nbr in pybel.ob.OBAtomAtomIter(atom.OBAtom):
            nbr_atom = template.atoms[nbr.GetIdx() - 1]
            nbr_idx = nbr_atom.idx
            # print (f"Add atom constraint to atom with nbr_idx: {nbr_idx}")
            constraints.AddAtomConstraint(nbr_idx)
import sys
# === Suppress logs ===
cerr_fd = sys.stderr.fileno()
devnull = os.open('/dev/null', os.O_WRONLY)
original_cerr = os.dup(cerr_fd)
os.dup2(devnull, cerr_fd)

try:
    ff = ob.OBForceField.FindForceField("uff")
    if not ff.Setup(template.OBMol, constraints):
        raise ValueError("Setup failed")
    
    print(f"Initial energy: {ff.Energy():.2f} kcal/mol")
    # ff.ConjugateGradients(force_field_steps)
    ff.ConjugateGradients(10000) 
    print(f"Final energy: {ff.Energy():.2f} kcal/mol")
    
    ff.GetCoordinates(template.OBMol)
finally:
    os.dup2(original_cerr, cerr_fd)
    os.close(original_cerr)
    os.close(devnull)

#%%
output_save_dir = "./test/"
if os.path.isdir(output_save_dir) != True:
    os.mkdir(output_save_dir)               # <-- where to save
output_sdf    = output_save_dir+"product.sdf"     
template.write("sdf", output_sdf, overwrite=True)

template
#%%


#%%
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
#%%
template
# %%



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
u1_d2_ON1:
u1_d2_ON2:

bis-ureate
u2_d1
u2_d1_d2_ON1
u2_d1_d2_ON2

u2_d1d2_iso1

u2_d2d2_iso7
"""
general_fn = "u2_d1d2_iso1"
template_sdf = f"{general_fn}.sdf"          # <-- your SDF file
ligand_test_smi = "C/[N]([*])=C([N](C)([*])C)/O[*]" # <----- ligand smile with attachement dummies *
# --------------------------------------------------------------
# Define output directory and filename
# --------------------------------------------------------------
output_save_dir = "./output_sdf/"
if os.path.isdir(output_save_dir) != True:
    os.mkdir(output_save_dir)               # <-- where to save
output_sdf    = output_save_dir+f"{general_fn}_product_N2.sdf"     

connect_N_dict = {35: "N2", 53: "N1"}
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
