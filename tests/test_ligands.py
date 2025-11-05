#%%
# import TM_catalyst_framework
from TM_catalyst_framework import Ligand
# L7 from DOI: 10.1021/acscatal.1c00014 
ligand_smi = "O=C(NC1=C(C(C)C)C=CC=C1C(C)C)C2=CC=CC=C2"
ureate = Ligand("ureate", ligand_smi, denticity=2, donor_atoms=["O", "N"])

# Generate 3D structure
ureate.generate_3D()

# Optimize using Open Babel
ureate.optimize_openbabel()
# Export files
ureate.save_xyz("ureate_opt.xyz")

# get xyz string
ureate_xyz = ureate.mol_3D_xyz
# %%
ureate_xyz
# %%
from TM_catalyst_framework import ORCA

ureate_orca_inp = ORCA().write_input(
    xyz_coordinates=ureate_xyz,
    functional="B3LYP",
    basis_set="def2-SVP",
    total_charge=0,
    multiplicity=1,
    task="Opt")

print (ureate_orca_inp)
# %%
# %%
