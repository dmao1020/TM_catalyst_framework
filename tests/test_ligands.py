#%%
# import TM_catalyst_framework
from TM_catalyst_framework import Ligand

ureate = Ligand("ureate", "C(=O)N", denticity=2, donor_atoms=["O", "N"])

# Generate 3D structure
ureate.generate_3D()

# Optimize using Open Babel
ureate.optimize_openbabel()

# Export files

ureate.save_xyz("ureate_opt.xyz")
ureate.save_mol("ureate_opt.mol")



# %%
