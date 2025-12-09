import openbabel.openbabel as ob
from openbabel import pybel
import sys, os

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

class GeoOpt:
    def __init__(
            self, 
            force_field: str = "uff", #mmff94, uff
            force_field_steps: int = 2000,
            metal_center: str = "Ti"
        ):
        """_summary_

        Args:
            mol (_type_): _description_
        """
        self.force_field = force_field
        self.force_field_steps = force_field_steps
        self.metal_center = metal_center

    def run_opt(self, 
                mol,
                constraints = ob.OBFFConstraints()
                ):
        # === Suppress logs ===
        cerr_fd = sys.stderr.fileno()
        devnull = os.open('/dev/null', os.O_WRONLY)
        original_cerr = os.dup(cerr_fd)
        os.dup2(devnull, cerr_fd)

        try:
            ff = ob.OBForceField.FindForceField(self.force_field)
            if not ff.Setup(mol.OBMol, constraints):
                raise ValueError("Setup failed")
            
            print(f"Initial energy: {ff.Energy():.2f} kcal/mol")
            # ff.ConjugateGradients(force_field_steps)
            ff.ConjugateGradients(self.force_field_steps) 
            print(f"Final energy: {ff.Energy():.2f} kcal/mol")
            
            ff.GetCoordinates(mol.OBMol)
        finally:
            os.dup2(original_cerr, cerr_fd)
            os.close(original_cerr)
            os.close(devnull)
    # constraint_optimization
    def constraint_metal_opt(self,
                             mol, 
                             constraints = ob.OBFFConstraints()):
        # mol = self.complex
        # print (mol)
        constraints.Clear()
        metal_atomic_num = element_dict[self.metal_center]
        metal_nbr_atom_ls = []
        for atom in mol:
            if atom.OBAtom.GetAtomicNum() == metal_atomic_num:  # * = dummy, 9 = fluorine
                metal_atom = atom
                # Find neighbor (the atom that was bonded to *)
                for nbr in pybel.ob.OBAtomAtomIter(atom.OBAtom):
                    nbr_atom = mol.atoms[nbr.GetIdx() - 1]
                    nbr_idx = nbr_atom.idx
                    # print (f"Add atom constraint to atom with nbr_idx: {nbr_idx}")
                    constraints.AddAtomConstraint(nbr_idx)
        self.run_opt(mol, constraints)
        return mol, constraints
