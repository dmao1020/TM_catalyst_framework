import openbabel.openbabel as ob
from openbabel import pybel
import sys, os
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import defaultdict
import numpy as np
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
            ff_convergence: float = 1.0e-4,
            metal_center: str = "Ti"
        ):
        """_summary_

        Args:
            mol (_type_): _description_
        """
        self.force_field = force_field
        self.force_field_steps = force_field_steps
        self.ff_convergence = ff_convergence
        self.metal_center = metal_center


    def organic_opt(self, 
                    smiles: str = "CC"):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol)
        ff = AllChem.UFFGetMoleculeForceField(mol)
        ff.Initialize()

        # print (f"Initial energy: {ff.CalcEnergy():.2f} kcal/mol")
        ff.Minimize(maxIts=self.force_field_steps)
        print (f"Final ligand energy: {ff.CalcEnergy():.2f} kcal/mol")
        self.ff_energy = ff.CalcEnergy()
        # Convert to pybel mol for consistency
        mol_sdf_block = Chem.MolToMolBlock(mol)
        obmol = pybel.readstring("mol", mol_sdf_block)
        return obmol
        

    def run_opt(self, 
                mol,
                constraints = None
                ):
        
        self.ff_energy = None
        
        # === Suppress logs ===
        cerr_fd = sys.stderr.fileno()
        devnull = os.open('/dev/null', os.O_WRONLY)
        original_cerr = os.dup(cerr_fd)
        os.dup2(devnull, cerr_fd)
        print ("Running constrained optimization...")
        try:
            ff = ob.OBForceField.FindForceField(self.force_field)
            if not ff.Setup(mol.OBMol, constraints):
                raise ValueError("Setup failed")
            ff.SetConstraints(constraints)
            # print(f"Initial energy: {ff.Energy():.2f} kcal/mol")
            # ff.ConjugateGradients(force_field_steps)
            ff.ConjugateGradients(self.force_field_steps) 
            # print(f"Final energy: {ff.Energy():.2f} kcal/mol")
            self.ff_energy = ff.Energy()
            
            ff.GetCoordinates(mol.OBMol)
        finally:
            os.dup2(original_cerr, cerr_fd)
            os.close(original_cerr)
            os.close(devnull)
    
    def free_opt(self, mol):
        self.run_opt(mol, None)
    # constraint_optimization
    def constraint_metal_opt(self,
                             mol, 
                             constraints = ob.OBFFConstraints()):
        # mol = self.complex
        # print (mol)
        if constraints != None:
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
    
    # constraint_optimization
    def constraint_metal_bangle_opt(self,
                                    mol, 
                                    constraints = ob.OBFFConstraints()):
        # mol = self.complex
        # print (mol)
        constraints.Clear()
        metal_atomic_num = element_dict[self.metal_center]
        metal_nbr_atom_ls = []
        for atom in mol:
            if atom.OBAtom.GetAtomicNum() == metal_atomic_num:  # * = dummy, 9 = fluorine
                metal_atom_idx = atom.idx + 1
                print (f"Metal atom idx: {metal_atom_idx}")
        angle = None
        for angle_itr in ob.OBMolAngleIter(mol.OBMol):
            b = (angle_itr[1] + 1)
            # print (f"b atom idx in angle: {b}")
            if b == metal_atom_idx:
                a = (angle_itr[0] + 1)
                b_atom = mol.OBMol.GetAtom(b)
                c = (angle_itr[2] + 1)
                angle = b_atom.GetAngle(a, c)
                print (f"Add angle constraint on {angle_itr} involving metal center: {angle}")
                constraints.AddAngleConstraint(a, b, c, angle)
            # else:
                # continue
                # print ("Angle does NOT involve metal center.")
                # print (f"Angle atoms idx: {angle_itr}")
        print (constraints)
        self.run_opt(mol, constraints)

        return mol, constraints
    
    import sys

    def fix_overlapping_atoms(self, 
                            filename: str = "molecule.sdf", 
                            delta: float = 0.1,
                            run_opt: bool = True
                            ):
        with open(filename, "r") as f:
            lines = [line.rstrip("\n") for line in f.readlines()]

        # Find the counts line (V2000 or V3000)
        counts_idx = None
        for i, line in enumerate(lines):
            if "V2000" in line or "V3000" in line:
                counts_idx = i
                break
        if counts_idx is None:
            raise ValueError("SDF file does not contain V2000 or V3000 block")

        counts_parts = lines[counts_idx].split()
        
        atom_start = counts_idx + 1
        atom_lines = lines[atom_start:]

        # Parse atom coordinates and element
        atoms = []
        natoms = 0
        for atom_num, line in enumerate(atom_lines, start=1):
            
            parts = line.split()
            if len(parts)==16:
                natoms += 1
                # print (parts)e
                if len(parts) < 4:
                    raise ValueError(f"Invalid atom line for atom {atom_num}")
                x = float(parts[0])
                y = float(parts[1])
                z = float(parts[2])
                element = parts[3]
                atoms.append({
                    "x": x,
                    "y": y,
                    "z": z,
                    "element": element,
                    "atom_num": atom_num
                })

        # Identify overlapping atoms (exact coordinates after rounding to avoid float issues)
        pos_to_atoms = defaultdict(list)
        for atom in atoms:
            key = (round(atom["x"], 9), round(atom["y"], 9), round(atom["z"], 9))
            pos_to_atoms[key].append(atom)

        # Perturb duplicates (keep the lowest-numbered atom unchanged, perturb others along +z)
        perturbed = []
        
        for group in pos_to_atoms.values():
            if len(group) > 1:
                group.sort(key=lambda a: a["atom_num"])  # process in atom order
                for i in range(1, len(group)):
                    delta_i = round(np.random.uniform(low=-delta, high=delta), 3)
                    atom = group[i]
                    old_z = atom["z"]
                    atom["z"] += delta_i * i
                    perturbed.append((atom["atom_num"], atom["element"], old_z, atom["z"]))

        if not perturbed:
            print("No overlapping atoms found.")
            return

        print("Overlapping atoms detected and perturbed:")
        for num, el, old_z, new_z in perturbed:
            print(f"  Atom {num} ({el}): z = {old_z:.4f} → {new_z:.4f} Å")

        # Rebuild atom block in standard fixed-width V2000 format
        new_atom_lines = []
        for atom in atoms:
            line = (
                f"{atom['x']:10.4f}"
                f"{atom['y']:10.4f}"
                f"{atom['z']:10.4f}"
                f"  {atom['element']:<2}"
                f" 0 0 0 0 0 0 0 0 0 0 0 0"
            )
            new_atom_lines.append(line)

        # Replace the old atom block
        lines[atom_start:atom_start + natoms] = new_atom_lines

        if run_opt:
            print("\nRunning geometry optimization to relax perturbed structure...")
            # Read with pybel
            mol = pybel.readstring("sdf", "\n".join(lines))
            self.constraint_metal_opt(mol)
            # Write back optimized structure
            mol.write("sdf", filename, overwrite=True)
            print("Geometry optimization complete.")

        else:
            # Overwrite the original file
            with open(filename, "w") as f:
                for line in lines:
                    f.write(line + "\n")

            print(f"\nFixed SDF file has been overwritten: {filename}")
            print("(The three duplicate hydrogens now have slightly different z-coordinates.)")

