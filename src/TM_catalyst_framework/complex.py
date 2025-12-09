from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
import openbabel
from pathlib import Path
from typing import List, Optional
from TM_catalyst_framework.ligands import Ligand
import numpy as np
from openbabel import pybel
from importlib import resources
import openbabel.openbabel as ob
import os
import sys

from TM_catalyst_framework.ligands import *
from TM_catalyst_framework.metal_template import Template#load_ti_template_sdf
from TM_catalyst_framework.GeoOpt import GeoOpt#load_ti_template_sdf

d_electrons_dict = {
            "Sc": 1, "Ti": 2, "V": 3, "Cr": 5, "Mn": 5,
            "Fe": 6, "Co": 7, "Ni": 8, "Cu": 10, "Zn": 10,
            "Zr": 2, "Nb": 3, "Mo": 6, "Hf": 2, "Ta": 3, "W": 6
        }

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
class Mapping:
    def __init__(self,
                 mapping_dict: dict = {}
                 ):
        self.mapping_dict = mapping_dict
    
    def add_monodentate(self,
                   ligand_SMILES: str = "*N(C)C",
                   atom: str = "N",
                   coordination_site: int = 6
                   ):
        if not self.mapping_dict:
            self.mapping_dict[1] = [ligand_SMILES, coordination_site]
        else:
            map_max_keys = max(list(self.mapping_dict.keys()))
            self.mapping_dict[map_max_keys+1] = [ligand_SMILES, coordination_site]
    
    def add_bidentate(self,
                   ligand_SMILES: str = "C/N=C([N](C)(C)[*])/O[*]",
                #    atom1: str = "O",
                   coordination_site_1: int = 2,
                #    atom2: str = "N",
                   coordination_site_2: int = 3
                   ):
        if not self.mapping_dict:
            self.mapping_dict[1] = [ligand_SMILES,
                                    [coordination_site_1,
                                    coordination_site_2]
                                    ]
        else:
            map_max_keys = max(list(self.mapping_dict.keys()))
            self.mapping_dict[map_max_keys+1] = [ligand_SMILES,
                                                 [coordination_site_1,
                                                 coordination_site_2]
                                                ]

class Complex:
    def __init__(self,
                 metal_center: str,
                 force_field: str = "uff",#uff <-- more stable, other options: mmff94
                 force_field_steps: int = 2000,
                 noble_map: dict = {
                    1:2,# helium at position 1
                    2:10, # neon at position 2
                    3:18, # Argon at position 3
                    4:36, # Kr at position 4
                    5:54, # Xe at position 5
                    6:86 # Rn at position 6
                }
                #  oxidation_state: int,
                #  ligands: list[Ligand],
                #  geometry: Optional[str] = None,
                #  template_name: str = "u1_d1_iso1.xyz"
                 ):
        """Parameters

        Args:
            metal_center (str): 
                Symbol of the metal center (e.g., "Ti").
            oxidation_state (int): 
                Formal oxidaton state of the metal center (postive integer).
            ligands (list[Ligand]): 
                List of Ligand objects to be coordinated to the metal center.
            geometry (Optional[str], optional): 
                Geometry type (e.g., 'tetrahedral', 'square_pyramidal, 'octahedral'). Defaults to None.
        """
        self.metal_center = metal_center
        self.force_field = force_field
        self.force_field_steps = force_field_steps
        self.noble_map= noble_map
        self.charge = None
        self.multiplicity = None
        self.mol3D = None # Placeholder for an RDKit Mol object
        self.xyz = None # XYZ string representation
    
    def find_dummy_and_anchor(self, 
                              template, 
                              dummy_atomic_num=9):
        dummy_atom = None
        anchor_atom = None
        for atom in template:
            # * = dummy on ligand SMILES, or any atomic number on metal template
            if atom.OBAtom.GetAtomicNum() == dummy_atomic_num:
                dummy_atom = atom
                # Find neighbor (the atom that was bonded to *)
                for nbr in pybel.ob.OBAtomAtomIter(atom.OBAtom):
                    anchor_atom = template.atoms[nbr.GetIdx() - 1]  # 0-based Pybel indexing
                break
        if dummy_atom is None:
            raise ValueError(f"Cannot find coordination site associated with the placeholder atom {dummy_atomic_num} in template, please choose another coordination site!")
        else:
            return dummy_atom, anchor_atom

    def find_fragment_attachment_point(self, frag):
        attach_atom = None
        for atom in frag:
            if atom.OBAtom.GetAtomicNum() == 0:  # *
                attach_atom = atom
                break
        if attach_atom is None:
            raise ValueError("No * in fragment")
        
        # Get the real atom atom after *
        connect_atom = None
        for nbr in pybel.ob.OBAtomAtomIter(attach_atom.OBAtom):
            connect_atom = nbr
            break
        if connect_atom is None:
            raise ValueError("Fragment * has no neighbor")
        
        return attach_atom, connect_atom

    
    def attach_monodentate_ligand(self,
                                  mol,
                                  ligand_smiles: str = "*N(C)C",
                                  coord_site: int = 2,
                                  halogen_mapping = True,
                                  ):
        GeoOpt_util = GeoOpt(force_field = self.force_field,
                             force_field_steps = self.force_field_steps,
                             metal_center = self.metal_center)
        if halogen_mapping == True:
            dummy_atomic_num = self.noble_map[coord_site]
        else:
            dummy_atomic_num = coord_site
        self.complex = mol
        # self.constraint_optimization()
        ligand = pybel.readstring("smi", ligand_smiles)
        ligand.make3D()

        # count nunmber of dummy 
        dummy_count = 0
        for atom in mol:
            if atom.OBAtom.GetAtomicNum() == dummy_atomic_num: 
                dummy_count += 1
        if dummy_count != 0:
            for dummy_i in range(dummy_count):
                # Merge fragment into template
                mol.OBMol += ligand.OBMol
                
                # Find dummy atom representing oxygen on complex template
                dummy_atom, anchor_atom = self.find_dummy_and_anchor(mol, dummy_atomic_num)

                ######## Ligand ########
                # find dummy
                attach_atom = None
                connect_atom = None
                for atom in ligand:
                    # find dummy atom with label *, which has atomic number == 0
                    if atom.OBAtom.GetAtomicNum() == 0:
                        # find the atom that is bonded to the dummy atom
                        for nbr in pybel.ob.OBAtomAtomIter(atom.OBAtom):
                            break
                        if nbr is None:
                            raise ValueError("Fragment * has no neighbor")
                    attach_atom = atom
                    connect_atom = nbr
                    break
                if attach_atom is None:
                    raise ValueError("No * in fragment")
                if connect_atom is None:
                    raise ValueError("Fragment * has no neighbor")  
                mol = self.add_ligand_bond(
                                mol,
                                ligand,
                                connect_atom.GetIdx(),
                                anchor_atom.idx,
                                dummy_atom,
                                )
                
                for atom in mol:
                    # print (f"New molecule Atom idx: {atom.idx}, atomic num: {atom.OBAtom.GetAtomicNum()}")
                    if atom.OBAtom.GetAtomicNum() == 0:  # *
                        # print (f"Found leftover * at index {atom.idx}, marking for deletion")
                        mol.OBMol.DeleteAtom(atom.OBAtom)
                print (f"Fragment attachment at index {attach_atom.idx}")
                # === Freeze atoms attaching the metal center ===
                constraints = ob.OBFFConstraints()
                mol, constraints = GeoOpt_util.constraint_metal_opt(mol, constraints)
            # --------------------------------------------------
            # Final geometry optimization with FF (no constraints)
            # --------------------------------------------------
            # self.run_opt(mol, constraints)
            self.complex = mol
            return mol

        else:
            raise ValueError("No dummy found!")

    def attach_bidentate_ligand(self,
                                mol,
                                ligand_smiles: str = "CC(C1=CC=CC=C1)N(/C(N(C2=C(C=CC=C2C)C)[*])=O\[*])C",
                                coordination_info: list = [2, 3],
                                ):
        GeoOpt_util = GeoOpt(force_field = self.force_field,
                             force_field_steps = self.force_field_steps,
                             metal_center = self.metal_center)
        # Merge fragment into template
        print (f"attach_bidentate_ligand function, ligand_smiles = {ligand_smiles}")
        ligand_util = Ligand(smiles = ligand_smiles)
        ligand = ligand_util.ligand
        mol.OBMol += ligand.OBMol

        # Find dummy atom representing the first dummy atom on complex template
        # print ("dummy_atomic_num_ls[0]:",dummy_atomic_num_ls[0])
        dummy1_atom, anchor_atom = self.find_dummy_and_anchor(mol, self.noble_map[coordination_info[0]])
        
        # print ("anchor_atom:",anchor_atom)
        
        # Find dummy atom representing the second dummy atom on complex template
        dummy2_atom, anchor_atom = self.find_dummy_and_anchor(mol, self.noble_map[coordination_info[1]])
        anchor_atom_idx = anchor_atom.OBAtom.GetIdx()
        ######## Ligand ########
        
        # find dummy atom 1
        attach_dummy1, connect_dummy1 = ligand_util.find_ligand_attachement_point(donor_atoms_idx = 1)#(donor_atoms=dummy_element_ls[0])
        connect_dummy1_idx = connect_dummy1.GetIdx()  # 1-based

        # find dummy atom 2
        attach_dummy2, connect_dummy2 = ligand_util.find_ligand_attachement_point(donor_atoms_idx = 2)#(donor_atoms=dummy_element_ls[1])
        connect_dummy2_idx = connect_dummy2.GetIdx()  # 1-based

        # add the first bond between the ligand and template
        mol = self.add_ligand_bond(mol, 
                                   ligand,
                                   connect_dummy1_idx, 
                                   anchor_atom_idx, 
                                   dummy1_atom)
        # add the second bond between the ligand and template
        mol = self.add_ligand_bond(mol, 
                                   ligand,
                                   connect_dummy2_idx, 
                                   anchor_atom_idx, 
                                   dummy2_atom)
        
        
        for atom in mol:
            # print (f"New molecule Atom idx: {atom.idx}, atomic num: {atom.OBAtom.GetAtomicNum()}")
            if atom.OBAtom.GetAtomicNum() == 0:  # *
                # print (f"Found leftover * at index {atom.idx}, marking for deletion")
                mol.OBMol.DeleteAtom(atom.OBAtom)
        constraints = ob.OBFFConstraints()
        mol, constraints = GeoOpt_util.constraint_metal_opt(
                                                mol, 
                                                constraints)
        self.complex = mol
        
        return mol, constraints


    
    def add_ligands_wt_geometry(self,
                    geometry: str = "octahedral",
                    output_filename: str = "output_test",
                    output_format: str = "sdf",
                    template_dir: str = "TM_catalyst_framework.template",
                    metal_coord_dummy_map: dict = {1: ["C/N=C([N](C)(C)[*])/O[*]",{"O":2, "N":3}],
                                                   2: ["C/N=C([N](C)(C)[*])/O[*]",{"O":4, "N":5}],
                                                   3: ["*N(C)C", {"O":1}],
                                                   4: ["*N(C)C",{"O":6}]
                                                   }
                    ):
        # self.force_field = force_field
        # self.force_field_steps = force_field_steps
        """Build 3D complex geometry by adding ligand to the metal center.

        Args:
            template_sdf (str): 
                Path to the template SDF file with dummy atoms.
            template_dir (str): 
                directory where the template SDF file is saved
            ligand_smiles (str):
                ureate ligand smiles with dummy atoms * attaching to O, N1, and N2
            output_sdf (str): 
                Path to save the output SDF file.
            force_field (str, optional): 
                Force field for geometry optimization ("uff" or "mmff94"). Defaults to "mmff94".
            force_field_steps (int, optional): 
                Number of optimization steps. Defaults to 2000.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        
        # --------------------------------------------------------------
        # 1. Prepare coordination
        # 1.1 Load the template (keep hydrogens if you need them)
        # --------------------------------------------------------------
        template_sdf = f"{geometry}.sdf" 
        template_util = Template(
            template_dir = template_dir,
            package = "openbabel",
            metal_center = self.metal_center
        )
        template = template_util.load_ti_template_sdf(
            template_name = template_sdf)
        if template is None:
            raise ValueError("Could not read the template SDF file.")
        
        # template.make3D(forcefield=force_field, steps=force_field_steps)
        skip_count = 0
        dummy_atom_dict = {}

        # --------------------------------------------------
        # 1.2. set up geometry optimization utils
        # --------------------------------------------------
        GeoOpt_util = GeoOpt(force_field = self.force_field,
                             force_field_steps = self.force_field_steps,
                             metal_center = self.metal_center)
        constraints = ob.OBFFConstraints()
        # --------------------------------------------------
        # 2. Coordinate ligands
        # --------------------------------------------------
        for ligand_i, ligand_info in metal_coord_dummy_map.items():
            ligand_smiles, coordination_info = ligand_info
            # print ("ligand_smiles:",ligand_smiles)
            # print ("coordination_dict:",coordination_dict)
            # check denticity
            print ("coordination_info:", coordination_info)
            if isinstance(coordination_info, list):
                denticity = len(coordination_info)
            else:
                denticity = 1
            print ("denticity:",denticity)

            if denticity == 1:#monodentate ligand
                coord_site = coordination_info
                print ("coord_site:",coord_site)
                template = self.attach_monodentate_ligand(
                                        template,
                                         ligand_smiles = ligand_smiles,
                                         coord_site =coord_site, 
                                         halogen_mapping = True
                                         )
            elif denticity == 2: # bidentate ligand
                # print ("denticity:", denticity)
                # print ("coordination_dict:", coordination_dict)
                template, constraints = self.attach_bidentate_ligand(
                                            template,
                                            ligand_smiles,
                                            coordination_info,
                                         )
        for atom in template:
            # print (f"New molecule Atom idx: {atom.idx}, atomic num: {atom.OBAtom.GetAtomicNum()}")
            if atom.OBAtom.GetAtomicNum() == 0:  # *
                # print (f"Found leftover * at index {atom.idx}, marking for deletion")
                template.OBMol.DeleteAtom(atom.OBAtom)
        template, constraints = GeoOpt_util.constraint_metal_opt(template, constraints)
                # template = self.complex
        # --------------------------------------------------
        # 3. Final geometry optimization with FF (no constraints)
        # --------------------------------------------------
        constraints.Clear()
        GeoOpt_util.run_opt(template, constraints)
        # --------------------------------------------------
        # 4. Save
        # --------------------------------------------------
        if output_format == "sdf" or "SDF":
            template.write("sdf", f"{output_filename}.sdf", overwrite=True)
        elif output_format == "xyz" or "XYZ":
            template.write("xyz", f"{output_filename}.xyz", overwrite=True)
        else:
            raise ValueError("Unsupported output format, please choose either 'sdf' or 'xyz'")
        self.complex = template
    
        if skip_count == 4:
            raise ValueError("No ligand is attached to the template because 0 dummy atoms were found.")
    def add_fragments(self,
                      template_sdf: str,
                      fragment_smiles_dict: dict,
                      N1N2_dict: dict,
                      output_sdf: str,
                      template_dir:str = "TM_catalyst_framework.metal_center_template",
                      force_field: str = "uff", # "uff" or "mmff94", for geometry optimization
                      force_field_steps: int = 2000, # Number of optimization steps
                      ):
        self.force_field = force_field
        self.force_field_steps = force_field_steps
        """
        Build 3D complex geometry by adding substituents
        on the ligands coordinated to the metal center.
        1. Identify dummy atoms in template and fragments.
        2. Remove dummy atoms and merge fragments into template.
        3. Add bonds between metal center and fragment attachment points.
       
        There are three substituents (R1, R2, R3) on the ureate ligand
        The placement of the substituent fragments 
        is determined by the keys in fragment_smiles_dict. 

        To identify which dummy atom corresponds to which substituent,
        we place dummy halogen atoms on the metal template:
        R1: Fluorine (F, atomic num 9)
        R2: Chlorine (Cl, atomic num 17)
        R3: Bromine (Br, atomic num 35)

        Thus, the fragment_smiles_dict is in the form of:
        {"R1": "SMILES_string_for_R1,
         "R2": "SMILES_string_for_R2,
         "R3": "SMILES_string_for_R3
        } 
        and the code will replace the corresponding
        dummy atoms with the fragments using the substituent_dummy_map.


        Args:
            template_sdf (str): 
                Path to the template SDF file with dummy atoms.
            fragment_smiles_dict (dict): 
                Dictionary mapping dummy atom symbols to fragment SMILES strings.
            output_sdf (str): 
                Path to save the output SDF file.
            force_field (str, optional): 
                Force field for geometry optimization ("uff" or "mmff94"). Defaults to "mmff94".
            force_field_steps (int, optional): 
                Number of optimization steps. Defaults to 2000.
          """
        # --------------------------------------------------------------
        # 1. Load the template (keep hydrogens if you need them)
        # --------------------------------------------------------------
        
        # template = load_ti_template_sdf(template_sdf, package = "openbabel",
        #                                 template_dir = "TM_catalyst_framework.template")
        template = Template.load_ti_template_sdf(
                            template_name = template_sdf, 
                            template_dir = template_dir,
                            package = "openbabel",)
        if template is None:
            raise ValueError("Could not read the template SDF file.")
        # template.make3D()

        constraints = ob.OBFFConstraints()

        for Ni, dummy_atomic_num_ls in N1N2_dict.items():
            for idx, dummy_atomic_num in enumerate(dummy_atomic_num_ls):
                if Ni == "N1": # nitrogen with R1
                    # --------------------------------------------------
                    # 2. Replace dummy atom with substituent fragment
                    # --------------------------------------------------
                    # Find dummy and anchor in template
                    dummy_atom, anchor_atom = self.find_dummy_and_anchor(template, dummy_atomic_num)
                    dummy_atom_idx = dummy_atom.idx
                    anchor_idx = anchor_atom.idx
                    # --------------------------------------------------
                    # 3. Load fragment and find attachment point
                    # --------------------------------------------------
                    # Load fragment and find attachment point
                    frag = pybel.readstring("smi", fragment_smiles_dict["R1"])
                    frag.make3D()
                    attach_atom, connect_atom = self.find_fragment_attachment_point(frag)
                    connect_idx = connect_atom.GetIdx()  # 1-based

                    # --------------------------------------------------
                    # 4. Delete dummy, merge, and ADD BOND
                    # --------------------------------------------------
                    # Replace dummy atom to nitrogen
                    print(f"dummy_atom atomic number: {dummy_atom.OBAtom.GetAtomicNum()}")
                    #TODO unhash -->dummy_atom.OBAtom.SetAtomicNum(7)

                    # Merge fragment into template
                    template.OBMol += frag.OBMol

                    # Now add bond: anchor (in template) → connect_atom (in fragment)
                    # After merge, fragment atoms start at: mol.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
                    frag_start_idx = template.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
                    connect_idx_in_merged = frag_start_idx + connect_idx - 1

                    # Add bond (1-based indices)
                    template.OBMol.AddBond(dummy_atom_idx, connect_idx_in_merged, 1)  # 1 = single bond

                    #cleanup: check if there's any remaining * atoms and remove them
                    delete_indices = []
                    for atom in template:
                        # print (f"New molecule Atom idx: {atom.idx}, atomic num: {atom.OBAtom.GetAtomicNum()}")
                        if atom.OBAtom.GetAtomicNum() == 0:  # *
                            # print (f"Found leftover * at index {atom.idx}, marking for deletion")
                            template.OBMol.DeleteAtom(atom.OBAtom)
                    print (f"Fragment attachment at index {attach_atom.idx}")
                    # === Freeze atoms attaching the metal center ===
                    
                    constraints.Clear()
                    constraints, template = self.constraint_optimization(constraints, template)
                
                if Ni == "N2": # nitrogen with R2 and R3
                    for r_idx in range(2, 4):
                        # --------------------------------------------------
                        # 2. Replace dummy atom with substituent fragment
                        # --------------------------------------------------
                        # Find dummy and anchor in template
                        dummy_atom, anchor_atom = self.find_dummy_and_anchor(template, dummy_atomic_num)
                        dummy_atom_idx = dummy_atom.idx
                        anchor_idx = anchor_atom.idx

                        frag = pybel.readstring("smi", fragment_smiles_dict["R%s"%r_idx])
                        frag.make3D()
                        attach_atom, connect_atom = self.find_fragment_attachment_point(frag)
                        connect_idx = connect_atom.GetIdx()  # 1-based

                        # --------------------------------------------------
                        # 4. Delete dummy, merge, and ADD BOND
                        # --------------------------------------------------
                        # Merge fragment into template
                        template.OBMol += frag.OBMol

                        # Now add bond: anchor (in template) → connect_atom (in fragment)
                        # After merge, fragment atoms start at: mol.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
                        frag_start_idx = template.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
                        connect_idx_in_merged = frag_start_idx + connect_idx - 1

                        # Add bond (1-based indices)
                        template.OBMol.AddBond(dummy_atom_idx, connect_idx_in_merged, 1)  # 1 = single bond

                        #cleanup: check if there's any remaining * atoms and remove them
                        delete_indices = []
                        for atom in template:
                            # print (f"New molecule Atom idx: {atom.idx}, atomic num: {atom.OBAtom.GetAtomicNum()}")
                            if atom.OBAtom.GetAtomicNum() == 0:  # *
                                # print (f"Found leftover * at index {atom.idx}, marking for deletion")
                                template.OBMol.DeleteAtom(atom.OBAtom)
                        print (f"Fragment attachment at index {attach_atom.idx}")

                        # === Freeze atoms attaching the metal center ===
                        constraints.Clear()
                        constraints, template = self.constraint_optimization(constraints, template)

        # --------------------------------------------------
        # Replace dummy atoms to nitrogen
        # --------------------------------------------------
        for Ni, dummy_atomic_num_ls in N1N2_dict.items():
            for dummy_atomic_num_i in dummy_atomic_num_ls:
                dummy_atom, anchor_atom = self.find_dummy_and_anchor(template, dummy_atomic_num_i)
                dummy_atom.OBAtom.SetAtomicNum(7)
                    
        # --------------------------------------------------
        # 5. Generate 3D and optimize
        # --------------------------------------------------
        # === Freeze atoms attaching the metal center ===
        constraints.Clear()

        metal_atomic_num = 22# d_electrons_dict[self.metal_center]
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

        # === Suppress logs ===
        cerr_fd = sys.stderr.fileno()
        devnull = os.open('/dev/null', os.O_WRONLY)
        original_cerr = os.dup(cerr_fd)
        os.dup2(devnull, cerr_fd)

        try:
            ff = ob.OBForceField.FindForceField(force_field)
            if not ff.Setup(template.OBMol, constraints):
                raise ValueError("Setup failed")
            
            print(f"Initial energy: {ff.Energy():.2f} kcal/mol")
            # ff.ConjugateGradients(force_field_steps)
            ff.ConjugateGradients(force_field_steps) 
            print(f"Final energy: {ff.Energy():.2f} kcal/mol")
            
            ff.GetCoordinates(template.OBMol)
        finally:
            os.dup2(original_cerr, cerr_fd)
            os.close(original_cerr)
            os.close(devnull)
        
        # Final geometry optimization with FF without the constraints
        # constraints.Clear()
        # template.make3D(forcefield=force_field, steps=force_field_steps)
        # --------------------------------------------------
        # 6. Save
        # --------------------------------------------------
        # template.make3D(force_field, force_field_steps)
        template.write("sdf", output_sdf, overwrite=True)
            
        print(f"Success! Saved to {output_sdf}")

    def add_ligand_bond(self,
                        mol,
                        ligand,
                        connect_idx,
                        anchor_idx,
                        dummy_atom,
                        ):
        dummy_coord = dummy_atom.OBAtom.GetVector()
        mol.OBMol.DeleteAtom(dummy_atom.OBAtom)
        
        # Now add bond: anchor (in template) → connect_atom (in fragment)
        # After merge, fragment atoms start at: mol.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
        frag_start_idx = mol.OBMol.NumAtoms() - ligand.OBMol.NumAtoms() + 1

        # connect atom index
        connect_idx_in_merged = frag_start_idx + connect_idx - 1
        # Add M-O bond (1-based indices)
        mol.OBMol.AddBond(anchor_idx, connect_idx_in_merged, 1)  # 1 = single bond
        connect_atom = mol.OBMol.GetAtom(connect_idx_in_merged)
        connect_atom.SetVector(dummy_coord)
        return mol
    def add_ligands(self,
                    template_sdf: str,
                    ligand_smiles: str,
                    template_dir: str = "TM_catalyst_framework.template",
                    output_sdf: str = "output.sdf",
                    force_field: str = "mmff94",
                    force_field_steps: int = 2000,
                    connect_N_dict: dict = {35: "N1", 53: "N1"},
                    ):
        self.force_field = force_field
        self.force_field_steps = force_field_steps
        """Build 3D complex geometry by adding ligand to the metal center.

        Args:
            template_sdf (str): 
                Path to the template SDF file with dummy atoms.
            ligand_smiles (str):
                ureate ligand smiles with dummy atoms * attaching to O, N1, and N2
            output_sdf (str): 
                Path to save the output SDF file.
            force_field (str, optional): 
                Force field for geometry optimization ("uff" or "mmff94"). Defaults to "mmff94".
            force_field_steps (int, optional): 
                Number of optimization steps. Defaults to 2000.

        Raises:
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        self.connect_N_dict = connect_N_dict
        # Dictionary that maps the dummy atoms representing 
        # the ligand-coordination postion in the template
        # metal_coord_dummy_map = {
        #     "O1": 9, # Fluorine, as the first oxygen
        #     "O2": 17, # Chlorine, as the second oxygen
        #     "N1": 35, # Bromine, as the first nitrogen
        #     "N2": 53 # Iodine, as the second nitrogen   
        # }

        metal_coord_dummy_map = {
            "O1": 9, # Fluorine, as the first oxygen
            "O2": 17, # Chlorine, as the second oxygen
            "N1": 10, # Bromine, as the first nitrogen
            "N2": 18 # Iodine, as the second nitrogen   
        }

        d2d2_angle_map = {"u2_d2d2_iso7":{"N":180, "O": 90}} 

        # --------------------------------------------------------------
        # 1. Load the template (keep hydrogens if you need them)
        # --------------------------------------------------------------
        template_util = Template(
            template_dir = template_dir,
            package = "openbabel",
        )
        template = template_util.load_ti_template_sdf(
                            template_name = template_sdf)
        if template is None:
            raise ValueError("Could not read the template SDF file.")
        
        # template.make3D(forcefield=force_field, steps=force_field_steps)
        skip_count = 0
        ligand_util = Ligand(smiles = ligand_smiles)
        ligand = ligand_util.ligand
        self.ligand = ligand
        dummy_atom_dict = {}

        # --------------------------------------------------
        # 2. Coordinate 1 or 2 ligands
        # --------------------------------------------------
        if "d2d2" in template_sdf:
            constraints = ob.OBFFConstraints()
            # --------------------------------------------------
            # 2a.1. Add the first ligand
            # --------------------------------------------------
            dummyO_atomic_num, dummyN_atomic_num = metal_coord_dummy_map["O1"], metal_coord_dummy_map["N1"]
            constraints, template = self.attach_bidentate_ligand(constraints, 
                                        template,
                                        dummyO_atomic_num, 
                                        dummyN_atomic_num
                                        )

            # --------------------------------------------------
            # 2a.2. Add the second ligand
            # --------------------------------------------------
            # Merge fragment into template
            dummyO_atomic_num, dummyN_atomic_num = metal_coord_dummy_map["O2"], metal_coord_dummy_map["N2"]
            constraints, template = self.attach_bidentate_ligand(constraints, 
                                        template,
                                        dummyO_atomic_num, 
                                        dummyN_atomic_num
                                        )
        else:
            # template.make3D(forcefield=force_field, steps=force_field_steps)
            # constraints = ob.OBFFConstraints()
            # --------------------------------------------------
            # 2b. Check if template contains dummy atom
            # --------------------------------------------------
            for dummy_atom_label, dummy_atomic_num in metal_coord_dummy_map.items():
                dummy_atom, anchor_atom = self.find_dummy_and_anchor(template, dummy_atomic_num)
                # The template doesn't contain the dummy atom, skip and find the next one
                if dummy_atom == None:
                    skip_count += 1
                    continue 
                anchor_idx = anchor_atom.idx # 1-based Open Babel index
                dummy_atom_dict[dummy_atom_label] = [dummy_atom, anchor_atom, anchor_idx]
            print (f"Dummy atoms found from template: {dummy_atom_dict.keys()}")
            constraints = ob.OBFFConstraints()
            # constraints, template = self.constraint_optimization(constraints, template)
            for i in range(1,3):
                if f"O{i}" in dummy_atom_dict.keys():
                    if f"N{i}" in dummy_atom_dict.keys():
                        # --------------------------------------------------
                        # 2b.1. Coordinate bidentate ligand
                        # --------------------------------------------------
                        print (f"Monodentate ligand N{i}O{i}")
                        dummyO_atomic_num = metal_coord_dummy_map["O%s"%(i)]
                        dummyN_atomic_num = metal_coord_dummy_map["N%s"%(i)]
                        constraints, template = self.attach_bidentate_ligand(
                                                    constraints, 
                                                    template,
                                                    dummyO_atomic_num, 
                                                    dummyN_atomic_num
                                                    )
                    else:
                        # --------------------------------------------------
                        # 2b.2. Coordinate monodentate ligand
                        # --------------------------------------------------
                        dummy_O_label = "O%s"%(i)
                        print (f"Monodentate ligand N{i}O{i}")
                        # Merge fragment into template
                        template.OBMol += ligand.OBMol

                        # Find F dummy atom on complex template
                        dummy_atomic_num = metal_coord_dummy_map[dummy_O_label]
                        dummy_O_atom, anchor_atom = ligand_util.find_dummy_and_anchor(template, dummy_atomic_num)
                        
                        ######## Ligand ########
                        # find O dummy
                        attach_O, connect_O = ligand_util.find_ligand_attachement_point(dummy_O_label)
                        connect_O_idx = connect_O.GetIdx()  # 1-based

                        template = self.add_ligand_bond(template, ligand,
                                            connect_O_idx, anchor_atom.idx, dummy_O_atom)
                        
                        
                        for atom in template:
                            # print (f"New molecule Atom idx: {atom.idx}, atomic num: {atom.OBAtom.GetAtomicNum()}")
                            if atom.OBAtom.GetAtomicNum() == 0:  # *
                                # print (f"Found leftover * at index {atom.idx}, marking for deletion")
                                template.OBMol.DeleteAtom(atom.OBAtom)
                        constraints, template = self.constraint_optimization(constraints, template)
        
        # --------------------------------------------------
        # 3. Final geometry optimization with FF (no constraints)
        # --------------------------------------------------
        constraints.Clear()
        self.GeoOpt.run_opt(constraints, template)
        # --------------------------------------------------
        # 4. Save
        # --------------------------------------------------
        template.write("sdf", output_sdf, overwrite=True)
        print(f"Success! Saved to {output_sdf}")
        self.complex = template
    
        if skip_count == 4:
            raise ValueError("No ligand is attached to the template because 0 dummy atoms were found.")
        

        # if "d2d2" in template_sdf:
        #     constraints = ob.OBFFConstraints()
        #     # --------------------------------------------------
        #     # 2a.1. Add the first ligand
        #     # --------------------------------------------------
        #     dummyO_atomic_num, dummyN_atomic_num = dummy_map[metal_coord_dummy_map["O1"]], dummy_map[metal_coord_dummy_map["N1"]]
        #     constraints, template = self.attach_bidentate_ligand(constraints, 
        #                                 template,
        #                                 dummyO_atomic_num, 
        #                                 dummyN_atomic_num
        #                                 )

        #     # --------------------------------------------------
        #     # 2a.2. Add the second ligand
        #     # --------------------------------------------------
        #     # Merge fragment into template
        #     dummyO_atomic_num, dummyN_atomic_num = dummy_map[metal_coord_dummy_map["O2"]], dummy_map[metal_coord_dummy_map["N2"]]
        #     constraints, template = self.attach_bidentate_ligand(constraints, 
        #                                 template,
        #                                 dummyO_atomic_num, 
        #                                 dummyN_atomic_num
        #                                 )
        # else:
        #     # template.make3D(forcefield=force_field, steps=force_field_steps)
        #     # constraints = ob.OBFFConstraints()
        #     # --------------------------------------------------
        #     # 2b. Check if template contains dummy atom
        #     # --------------------------------------------------
        #     for dummy_atom_label, dummy_atomic_num in metal_coord_dummy_map.items():
        #         dummy_atom, anchor_atom = self.find_dummy_and_anchor(template, noble_map[dummy_atomic_num])
        #         # The template doesn't contain the dummy atom, skip and find the next one
        #         if dummy_atom == None:
        #             skip_count += 1
        #             continue 
        #         anchor_idx = anchor_atom.idx # 1-based Open Babel index
        #         dummy_atom_dict[dummy_atom_label] = [dummy_atom, anchor_atom, anchor_idx]
        #     print (f"Dummy atoms found from template: {dummy_atom_dict.keys()}")
        #     constraints = ob.OBFFConstraints()
        #     # constraints, template = self.constraint_optimization(constraints, template)
        #     for i in range(1,3):
        #         if f"O{i}" in dummy_atom_dict.keys():
        #             if f"N{i}" in dummy_atom_dict.keys():
        #                 # --------------------------------------------------
        #                 # 2b.1. Coordinate bidentate ligand
        #                 # --------------------------------------------------
        #                 print (f"Monodentate ligand N{i}O{i}")
        #                 dummyO_atomic_num = noble_map[metal_coord_dummy_map["O%s"%(i)]]
        #                 dummyN_atomic_num = noble_map[metal_coord_dummy_map["N%s"%(i)]]
        #                 constraints, template = self.attach_bidentate_ligand(
        #                                             constraints, 
        #                                             template,
        #                                             dummyO_atomic_num, 
        #                                             dummyN_atomic_num
        #                                             )
        #             else:
        #                 # --------------------------------------------------
        #                 # 2b.2. Coordinate monodentate ligand
        #                 # --------------------------------------------------
        #                 dummy_O_label = "O%s"%(i)
        #                 print (f"Monodentate ligand N{i}O{i}")
        #                 # Merge fragment into template
        #                 template.OBMol += ligand.OBMol

        #                 # Find F dummy atom on complex template
        #                 dummy_atomic_num = metal_coord_dummy_map[dummy_O_label]
        #                 dummy_O_atom, anchor_atom = self.find_dummy_and_anchor(template, noble_map[dummy_atomic_num])
                        
        #                 ######## Ligand ########
        #                 # find O dummy
        #                 attach_O, connect_O = self.find_ligand_attachement_point(ligand, 
        #                                         connect_label=dummy_O_label)
        #                 connect_O_idx = connect_O.GetIdx()  # 1-based

        #                 template = self.add_ligand_bond(template, ligand,
        #                                     connect_O_idx, anchor_atom.idx, dummy_O_atom)
                        
                        
        #                 for atom in template:
        #                     # print (f"New molecule Atom idx: {atom.idx}, atomic num: {atom.OBAtom.GetAtomicNum()}")
        #                     if atom.OBAtom.GetAtomicNum() == 0:  # *
        #                         print (f"Found leftover * at index {atom.idx}, marking for deletion")
        #                         template.OBMol.DeleteAtom(atom.OBAtom)
        #                 constraints, template = self.constraint_optimization(constraints, template)
        
        # # --------------------------------------------------
        # # 3. Final geometry optimization with FF (no constraints)
        # # --------------------------------------------------
        # constraints.Clear()
        # self.run_opt(constraints, template)
        # # --------------------------------------------------
        # # 4. Save
        # # --------------------------------------------------
        # template.write("sdf", output_sdf, overwrite=True)
        # print(f"Success! Saved to {output_sdf}")
        # self.complex = template
    
        # if skip_count == 4:
        #     raise ValueError("No ligand is attached to the template because 0 dummy atoms were found.")

    # =========================================================================
    # Electronic configuration estimation
    # =========================================================================
    
    def _estimate_charge_and_multiplicity(self):
        """Estimate total charge and spin multiplicity of the complex.
        """
        total_ligand_charge = sum(l.charge for l in self.ligands)
        self.charge = self.oxidation_state + total_ligand_charge

        # Approximate d-electron count for 3d and early 4d/5d metals
        d_electrons_dict = {
            "Sc": 1, "Ti": 2, "V": 3, "Cr": 5, "Mn": 5,
            "Fe": 6, "Co": 7, "Ni": 8, "Cu": 10, "Zn": 10,
            "Zr": 2, "Nb": 3, "Mo": 6, "Hf": 2, "Ta": 3, "W": 6
        }

        base_d = d_electrons_dict.get(self.metal_center, 0)
        d_electrons = max(base_d - self.oxidation_state, 0
                          )
        
        #Heuristic multiplicity guess
        if d_electrons == 0:
            self.multiplicity = 1 # singlet
        elif d_electrons == 1:
            self.multiplicity = 2 # doublet
        elif d_electrons in [2, 3]:
            self.multiplicity = 3 #triplet
        else: 
            self.multiplicity = 1 #fallback
        
    # =========================================================================
    # 2. Genereate 3D structure (basic mock-up)
    # =========================================================================

    

    # =========================================================================
    # 3. XYZ string export
    # =========================================================================

    def _mol_to_xyz(self):
        """ Convert RDKit Mol object to XYZ-format string."""

        if self.mol3D is None:
            raise ValueError("3D structure not built yet. Run build_geometry() first")
        
        conf = self.mol3D.GetConformer()
        n_atoms = self.mol3D.GetNumAtoms()
        lines = [f"{n_atoms}", f"{self.metal_center} complex"]

        for i in range(n_atoms):
            atom = self.mol3D.GetAtomWithIdx(i)
            pos = conf.GetAtomPosition(i)
            lines.append(f"{atom.GetSymbol()} {pos.x: 4f} {pos.y:4f}")
        return "\n".join(lines)
    
    # -------------------------------------------------------------------------
    # 4. ORCA input helper
    # -------------------------------------------------------------------------
    def get_orca_header(self):
        """Return '* xyz charge multiplicity' line for ORCA input."""
        if self.charge is None or self.multiplicity is None:
            self._estimate_charge_and_multiplicity()
        return f"* xyz {self.charge} {self.multiplicity}"

    # -------------------------------------------------------------------------
    # 5. Representation
    # -------------------------------------------------------------------------
    def __repr__(self):
        return (
            f"<Complex metal={self.metal_center}, ox={self.oxidation_state}, "
            f"charge={self.charge}, mult={self.multiplicity}, "
            f"ligands={[l.name for l in self.ligands]}>"
        )


# class Ti_ureate_Complex:
#     def __init__(self,
#                  metal_center: str,
#                 #  oxidation_state: int,
#                 #  ligands: list[Ligand],
#                 #  geometry: Optional[str] = None,
#                 #  template_name: str = "u1_d1_iso1.xyz"
#                  ):
#         """Parameters

#         Args:
#             metal_center (str): 
#                 Symbol of the metal center (e.g., "Ti").
#             oxidation_state (int): 
#                 Formal oxidaton state of the metal center (postive integer).
#             ligands (list[Ligand]): 
#                 List of Ligand objects to be coordinated to the metal center.
#             geometry (Optional[str], optional): 
#                 Geometry type (e.g., 'tetrahedral', 'square_pyramidal, 'octahedral'). Defaults to None.
#         """
#         self.metal_center = metal_center
#         # self.ligands = ligands
#         # self.oxidation_state = oxidation_state
#         # self.geometry = geometry or "unspecified" 
#         # self.template_name = template_name
#         # self.mol = self._load_template()

#         self.charge = None
#         self.multiplicity = None
#         self.mol3D = None # Placeholder for an RDKit Mol object
#         self.xyz = None # XYZ string representation

#         # self._estimate_charge_and_multiplicity()
    
    
    
#     def find_dummy_and_anchor(self, 
#                               template, 
#                               dummy_atomic_num=9):
#         dummy_atom = None
#         anchor_atom = None
#         for atom in template:
#             if atom.OBAtom.GetAtomicNum() == dummy_atomic_num:  # * = dummy, 9 = fluorine
#                 dummy_atom = atom
#                 # Find neighbor (the atom that was bonded to *)
#                 for nbr in pybel.ob.OBAtomAtomIter(atom.OBAtom):
#                     anchor_atom = template.atoms[nbr.GetIdx() - 1]  # 0-based Pybel indexing
#                 break

#         # if dummy_atom is None:
#         #     raise ValueError("No dummy atom (*) found in template")
#         # if anchor_atom is None:
#         #     raise ValueError("Dummy has no neighbor")
#         return dummy_atom, anchor_atom


#     def find_ligand_attachement_point(self, 
#                                       ligand, 
#                                       connect_label: str = "N1"):
#         attach_atom = None
#         connect_atom = None

#         if connect_label not in list(element_dict.keys()):
#             raise ValueError(f"The connecting atom label not recognized; connect_label provided is: {connect_label}")
        
#         connect_atomic_num = element_dict[connect_label]
#         for atom in ligand:
#             # find dummy atom with label *, which has atomic number == 0
#             if atom.OBAtom.GetAtomicNum() == 0:
#                 # find the atom that is bonded to the dummy atom
#                 for nbr in pybel.ob.OBAtomAtomIter(atom.OBAtom):
#                     break
#                 if nbr is None:
#                     raise ValueError("Fragment * has no neighbor")
#                 # check if the connect atom matches connect_label
#                 # i.e. O, N1 or N2
#                 # skip to the next * atom if the nbr doesn't match the connect atomic number
#                 if nbr.GetAtomicNum() != connect_atomic_num:
#                     continue
                
#                 if nbr.GetAtomicNum() == 8:
#                     # if oxygen we can return the attch atom and connect atom
#                     attach_atom = atom
#                     connect_atom = nbr
#                     break
#                 elif nbr.GetAtomicNum() == 7:
#                     # if nitrogen, we need to check if it's N1 or N2
#                     # N1 has 3 neighboring atoms including *
#                     # N2 has 4 neighboring atoms including *
#                     connect_nbr_dict = {"N1": 3, "N2": 4}
#                     nbr_nbr_count = 0
#                     for nbr_nbr in pybel.ob.OBAtomAtomIter(nbr):
#                         nbr_nbr_count += 1
#                     print (f"Number of neighbors: {nbr_nbr_count}")
#                     if nbr_nbr_count == connect_nbr_dict[connect_label]:
#                         attach_atom = atom
#                         connect_atom = nbr
#                         break
#                     else: # check the next dummy *
#                         continue
#                 else:
#                     raise ValueError("The connecting atom is neither an oxygen or nitrogen")
#         if attach_atom is None:
#             raise ValueError("No * in fragment")
#         if connect_atom is None:
#             raise ValueError("Fragment * has no neighbor")  
        
#         return attach_atom, connect_atom

#     def find_fragment_attachment_point(self, frag):
#         attach_atom = None
#         for atom in frag:
#             if atom.OBAtom.GetAtomicNum() == 0:  # *
#                 attach_atom = atom
#                 break
#         if attach_atom is None:
#             raise ValueError("No * in fragment")
        
#         # Get the real atom atom after *
#         connect_atom = None
#         for nbr in pybel.ob.OBAtomAtomIter(attach_atom.OBAtom):
#             connect_atom = nbr
#             break
#         if connect_atom is None:
#             raise ValueError("Fragment * has no neighbor")
        
#         return attach_atom, connect_atom
#     def run_opt(self, 
#                 constraints,
#                 mol,
#                 ):
#         # === Suppress logs ===
#         cerr_fd = sys.stderr.fileno()
#         devnull = os.open('/dev/null', os.O_WRONLY)
#         original_cerr = os.dup(cerr_fd)
#         os.dup2(devnull, cerr_fd)

#         try:
#             ff = ob.OBForceField.FindForceField(self.force_field)
#             if not ff.Setup(mol.OBMol, constraints):
#                 raise ValueError("Setup failed")
            
#             print(f"Initial energy: {ff.Energy():.2f} kcal/mol")
#             # ff.ConjugateGradients(force_field_steps)
#             ff.ConjugateGradients(self.force_field_steps) 
#             print(f"Final energy: {ff.Energy():.2f} kcal/mol")
            
#             ff.GetCoordinates(mol.OBMol)
#         finally:
#             os.dup2(original_cerr, cerr_fd)
#             os.close(original_cerr)
#             os.close(devnull)

#     def constraint_optimization(self, constraints, mol):
#         constraints.Clear()
#         metal_atomic_num = element_dict[self.metal_center]
#         metal_nbr_atom_ls = []
#         for atom in mol:
#             if atom.OBAtom.GetAtomicNum() == metal_atomic_num:  # * = dummy, 9 = fluorine
#                 metal_atom = atom
#                 # Find neighbor (the atom that was bonded to *)
#                 for nbr in pybel.ob.OBAtomAtomIter(atom.OBAtom):
#                     nbr_atom = mol.atoms[nbr.GetIdx() - 1]
#                     nbr_idx = nbr_atom.idx
#                     # print (f"Add atom constraint to atom with nbr_idx: {nbr_idx}")
#                     constraints.AddAtomConstraint(nbr_idx)

#         self.run_opt(constraints, mol)
#         return constraints, mol

#     def add_fragments(self,
#                       template_sdf: str,
#                       fragment_smiles_dict: dict,
#                       N1N2_dict: dict,
#                       output_sdf: str,
#                       force_field: str = "mmff94", # "uff" or "mmff94", for geometry optimization
#                       force_field_steps: int = 2000, # Number of optimization steps
#                       ):
#         self.force_field = force_field
#         self.force_field_steps = force_field_steps
#         """Build 3D complex geometry by adding substituents
#         on the ligands coordinated to the metal center.
#         1. Identify dummy atoms in template and fragments.
#         2. Remove dummy atoms and merge fragments into template.
#         3. Add bonds between metal center and fragment attachment points.
       
#         There are three substituents (R1, R2, R3) on the ureate ligand
#         The placement of the substituent fragments 
#         is determined by the keys in fragment_smiles_dict. 

#         To identify which dummy atom corresponds to which substituent,
#         we place dummy halogen atoms on the metal template:
#         R1: Fluorine (F, atomic num 9)
#         R2: Chlorine (Cl, atomic num 17)
#         R3: Bromine (Br, atomic num 35)

#         Thus, the fragment_smiles_dict is in the form of:
#         {"R1": "SMILES_string_for_R1,
#          "R2": "SMILES_string_for_R2,
#          "R3": "SMILES_string_for_R3
#         } 
#         and the code will replace the corresponding
#         dummy atoms with the fragments using the substituent_dummy_map.


#         Args:
#             template_sdf (str): 
#                 Path to the template SDF file with dummy atoms.
#             fragment_smiles_dict (dict): 
#                 Dictionary mapping dummy atom symbols to fragment SMILES strings.
#             output_sdf (str): 
#                 Path to save the output SDF file.
#             force_field (str, optional): 
#                 Force field for geometry optimization ("uff" or "mmff94"). Defaults to "mmff94".
#             force_field_steps (int, optional): 
#                 Number of optimization steps. Defaults to 2000.
#           """
#         # --------------------------------------------------------------
#         # 1. Load the template (keep hydrogens if you need them)
#         # --------------------------------------------------------------
        
#         # template = load_ti_template_sdf(template_sdf, package = "openbabel",
#         #                                 template_dir = "TM_catalyst_framework.template")
#         template = load_ti_template_sdf(template_sdf, 
#                                         package = "openbabel", 
#                                         template_dir = "TM_catalyst_framework.metal_center_template")
#         if template is None:
#             raise ValueError("Could not read the template SDF file.")
#         # template.make3D()

#         constraints = ob.OBFFConstraints()

#         for Ni, dummy_atomic_num_ls in N1N2_dict.items():
#             for idx, dummy_atomic_num in enumerate(dummy_atomic_num_ls):
#                 if Ni == "N1": # nitrogen with R1
#                     # --------------------------------------------------
#                     # 2. Replace dummy atom with substituent fragment
#                     # --------------------------------------------------
#                     # Find dummy and anchor in template
#                     dummy_atom, anchor_atom = self.find_dummy_and_anchor(template, dummy_atomic_num)
#                     dummy_atom_idx = dummy_atom.idx
#                     anchor_idx = anchor_atom.idx
#                     # --------------------------------------------------
#                     # 3. Load fragment and find attachment point
#                     # --------------------------------------------------
#                     # Load fragment and find attachment point
#                     frag = pybel.readstring("smi", fragment_smiles_dict["R1"])
#                     frag.make3D()
#                     attach_atom, connect_atom = self.find_fragment_attachment_point(frag)
#                     connect_idx = connect_atom.GetIdx()  # 1-based

#                     # --------------------------------------------------
#                     # 4. Delete dummy, merge, and ADD BOND
#                     # --------------------------------------------------
#                     # Replace dummy atom to nitrogen
#                     print(f"dummy_atom atomic number: {dummy_atom.OBAtom.GetAtomicNum()}")
#                     #TODO unhash -->dummy_atom.OBAtom.SetAtomicNum(7)

#                     # Merge fragment into template
#                     template.OBMol += frag.OBMol

#                     # Now add bond: anchor (in template) → connect_atom (in fragment)
#                     # After merge, fragment atoms start at: mol.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
#                     frag_start_idx = template.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
#                     connect_idx_in_merged = frag_start_idx + connect_idx - 1

#                     # Add bond (1-based indices)
#                     template.OBMol.AddBond(dummy_atom_idx, connect_idx_in_merged, 1)  # 1 = single bond

#                     #cleanup: check if there's any remaining * atoms and remove them
#                     delete_indices = []
#                     for atom in template:
#                         # print (f"New molecule Atom idx: {atom.idx}, atomic num: {atom.OBAtom.GetAtomicNum()}")
#                         if atom.OBAtom.GetAtomicNum() == 0:  # *
#                             # print (f"Found leftover * at index {atom.idx}, marking for deletion")
#                             template.OBMol.DeleteAtom(atom.OBAtom)
#                     print (f"Fragment attachment at index {attach_atom.idx}")
#                     # === Freeze atoms attaching the metal center ===
                    
#                     constraints.Clear()
#                     constraints, template = self.constraint_optimization(constraints, template)
                
#                 if Ni == "N2": # nitrogen with R2 and R3
#                     for r_idx in range(2, 4):
#                         # --------------------------------------------------
#                         # 2. Replace dummy atom with substituent fragment
#                         # --------------------------------------------------
#                         # Find dummy and anchor in template
#                         dummy_atom, anchor_atom = self.find_dummy_and_anchor(template, dummy_atomic_num)
#                         dummy_atom_idx = dummy_atom.idx
#                         anchor_idx = anchor_atom.idx

#                         frag = pybel.readstring("smi", fragment_smiles_dict["R%s"%r_idx])
#                         frag.make3D()
#                         attach_atom, connect_atom = self.find_fragment_attachment_point(frag)
#                         connect_idx = connect_atom.GetIdx()  # 1-based

#                         # --------------------------------------------------
#                         # 4. Delete dummy, merge, and ADD BOND
#                         # --------------------------------------------------
#                         # Merge fragment into template
#                         template.OBMol += frag.OBMol

#                         # Now add bond: anchor (in template) → connect_atom (in fragment)
#                         # After merge, fragment atoms start at: mol.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
#                         frag_start_idx = template.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
#                         connect_idx_in_merged = frag_start_idx + connect_idx - 1

#                         # Add bond (1-based indices)
#                         template.OBMol.AddBond(dummy_atom_idx, connect_idx_in_merged, 1)  # 1 = single bond

#                         #cleanup: check if there's any remaining * atoms and remove them
#                         delete_indices = []
#                         for atom in template:
#                             # print (f"New molecule Atom idx: {atom.idx}, atomic num: {atom.OBAtom.GetAtomicNum()}")
#                             if atom.OBAtom.GetAtomicNum() == 0:  # *
#                                 # print (f"Found leftover * at index {atom.idx}, marking for deletion")
#                                 template.OBMol.DeleteAtom(atom.OBAtom)
#                         print (f"Fragment attachment at index {attach_atom.idx}")

#                         # === Freeze atoms attaching the metal center ===
#                         constraints.Clear()
#                         constraints, template = self.constraint_optimization(constraints, template)

#         # --------------------------------------------------
#         # Replace dummy atoms to nitrogen
#         # --------------------------------------------------
#         for Ni, dummy_atomic_num_ls in N1N2_dict.items():
#             for dummy_atomic_num_i in dummy_atomic_num_ls:
#                 dummy_atom, anchor_atom = self.find_dummy_and_anchor(template, dummy_atomic_num_i)
#                 dummy_atom.OBAtom.SetAtomicNum(7)
                    
#         # --------------------------------------------------
#         # 5. Generate 3D and optimize
#         # --------------------------------------------------
#         # === Freeze atoms attaching the metal center ===
#         constraints.Clear()

#         metal_atomic_num = 22# d_electrons_dict[self.metal_center]
#         metal_nbr_atom_ls = []
#         for atom in template:
#             if atom.OBAtom.GetAtomicNum() == metal_atomic_num:  # * = dummy, 9 = fluorine
#                 metal_atom = atom
#                 # Find neighbor (the atom that was bonded to *)
#                 for nbr in pybel.ob.OBAtomAtomIter(atom.OBAtom):
#                     nbr_atom = template.atoms[nbr.GetIdx() - 1]
#                     nbr_idx = nbr_atom.idx
#                     # print (f"Add atom constraint to atom with nbr_idx: {nbr_idx}")
#                     constraints.AddAtomConstraint(nbr_idx)

#         # === Suppress logs ===
#         cerr_fd = sys.stderr.fileno()
#         devnull = os.open('/dev/null', os.O_WRONLY)
#         original_cerr = os.dup(cerr_fd)
#         os.dup2(devnull, cerr_fd)

#         try:
#             ff = ob.OBForceField.FindForceField(force_field)
#             if not ff.Setup(template.OBMol, constraints):
#                 raise ValueError("Setup failed")
            
#             print(f"Initial energy: {ff.Energy():.2f} kcal/mol")
#             # ff.ConjugateGradients(force_field_steps)
#             ff.ConjugateGradients(force_field_steps) 
#             print(f"Final energy: {ff.Energy():.2f} kcal/mol")
            
#             ff.GetCoordinates(template.OBMol)
#         finally:
#             os.dup2(original_cerr, cerr_fd)
#             os.close(original_cerr)
#             os.close(devnull)
        
#         # Final geometry optimization with FF without the constraints
#         # constraints.Clear()
#         # template.make3D(forcefield=force_field, steps=force_field_steps)
#         # --------------------------------------------------
#         # 6. Save
#         # --------------------------------------------------
#         # template.make3D(force_field, force_field_steps)
#         template.write("sdf", output_sdf, overwrite=True)
            
#         print(f"Success! Saved to {output_sdf}")

#     def add_ligand_bond(self,
#                         mol,
#                         ligand,
#                         connect_idx,
#                         anchor_idx,
#                         dummy_atom,
#                         ):
#         dummy_coord = dummy_atom.OBAtom.GetVector()
#         mol.OBMol.DeleteAtom(dummy_atom.OBAtom)
        
#         # Now add bond: anchor (in template) → connect_atom (in fragment)
#         # After merge, fragment atoms start at: mol.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
#         frag_start_idx = mol.OBMol.NumAtoms() - ligand.OBMol.NumAtoms() + 1

#         # connect atom index
#         connect_idx_in_merged = frag_start_idx + connect_idx - 1
#         # Add M-O bond (1-based indices)
#         mol.OBMol.AddBond(anchor_idx, connect_idx_in_merged, 1)  # 1 = single bond
#         connect_atom = mol.OBMol.GetAtom(connect_idx_in_merged)
#         connect_atom.SetVector(dummy_coord)
#         return mol
    
#     def attach_monodentate_ligand(self,
#                                   mol,
#                                   ligand_smiles: str = "*N(C)C",
#                                   dummy_atomic_num_: int = 2,
#                                   halogen_mapping = False,
#                                   ):
#         if halogen_mapping == True:
#             noble_map= {
#                     1:2,# helium at position 1
#                     2:10, # neon at position 2
#                     3:18, # Argon at position 3
#                     4:36, # Kr at position 4
#                     5:54, # Xe at position 5
#                     6:86 # Rn at position 6
#                 }
#             dummy_atomic_num = noble_map[dummy_atomic_num_]
            
#         else:
#             dummy_atomic_num = dummy_atomic_num_
#         constraints = ob.OBFFConstraints()
#         constraints, mol = self.constraint_optimization(constraints, mol)
#         ligand = pybel.readstring("smi", ligand_smiles)
#         ligand.make3D()

#         # count nunmber of dummy 
#         dummy_count = 0
#         for atom in mol:
#             if atom.OBAtom.GetAtomicNum() == dummy_atomic_num: 
#                 dummy_count += 1
#         if dummy_count != 0:
#             for dummy_i in range(dummy_count):
#                 # Merge fragment into template
#                 mol.OBMol += ligand.OBMol
                
#                 # Find dummy atom representing oxygen on complex template
#                 dummy_atom, anchor_atom = self.find_dummy_and_anchor(mol, dummy_atomic_num)

#                 ######## Ligand ########
#                 # find dummy
#                 attach_atom = None
#                 connect_atom = None
#                 for atom in ligand:
#                     # find dummy atom with label *, which has atomic number == 0
#                     if atom.OBAtom.GetAtomicNum() == 0:
#                         # find the atom that is bonded to the dummy atom
#                         for nbr in pybel.ob.OBAtomAtomIter(atom.OBAtom):
#                             break
#                         if nbr is None:
#                             raise ValueError("Fragment * has no neighbor")
#                     attach_atom = atom
#                     connect_atom = nbr
#                     break
#                 if attach_atom is None:
#                     raise ValueError("No * in fragment")
#                 if connect_atom is None:
#                     raise ValueError("Fragment * has no neighbor")  
#                 mol = self.add_ligand_bond(
#                                 mol,
#                                 ligand,
#                                 connect_atom.GetIdx(),
#                                 anchor_atom.idx,
#                                 dummy_atom,
#                                 )
                
#                 for atom in mol:
#                     # print (f"New molecule Atom idx: {atom.idx}, atomic num: {atom.OBAtom.GetAtomicNum()}")
#                     if atom.OBAtom.GetAtomicNum() == 0:  # *
#                         # print (f"Found leftover * at index {atom.idx}, marking for deletion")
#                         mol.OBMol.DeleteAtom(atom.OBAtom)
#                 print (f"Fragment attachment at index {attach_atom.idx}")
#                 # === Freeze atoms attaching the metal center ===
                
#                 constraints.Clear()
#                 constraints, mol = self.constraint_optimization(constraints, mol)
#             # --------------------------------------------------
#             # Final geometry optimization with FF (no constraints)
#             # --------------------------------------------------
#             constraints.Clear()
#             self.run_opt(constraints, mol)
#             self.complex = mol

#         else:
#             raise ValueError("No dummy found!")

#     def attach_bidentate_ligand(self,
#                                 constraints, 
#                                 mol,
#                                 dummyO_atomic_num, 
#                                 dummyN_atomic_num
#                                 ):
#         # Merge fragment into template
#         mol.OBMol += self.ligand.OBMol

#         # Find dummy atom representing oxygen on complex template
#         dummy_O_atom, anchor_atom = self.find_dummy_and_anchor(mol, dummyO_atomic_num)
#         # print (dummy_O_atom)
#         # Get dummy coordinate
        
#         # Find dummy atom representing nitrogen on complex template
#         dummy_N_atom, anchor_atom = self.find_dummy_and_anchor(mol, dummyN_atomic_num)
#         # print (dummy_O_atom)
#         # Get dummy coordinate
        
#         ######## Ligand ########
#         # find O dummy
#         dummy_O_label = "O1"
#         attach_O, connect_O = self.find_ligand_attachement_point(self.ligand, 
#                                 connect_label=dummy_O_label)
#         connect_O_idx = connect_O.GetIdx()  # 1-based

#         # find Ni dummy
#         dummy_N_label = self.connect_N_dict[dummyN_atomic_num]
#         attach_N, connect_N = self.find_ligand_attachement_point(self.ligand, 
#                                 connect_label=dummy_N_label)
#         connect_N_idx = connect_N.GetIdx()  # 1-based

#         mol = self.add_ligand_bond(mol, self.ligand,
#                                 connect_O_idx, anchor_atom.idx, dummy_O_atom)
        
#         mol = self.add_ligand_bond(mol, self.ligand,
#                                 connect_N_idx, anchor_atom.idx, dummy_N_atom)
        
#         for atom in mol:
#             # print (f"New molecule Atom idx: {atom.idx}, atomic num: {atom.OBAtom.GetAtomicNum()}")
#             if atom.OBAtom.GetAtomicNum() == 0:  # *
#                 print (f"Found leftover * at index {atom.idx}, marking for deletion")
#                 mol.OBMol.DeleteAtom(atom.OBAtom)
#         constraints, mol = self.constraint_optimization(constraints, mol)
        
#         return constraints, mol


#     def add_ligands(self,
#                     template_sdf: str,
#                     ligand_smiles: str,
#                     output_sdf: str,
#                     force_field: str = "mmff94",
#                     force_field_steps: int = 2000,
#                     connect_N_dict: dict = {35: "N1", 53: "N1"},
#                     ):
#         self.force_field = force_field
#         self.force_field_steps = force_field_steps
#         """Build 3D complex geometry by adding ligand to the metal center.

#         Args:
#             template_sdf (str): 
#                 Path to the template SDF file with dummy atoms.
#             ligand_smiles (str):
#                 ureate ligand smiles with dummy atoms * attaching to O, N1, and N2
#             output_sdf (str): 
#                 Path to save the output SDF file.
#             force_field (str, optional): 
#                 Force field for geometry optimization ("uff" or "mmff94"). Defaults to "mmff94".
#             force_field_steps (int, optional): 
#                 Number of optimization steps. Defaults to 2000.

#         Raises:
#             ValueError: _description_

#         Returns:
#             _type_: _description_
#         """
#         self.connect_N_dict = connect_N_dict
#         # Dictionary that maps the dummy atoms representing 
#         # the ligand-coordination postion in the template
#         # metal_coord_dummy_map = {
#         #     "O1": 9, # Fluorine, as the first oxygen
#         #     "O2": 17, # Chlorine, as the second oxygen
#         #     "N1": 35, # Bromine, as the first nitrogen
#         #     "N2": 53 # Iodine, as the second nitrogen   
#         # }

#         metal_coord_dummy_map = {
#             "O1": 9, # Fluorine, as the first oxygen
#             "O2": 17, # Chlorine, as the second oxygen
#             "N1": 10, # Bromine, as the first nitrogen
#             "N2": 18 # Iodine, as the second nitrogen   
#         }
#         d2d2_angle_map = {"u2_d2d2_iso7":{"N":180, "O": 90}} 

#         # --------------------------------------------------------------
#         # 1. Load the template (keep hydrogens if you need them)
#         # --------------------------------------------------------------

#         template = load_ti_template_sdf(template_sdf, 
#                                         package = "openbabel", 
#                                         template_dir = "TM_catalyst_framework.template")
#         if template is None:
#             raise ValueError("Could not read the template SDF file.")
        
#         # template.make3D(forcefield=force_field, steps=force_field_steps)
#         skip_count = 0
#         ligand = pybel.readstring("smi", ligand_smiles)
#         ligand.make3D()
#         self.ligand = ligand
#         dummy_atom_dict = {}

#         # --------------------------------------------------
#         # 2. Coordinate 1 or 2 ligands
#         # --------------------------------------------------
#         if "d2d2" in template_sdf:
#             constraints = ob.OBFFConstraints()
#             # --------------------------------------------------
#             # 2a.1. Add the first ligand
#             # --------------------------------------------------
#             dummyO_atomic_num, dummyN_atomic_num = metal_coord_dummy_map["O1"], metal_coord_dummy_map["N1"]
#             constraints, template = self.attach_bidentate_ligand(constraints, 
#                                         template,
#                                         dummyO_atomic_num, 
#                                         dummyN_atomic_num
#                                         )

#             # --------------------------------------------------
#             # 2a.2. Add the second ligand
#             # --------------------------------------------------
#             # Merge fragment into template
#             dummyO_atomic_num, dummyN_atomic_num = metal_coord_dummy_map["O2"], metal_coord_dummy_map["N2"]
#             constraints, template = self.attach_bidentate_ligand(constraints, 
#                                         template,
#                                         dummyO_atomic_num, 
#                                         dummyN_atomic_num
#                                         )
#         else:
#             # template.make3D(forcefield=force_field, steps=force_field_steps)
#             # constraints = ob.OBFFConstraints()
#             # --------------------------------------------------
#             # 2b. Check if template contains dummy atom
#             # --------------------------------------------------
#             for dummy_atom_label, dummy_atomic_num in metal_coord_dummy_map.items():
#                 dummy_atom, anchor_atom = self.find_dummy_and_anchor(template, dummy_atomic_num)
#                 # The template doesn't contain the dummy atom, skip and find the next one
#                 if dummy_atom == None:
#                     skip_count += 1
#                     continue 
#                 anchor_idx = anchor_atom.idx # 1-based Open Babel index
#                 dummy_atom_dict[dummy_atom_label] = [dummy_atom, anchor_atom, anchor_idx]
#             print (f"Dummy atoms found from template: {dummy_atom_dict.keys()}")
#             constraints = ob.OBFFConstraints()
#             # constraints, template = self.constraint_optimization(constraints, template)
#             for i in range(1,3):
#                 if f"O{i}" in dummy_atom_dict.keys():
#                     if f"N{i}" in dummy_atom_dict.keys():
#                         # --------------------------------------------------
#                         # 2b.1. Coordinate bidentate ligand
#                         # --------------------------------------------------
#                         print (f"Monodentate ligand N{i}O{i}")
#                         dummyO_atomic_num = metal_coord_dummy_map["O%s"%(i)]
#                         dummyN_atomic_num = metal_coord_dummy_map["N%s"%(i)]
#                         constraints, template = self.attach_bidentate_ligand(
#                                                     constraints, 
#                                                     template,
#                                                     dummyO_atomic_num, 
#                                                     dummyN_atomic_num
#                                                     )
#                     else:
#                         # --------------------------------------------------
#                         # 2b.2. Coordinate monodentate ligand
#                         # --------------------------------------------------
#                         dummy_O_label = "O%s"%(i)
#                         print (f"Monodentate ligand N{i}O{i}")
#                         # Merge fragment into template
#                         template.OBMol += ligand.OBMol

#                         # Find F dummy atom on complex template
#                         dummy_atomic_num = metal_coord_dummy_map[dummy_O_label]
#                         dummy_O_atom, anchor_atom = self.find_dummy_and_anchor(template, dummy_atomic_num)
                        
#                         ######## Ligand ########
#                         # find O dummy
#                         attach_O, connect_O = self.find_ligand_attachement_point(ligand, 
#                                                 connect_label=dummy_O_label)
#                         connect_O_idx = connect_O.GetIdx()  # 1-based

#                         template = self.add_ligand_bond(template, ligand,
#                                             connect_O_idx, anchor_atom.idx, dummy_O_atom)
                        
                        
#                         for atom in template:
#                             # print (f"New molecule Atom idx: {atom.idx}, atomic num: {atom.OBAtom.GetAtomicNum()}")
#                             if atom.OBAtom.GetAtomicNum() == 0:  # *
#                                 print (f"Found leftover * at index {atom.idx}, marking for deletion")
#                                 template.OBMol.DeleteAtom(atom.OBAtom)
#                         constraints, template = self.constraint_optimization(constraints, template)
        
#         # --------------------------------------------------
#         # 3. Final geometry optimization with FF (no constraints)
#         # --------------------------------------------------
#         constraints.Clear()
#         self.run_opt(constraints, template)
#         # --------------------------------------------------
#         # 4. Save
#         # --------------------------------------------------
#         template.write("sdf", output_sdf, overwrite=True)
#         print(f"Success! Saved to {output_sdf}")
#         self.complex = template
    
#         if skip_count == 4:
#             raise ValueError("No ligand is attached to the template because 0 dummy atoms were found.")
        
#     def add_ligands_wt_geometry(self,
#                     template_sdf: str,
#                     ligand_smiles: str,
#                     output_sdf: str,
#                     force_field: str = "mmff94",
#                     force_field_steps: int = 2000,
#                     connect_N_identity_dict: dict = {3: "N1", 5: "N1"},
#                     metal_coord_dummy_map = {"O1":2, "N1":3, "O2":4, "N2":5}
#                     ):
#         self.force_field = force_field
#         self.force_field_steps = force_field_steps
#         """Build 3D complex geometry by adding ligand to the metal center.

#         Args:
#             template_sdf (str): 
#                 Path to the template SDF file with dummy atoms.
#             ligand_smiles (str):
#                 ureate ligand smiles with dummy atoms * attaching to O, N1, and N2
#             output_sdf (str): 
#                 Path to save the output SDF file.
#             force_field (str, optional): 
#                 Force field for geometry optimization ("uff" or "mmff94"). Defaults to "mmff94".
#             force_field_steps (int, optional): 
#                 Number of optimization steps. Defaults to 2000.

#         Raises:
#             ValueError: _description_

#         Returns:
#             _type_: _description_
#         """
#         noble_map= {
#             1:2,# helium at position 1
#             2:10, # neon at position 2
#             3:18, # Argon at position 3
#             4:36, # Kr at position 4
#             5:54, # Xe at position 5
#             6:86 # Rn at position 6
#         }
#         connect_N_dict = {}
#         for coord_num, nitrogen_identity in connect_N_identity_dict.items():
#             connect_N_dict[noble_map[coord_num]] = nitrogen_identity
#         self.connect_N_dict = connect_N_dict
#         # Dictionary that maps the dummy atoms representing 
#         # the ligand-coordination postion in the template
#         # metal_coord_dummy_map = {
#         #     "O1": 9, # Fluorine, as the first oxygen
#         #     "O2": 17, # Chlorine, as the second oxygen
#         #     "N1": 35, # Bromine, as the first nitrogen
#         #     "N2": 53 # Iodine, as the second nitrogen   
#         # }

#         # metal_coord_dummy_map = {
#         #     "O1": 9, # Fluorine, as the first oxygen
#         #     "O2": 17, # Chlorine, as the second oxygen
#         #     "N1": 10, # Bromine, as the first nitrogen
#         #     "N2": 18 # Iodine, as the second nitrogen   
#         # }
        
#         # --------------------------------------------------------------
#         # 1. Load the template (keep hydrogens if you need them)
#         # --------------------------------------------------------------

#         template = load_ti_template_sdf(template_sdf, 
#                                         package = "openbabel", 
#                                         template_dir = "TM_catalyst_framework.template")
#         if template is None:
#             raise ValueError("Could not read the template SDF file.")
        
#         # template.make3D(forcefield=force_field, steps=force_field_steps)
#         skip_count = 0
#         ligand = pybel.readstring("smi", ligand_smiles)
#         ligand.make3D()
#         self.ligand = ligand
#         dummy_atom_dict = {}

#         # --------------------------------------------------
#         # 2. Coordinate 1 or 2 ligands
#         # --------------------------------------------------
#         if "d2d2" in template_sdf:
#             constraints = ob.OBFFConstraints()
#             # --------------------------------------------------
#             # 2a.1. Add the first ligand
#             # --------------------------------------------------
#             dummyO_atomic_num, dummyN_atomic_num = noble_map[metal_coord_dummy_map["O1"]], noble_map[metal_coord_dummy_map["N1"]]
#             constraints, template = self.attach_bidentate_ligand(constraints, 
#                                         template,
#                                         dummyO_atomic_num, 
#                                         dummyN_atomic_num
#                                         )

#             # --------------------------------------------------
#             # 2a.2. Add the second ligand
#             # --------------------------------------------------
#             # Merge fragment into template
#             dummyO_atomic_num, dummyN_atomic_num = noble_map[metal_coord_dummy_map["O2"]], noble_map[metal_coord_dummy_map["N2"]]
#             constraints, template = self.attach_bidentate_ligand(constraints, 
#                                         template,
#                                         dummyO_atomic_num, 
#                                         dummyN_atomic_num
#                                         )
#         else:
#             # template.make3D(forcefield=force_field, steps=force_field_steps)
#             # constraints = ob.OBFFConstraints()
#             # --------------------------------------------------
#             # 2b. Check if template contains dummy atom
#             # --------------------------------------------------
#             for dummy_atom_label, dummy_atomic_num in metal_coord_dummy_map.items():
#                 dummy_atom, anchor_atom = self.find_dummy_and_anchor(template, noble_map[dummy_atomic_num])
#                 # The template doesn't contain the dummy atom, skip and find the next one
#                 if dummy_atom == None:
#                     skip_count += 1
#                     continue 
#                 anchor_idx = anchor_atom.idx # 1-based Open Babel index
#                 dummy_atom_dict[dummy_atom_label] = [dummy_atom, anchor_atom, anchor_idx]
#             print (f"Dummy atoms found from template: {dummy_atom_dict.keys()}")
#             constraints = ob.OBFFConstraints()
#             # constraints, template = self.constraint_optimization(constraints, template)
#             for i in range(1,3):
#                 if f"O{i}" in dummy_atom_dict.keys():
#                     if f"N{i}" in dummy_atom_dict.keys():
#                         # --------------------------------------------------
#                         # 2b.1. Coordinate bidentate ligand
#                         # --------------------------------------------------
#                         print (f"Monodentate ligand N{i}O{i}")
#                         dummyO_atomic_num = noble_map[metal_coord_dummy_map["O%s"%(i)]]
#                         dummyN_atomic_num = noble_map[metal_coord_dummy_map["N%s"%(i)]]
#                         constraints, template = self.attach_bidentate_ligand(
#                                                     constraints, 
#                                                     template,
#                                                     dummyO_atomic_num, 
#                                                     dummyN_atomic_num
#                                                     )
#                     else:
#                         # --------------------------------------------------
#                         # 2b.2. Coordinate monodentate ligand
#                         # --------------------------------------------------
#                         dummy_O_label = "O%s"%(i)
#                         print (f"Monodentate ligand N{i}O{i}")
#                         # Merge fragment into template
#                         template.OBMol += ligand.OBMol

#                         # Find F dummy atom on complex template
#                         dummy_atomic_num = metal_coord_dummy_map[dummy_O_label]
#                         dummy_O_atom, anchor_atom = self.find_dummy_and_anchor(template, noble_map[dummy_atomic_num])
                        
#                         ######## Ligand ########
#                         # find O dummy
#                         attach_O, connect_O = self.find_ligand_attachement_point(ligand, 
#                                                 connect_label=dummy_O_label)
#                         connect_O_idx = connect_O.GetIdx()  # 1-based

#                         template = self.add_ligand_bond(template, ligand,
#                                             connect_O_idx, anchor_atom.idx, dummy_O_atom)
                        
                        
#                         for atom in template:
#                             # print (f"New molecule Atom idx: {atom.idx}, atomic num: {atom.OBAtom.GetAtomicNum()}")
#                             if atom.OBAtom.GetAtomicNum() == 0:  # *
#                                 print (f"Found leftover * at index {atom.idx}, marking for deletion")
#                                 template.OBMol.DeleteAtom(atom.OBAtom)
#                         constraints, template = self.constraint_optimization(constraints, template)
        
#         # --------------------------------------------------
#         # 3. Final geometry optimization with FF (no constraints)
#         # --------------------------------------------------
#         constraints.Clear()
#         self.run_opt(constraints, template)
#         # --------------------------------------------------
#         # 4. Save
#         # --------------------------------------------------
#         template.write("sdf", output_sdf, overwrite=True)
#         print(f"Success! Saved to {output_sdf}")
#         self.complex = template
    
#         if skip_count == 4:
#             raise ValueError("No ligand is attached to the template because 0 dummy atoms were found.")

#     # =========================================================================
#     # Electronic configuration estimation
#     # =========================================================================
    
#     def _estimate_charge_and_multiplicity(self):
#         """Estimate total charge and spin multiplicity of the complex.
#         """
#         total_ligand_charge = sum(l.charge for l in self.ligands)
#         self.charge = self.oxidation_state + total_ligand_charge

#         # Approximate d-electron count for 3d and early 4d/5d metals
#         d_electrons_dict = {
#             "Sc": 1, "Ti": 2, "V": 3, "Cr": 5, "Mn": 5,
#             "Fe": 6, "Co": 7, "Ni": 8, "Cu": 10, "Zn": 10,
#             "Zr": 2, "Nb": 3, "Mo": 6, "Hf": 2, "Ta": 3, "W": 6
#         }

#         base_d = d_electrons_dict.get(self.metal_center, 0)
#         d_electrons = max(base_d - self.oxidation_state, 0
#                           )
        
#         #Heuristic multiplicity guess
#         if d_electrons == 0:
#             self.multiplicity = 1 # singlet
#         elif d_electrons == 1:
#             self.multiplicity = 2 # doublet
#         elif d_electrons in [2, 3]:
#             self.multiplicity = 3 #triplet
#         else: 
#             self.multiplicity = 1 #fallback
        
#     # =========================================================================
#     # 2. Genereate 3D structure (basic mock-up)
#     # =========================================================================

    

#     # =========================================================================
#     # 3. XYZ string export
#     # =========================================================================

#     def _mol_to_xyz(self):
#         """ Convert RDKit Mol object to XYZ-format string."""

#         if self.mol3D is None:
#             raise ValueError("3D structure not built yet. Run build_geometry() first")
        
#         conf = self.mol3D.GetConformer()
#         n_atoms = self.mol3D.GetNumAtoms()
#         lines = [f"{n_atoms}", f"{self.metal_center} complex"]

#         for i in range(n_atoms):
#             atom = self.mol3D.GetAtomWithIdx(i)
#             pos = conf.GetAtomPosition(i)
#             lines.append(f"{atom.GetSymbol()} {pos.x: 4f} {pos.y:4f}")
#         return "\n".join(lines)
    
#     # -------------------------------------------------------------------------
#     # 4. ORCA input helper
#     # -------------------------------------------------------------------------
#     def get_orca_header(self):
#         """Return '* xyz charge multiplicity' line for ORCA input."""
#         if self.charge is None or self.multiplicity is None:
#             self._estimate_charge_and_multiplicity()
#         return f"* xyz {self.charge} {self.multiplicity}"

#     # -------------------------------------------------------------------------
#     # 5. Representation
#     # -------------------------------------------------------------------------
#     def __repr__(self):
#         return (
#             f"<Complex metal={self.metal_center}, ox={self.oxidation_state}, "
#             f"charge={self.charge}, mult={self.multiplicity}, "
#             f"ligands={[l.name for l in self.ligands]}>"
#         )
