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

from TM_catalyst_framework.metal_template import load_ti_template_sdf
d_electrons_dict = {
            "Sc": 1, "Ti": 22, "V": 3, "Cr": 5, "Mn": 5,
            "Fe": 6, "Co": 7, "Ni": 8, "Cu": 10, "Zn": 10,
            "Zr": 2, "Nb": 3, "Mo": 6, "Hf": 2, "Ta": 3, "W": 6
        }
class Complex:
    def __init__(self,
                 metal_center: str,
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
        # self.ligands = ligands
        # self.oxidation_state = oxidation_state
        # self.geometry = geometry or "unspecified" 
        # self.template_name = template_name
        # self.mol = self._load_template()

        self.charge = None
        self.multiplicity = None
        self.mol3D = None # Placeholder for an RDKit Mol object
        self.xyz = None # XYZ string representation

        # self._estimate_charge_and_multiplicity()
    
    
    
    def find_dummy_and_anchor(self, template, dummy_atomic_num=9):
        dummy_atom = None
        anchor_atom = None
        for atom in template:
            if atom.OBAtom.GetAtomicNum() == dummy_atomic_num:  # * = dummy, 9 = fluorine
                dummy_atom = atom
                # Find neighbor (the atom that was bonded to *)
                for nbr in pybel.ob.OBAtomAtomIter(atom.OBAtom):
                    anchor_atom = template.atoms[nbr.GetIdx() - 1]  # 0-based Pybel indexing
                break

        # if dummy_atom is None:
        #     raise ValueError("No dummy atom (*) found in template")
        # if anchor_atom is None:
        #     raise ValueError("Dummy has no neighbor")
        return dummy_atom, anchor_atom


    def find_ligand_attachement_point(self, 
                                      ligand, 
                                      connect_label: str = "N1"):
        attach_atom = None
        connect_atom = None
        element_dict = {"O1": 8, "O2": 8, "N1": 7, "N2": 7}

        if connect_label not in list(element_dict.keys()):
            raise ValueError(f"The connecting atom label not recognized; connect_label provided is: {connect_label}")
        
        connect_atomic_num = element_dict[connect_label]
        for atom in ligand:
            # find dummy atom with label *, which has atomic number == 0
            if atom.OBAtom.GetAtomicNum() == 0:
                # find the atom that is bonded to the dummy atom
                for nbr in pybel.ob.OBAtomAtomIter(atom.OBAtom):
                    break
                if nbr is None:
                    raise ValueError("Fragment * has no neighbor")
                # check if the connect atom matches connect_label
                # i.e. O, N1 or N2
                # skip to the next * atom if the nbr doesn't match the connect atomic number
                if nbr.GetAtomicNum() != connect_atomic_num:
                    continue
                
                if nbr.GetAtomicNum() == 8:
                    # if oxygen we can return the attch atom and connect atom
                    attach_atom = atom
                    connect_atom = nbr
                    break
                elif nbr.GetAtomicNum() == 7:
                    # if nitrogen, we need to check if it's N1 or N2
                    # N1 has 3 neighboring atoms including *
                    # N2 has 4 neighboring atoms including *
                    connect_nbr_dict = {"N1": 3, "N2": 4}
                    nbr_nbr_count = 0
                    for nbr_nbr in pybel.ob.OBAtomAtomIter(nbr):
                        nbr_nbr_count += 1
                    print (f"Number of neighbors: {nbr_nbr_count}")
                    if nbr_nbr_count == connect_nbr_dict[connect_label]:
                        attach_atom = atom
                        connect_atom = nbr
                        break
                    else: # check the next dummy *
                        continue
                else:
                    raise ValueError("The connecting atom is neither an oxygen or nitrogen")
        if attach_atom is None:
            raise ValueError("No * in fragment")
        if connect_atom is None:
            raise ValueError("Fragment * has no neighbor")  
        
        return attach_atom, connect_atom





    
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
    
    def add_fragments(self,
                      template_sdf: str,
                      fragment_smiles_dict: dict,
                      N1N2_dict: dict,
                      output_sdf: str,
                      force_field: str = "mmff94", # "uff" or "mmff94", for geometry optimization
                      force_field_steps: int = 2000, # Number of optimization steps
                      ):
        """Build 3D complex geometry by adding substituents
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
        template = load_ti_template_sdf(template_sdf, 
                                        package = "openbabel", 
                                        template_dir = "TM_catalyst_framework.metal_center_template")
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
                    metal_atomic_num = 22# d_electrons_dict[self.metal_center]
                    metal_nbr_atom_ls = []
                    for atom in template:
                        if atom.OBAtom.GetAtomicNum() == metal_atomic_num:  # * = dummy, 9 = fluorine
                            metal_atom = atom
                            # Find neighbor (the atom that was bonded to *)
                            for nbr in pybel.ob.OBAtomAtomIter(atom.OBAtom):
                                nbr_atom = template.atoms[nbr.GetIdx() - 1]
                                nbr_idx = nbr_atom.idx
                                print (f"Add atom constraint to atom with nbr_idx: {nbr_idx}")
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

                        metal_atomic_num = 22# d_electrons_dict[self.metal_center]
                        metal_nbr_atom_ls = []
                        for atom in template:
                            if atom.OBAtom.GetAtomicNum() == metal_atomic_num:  # * = dummy, 9 = fluorine
                                metal_atom = atom
                                # Find neighbor (the atom that was bonded to *)
                                for nbr in pybel.ob.OBAtomAtomIter(atom.OBAtom):
                                    nbr_atom = template.atoms[nbr.GetIdx() - 1]
                                    nbr_idx = nbr_atom.idx
                                    print (f"Add atom constraint to atom with nbr_idx: {nbr_idx}")
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
                    print (f"Add atom constraint to atom with nbr_idx: {nbr_idx}")
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
        # --------------------------------------------------
        # 6. Save
        # --------------------------------------------------
        # template.make3D(force_field, force_field_steps)
        template.write("sdf", output_sdf, overwrite=True)
            
        print(f"Success! Saved to {output_sdf}")

            
        #TODO
        # for substituent_label, fragment_smiles in fragment_smiles_dict.items():
        #     dummy_atomic_num = substituent_dummy_map.get(substituent_label)
        #     # --------------------------------------------------
        #     # 2. Replace dummy F with fragment
        #     # --------------------------------------------------
        #     # Find dummy and anchor in template
        #     dummy_atom, anchor_atom = self.find_dummy_and_anchor(template, dummy_atomic_num)
        #     anchor_idx = anchor_atom.idx  # 1-based Open Babel index

        #     # --------------------------------------------------
        #     # 3. Load fragment and find attachment point
        #     # --------------------------------------------------
        #     # Load fragment and find attachment point
        #     frag = pybel.readstring("smi", fragment_smiles)
        #     attach_atom, connect_atom = self.find_fragment_attachment_point(frag)
        #     connect_idx = connect_atom.GetIdx()  # 1-based

        #     # --------------------------------------------------
        #     # 4. Delete dummy, merge, and ADD BOND
        #     # --------------------------------------------------
        #     # Delete dummy from template and fragment
        #     template.OBMol.DeleteAtom(dummy_atom.OBAtom)

        #     # Merge fragment into template
        #     template.OBMol += frag.OBMol

        #     # Now add bond: anchor (in template) → connect_atom (in fragment)
        #     # After merge, fragment atoms start at: mol.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
        #     frag_start_idx = template.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
        #     connect_idx_in_merged = frag_start_idx + connect_idx - 1

        #     # Add bond (1-based indices)
        #     template.OBMol.AddBond(anchor_idx, connect_idx_in_merged, 1)  # 1 = single bond

        #     #cleanup: check if there's any remaining * atoms and remove them
        #     delete_indices = []
        #     for atom in template:
        #         print (f"New molecule Atom idx: {atom.idx}, atomic num: {atom.OBAtom.GetAtomicNum()}")
        #         if atom.OBAtom.GetAtomicNum() == 0:  # *
        #             print (f"Found leftover * at index {atom.idx}, marking for deletion")
        #             template.OBMol.DeleteAtom(atom.OBAtom)
        #     print (f"Fragment attachment at index {attach_atom.idx}")
        
        
        
    def add_ligands(self,
                    template_sdf: str,
                    ligand_smiles: str,
                    output_sdf: str,
                    force_field: str = "mmff94",
                    force_field_steps: int = 2000,
                    ):
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

        # Dictionary that maps the dummy atoms representing 
        # the ligand-coordination postion in the template
        metal_coord_dummy_map = {
            "O1": 9, # Fluorine, as the first oxygen
            "O2": 17, # Chlorine, as the second oxygen
            "N1": 35, # Bromine, as the first nitrogen
            "N2": 53 # Iodine, as the second nitrogen   
        }

        d2d2_angle_map = {"u2_d2d2_iso7":{"N":180, "O": 90}} 

        # --------------------------------------------------------------
        # 1. Load the template (keep hydrogens if you need them)
        # --------------------------------------------------------------

        template = load_ti_template_sdf(template_sdf, 
                                        package = "openbabel", 
                                        template_dir = "TM_catalyst_framework.metal_center_template")
        if template is None:
            raise ValueError("Could not read the template SDF file.")
        
        template.make3D(forcefield=force_field, steps=force_field_steps)
        skip_count = 0
        ligand = pybel.readstring("smi", ligand_smiles)
        dummy_atom_dict = {}

        # --------------------------------------------------
        # 2. Check if template contains dummy atom
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

        # check number of ligands (mono- or bis-ligated)
        num_ligands = int(template_sdf[1])
        print (f"There are {num_ligands} ureate ligands")

        # --------------------------------------------------
        # 3. Coordinate 1 or 2 ligands
        # --------------------------------------------------
        if "d2d2" in template_sdf:
            # --------------------------------------------------
            # 3b) Coordinate bidentate ligand
            # --------------------------------------------------
            # print (f"Bidentate ligand N{i}O{i}")

            # find Oi dummy
            dummy_O1_label = "O1"
            attach_O1, connect_O1 = self.find_ligand_attachement_point(ligand, 
                                    connect_label=dummy_O1_label)
            connect_O1_idx = connect_O1.GetIdx()  # 1-based

            # find Ni dummy
            dummy_N1_label = "N1"
            attach_N1, connect_N1 = self.find_ligand_attachement_point(ligand, 
                                    connect_label=dummy_N1_label)
            connect_N1_idx = connect_N1.GetIdx()  # 1-based
            
            # --------------------------------------------------
            # 3b.2. Add the first bidentate ligand
            # --------------------------------------------------
            # Merge fragment into template
            template.OBMol += ligand.OBMol

            # Delete dummy O from template and fragment
            dummy_atom, anchor_atom, anchor_O1_idx = dummy_atom_dict[dummy_O1_label]
            template.OBMol.DeleteAtom(dummy_atom.OBAtom)

            # Now add bond: anchor (in template) → connect_atom (in fragment)
            # After merge, fragment atoms start at: mol.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
            frag_start_idx = template.OBMol.NumAtoms() - ligand.OBMol.NumAtoms() + 1

            # O-index
            connect_O1_idx_in_merged = frag_start_idx + connect_O1_idx - 1
            # Add M-O bond (1-based indices)
            template.OBMol.AddBond(anchor_O1_idx, connect_O1_idx_in_merged, 1)  # 1 = single bond

            # template.make3D(forcefield=force_field, steps=force_field_steps)

            # Delete dummy N from template and fragment
            dummy_atom, anchor_atom, anchor_N1_idx = dummy_atom_dict[dummy_N1_label]
            template.OBMol.DeleteAtom(dummy_atom.OBAtom)

            frag_start_idx = template.OBMol.NumAtoms() - ligand.OBMol.NumAtoms() + 1
            # Now add bond: anchor (in template) → connect_atom (in fragment)
            # After merge, fragment atoms start at: mol.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
            # N-index
            connect_N1_idx_in_merged = frag_start_idx + connect_N1_idx - 1
            # Add M-O bond (1-based indices)
            template.OBMol.AddBond(anchor_N1_idx, connect_N1_idx_in_merged, 1)  # 1 = single bond

            # --------------------------------------------------
            # 3b.2. Add the second bidentate ligand
            # --------------------------------------------------
            # find Oi dummy
            dummy_O2_label = "O2"
            attach_O2, connect_O2 = self.find_ligand_attachement_point(ligand, 
                                    connect_label=dummy_O2_label)
            connect_O2_idx = connect_O2.GetIdx()  # 1-based

            # find Ni dummy
            dummy_N2_label = "N2"
            attach_N2, connect_N2 = self.find_ligand_attachement_point(ligand, 
                                    connect_label=dummy_N2_label)
            connect_N2_idx = connect_N2.GetIdx()  # 1-based
            
            # Merge fragment into template
            template.OBMol += ligand.OBMol

            # Delete dummy O from template and fragment
            dummy_atom, anchor_atom, anchor_O2_idx = dummy_atom_dict[dummy_O2_label]
            template.OBMol.DeleteAtom(dummy_atom.OBAtom)

            # Now add bond: anchor (in template) → connect_atom (in fragment)
            # After merge, fragment atoms start at: mol.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
            frag_start_idx = template.OBMol.NumAtoms() - ligand.OBMol.NumAtoms() + 1

            # O-index
            connect_O2_idx_in_merged = frag_start_idx + connect_O2_idx - 1
            # Add M-O bond (1-based indices)
            template.OBMol.AddBond(anchor_O2_idx, connect_O2_idx_in_merged, 1)  # 1 = single bond

            # template.make3D(forcefield=force_field, steps=force_field_steps)

            # Delete dummy N from template and fragment
            dummy_atom, anchor_atom, anchor_N2_idx = dummy_atom_dict[dummy_N2_label]
            template.OBMol.DeleteAtom(dummy_atom.OBAtom)

            frag_start_idx = template.OBMol.NumAtoms() - ligand.OBMol.NumAtoms() + 1
            # Now add bond: anchor (in template) → connect_atom (in fragment)
            # After merge, fragment atoms start at: mol.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
            # N-index
            connect_N2_idx_in_merged = frag_start_idx + connect_N2_idx - 1
            # Add M-O bond (1-based indices)
            template.OBMol.AddBond(anchor_N2_idx, connect_N2_idx_in_merged, 1)  # 1 = single bond

            template.make3D(forcefield=force_field, steps=force_field_steps)
            # ================================ angle constraints ================================#
            constraints = ob.OBFFConstraints()
            constraints.AddAngleConstraint(frag_start_idx + connect_N1_idx - 1, 
                                           anchor_N2_idx, 
                                           frag_start_idx + connect_N2_idx - 1, 
                                           180.0)
            constraints.AddAngleConstraint(frag_start_idx + connect_O1_idx - 1, 
                                           anchor_N2_idx, 
                                           frag_start_idx + connect_O2_idx - 1, 
                                           90.0)
            
            constraints.AddAngleConstraint(frag_start_idx + connect_O1_idx - 1, 
                                           anchor_N2_idx, 
                                           frag_start_idx + connect_N1_idx - 1, 
                                           90.0)
            constraints.AddAngleConstraint(frag_start_idx + connect_O1_idx - 1, 
                                           anchor_N2_idx, 
                                           frag_start_idx + connect_N2_idx - 1, 
                                           90.0)
            # Setup the force field with the constraints
            forcefield = ob.OBForceField.FindForceField(force_field)
            forcefield.Setup(template.OBMol, constraints)
            forcefield.SetConstraints(constraints)

            # Do a 500 steps conjugate gradient minimiazation
            # and save the coordinates to mol.
            forcefield.ConjugateGradients(500)
            forcefield.GetCoordinates(template.OBMol)

            # template.make3D(forcefield=forcefield, steps=force_field_steps)

            # Do a 500 steps conjugate gradient minimiazation
            # and save the coordinates to mol.
            # forcefield.ConjugateGradients(force_field_steps)

        else:
            for i in range(1,3):
                if f"O{i}" in dummy_atom_dict.keys():
                    if f"N{i}" in dummy_atom_dict.keys():
                        # --------------------------------------------------
                        # 3b) Coordinate bidentate ligand
                        # --------------------------------------------------
                        print (f"Bidentate ligand N{i}O{i}")

                        # find Oi dummy
                        dummy_O_label = f"O{i}"
                        attach_O, connect_O = self.find_ligand_attachement_point(ligand, 
                                                connect_label=dummy_O_label)
                        connect_O_idx = connect_O.GetIdx()  # 1-based

                        # find Ni dummy
                        dummy_N_label = f"N{i}"
                        attach_N, connect_N = self.find_ligand_attachement_point(ligand, 
                                                connect_label=dummy_N_label)
                        connect_N_idx = connect_N.GetIdx()  # 1-based
                        
                        # --------------------------------------------------
                        # 3b.2. Delete dummy, merge, and ADD BOND
                        # --------------------------------------------------
                        # Merge fragment into template
                        template.OBMol += ligand.OBMol

                        # Delete dummy O from template and fragment
                        dummy_atom, anchor_atom, anchor_O_idx = dummy_atom_dict[dummy_O_label]
                        template.OBMol.DeleteAtom(dummy_atom.OBAtom)

                        # Now add bond: anchor (in template) → connect_atom (in fragment)
                        # After merge, fragment atoms start at: mol.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
                        frag_start_idx = template.OBMol.NumAtoms() - ligand.OBMol.NumAtoms() + 1

                        # O-index
                        connect_O_idx_in_merged = frag_start_idx + connect_O_idx - 1
                        # Add M-O bond (1-based indices)
                        template.OBMol.AddBond(anchor_O_idx, connect_O_idx_in_merged, 1)  # 1 = single bond

                        # template.make3D(forcefield=force_field, steps=force_field_steps)

                        # Delete dummy N from template and fragment
                        dummy_atom, anchor_atom, anchor_N_idx = dummy_atom_dict[dummy_N_label]
                        template.OBMol.DeleteAtom(dummy_atom.OBAtom)

                        # Now add bond: anchor (in template) → connect_atom (in fragment)
                        # After merge, fragment atoms start at: mol.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
                        # N-index
                        connect_N_idx_in_merged = frag_start_idx + connect_N_idx - 1
                        # Add M-O bond (1-based indices)
                        template.OBMol.AddBond(anchor_N_idx, connect_N_idx_in_merged, 1)  # 1 = single bond
                    else:
                        # --------------------------------------------------
                        # 3b) Coordinate monodentate ligand
                        # --------------------------------------------------
                        print (f"Monodentate O{i}")
                        dummy_atom_label = f"O{i}"
                        attach_atom, connect_atom = self.find_ligand_attachement_point(ligand, 
                                                connect_label=dummy_atom_label)
                        connect_idx = connect_atom.GetIdx()  # 1-based
                        
                        # --------------------------------------------------
                        # 3b.1. Delete dummy, merge, and ADD BOND
                        # --------------------------------------------------
                        # Delete dummy from template and fragment
                        dummy_atom, anchor_atom, anchor_idx = dummy_atom_dict[dummy_atom_label]
                        template.OBMol.DeleteAtom(dummy_atom.OBAtom)

                        # Merge fragment into template
                        template.OBMol += ligand.OBMol

                        # Now add bond: anchor (in template) → connect_atom (in fragment)
                        # After merge, fragment atoms start at: mol.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
                        frag_start_idx = template.OBMol.NumAtoms() - ligand.OBMol.NumAtoms() + 1
                        connect_idx_in_merged = frag_start_idx + connect_idx - 1

                        # Add bond (1-based indices)
                        template.OBMol.AddBond(anchor_idx, connect_idx_in_merged, 1)  # 1 = single bond
        
        # --------------------------------------------------
        # 5. Clean up: get rid of dummies * attached to ligand(s)
        # --------------------------------------------------
        for atom in template:
            print (f"New molecule Atom idx: {atom.idx}, atomic num: {atom.OBAtom.GetAtomicNum()}")
            if atom.OBAtom.GetAtomicNum() == 0:  # *
                print (f"Found leftover * at index {atom.idx}, marking for deletion")
                template.OBMol.DeleteAtom(atom.OBAtom)
    
        # --------------------------------------------------
        # 6. Generate 3D and optimize
        # --------------------------------------------------
        template.make3D(forcefield=force_field, 
                        steps=force_field_steps)

        # --------------------------------------------------
        # 7. Save
        # --------------------------------------------------
        template.write("sdf", output_sdf, overwrite=True)
        print(f"Success! Saved to {output_sdf}")
    
        if skip_count == 4:
            raise ValueError("No ligand is attached to the template because 0 dummy atoms were found.")

                            
        
        

        

        # TODO unhash from here
        #     # --------------------------------------------------
        #     # 3. Load ligand fragment and find attachment point
        #     # --------------------------------------------------
        #     # Load fragment and find attachment point

        #     attach_atom, connect_atom = self.find_ligand_attachement_point(ligand, 
        #                                      connect_label=dummy_atom_label)

        #     connect_idx = connect_atom.GetIdx()  # 1-based

        #     # --------------------------------------------------
        #     # 4. Delete dummy, merge, and ADD BOND
        #     # --------------------------------------------------
        #     # Delete dummy from template and fragment
        #     template.OBMol.DeleteAtom(dummy_atom.OBAtom)

        #     # Merge fragment into template
        #     template.OBMol += ligand.OBMol

        #     # Now add bond: anchor (in template) → connect_atom (in fragment)
        #     # After merge, fragment atoms start at: mol.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
        #     frag_start_idx = template.OBMol.NumAtoms() - ligand.OBMol.NumAtoms() + 1
        #     connect_idx_in_merged = frag_start_idx + connect_idx - 1

        #     # Add bond (1-based indices)
        #     template.OBMol.AddBond(anchor_idx, connect_idx_in_merged, 1)  # 1 = single bond

        # # --------------------------------------------------
        # # 5. Clean up: get rid of dummies * attached to ligand(s)
        # # --------------------------------------------------
        # for atom in template:
        #     print (f"New molecule Atom idx: {atom.idx}, atomic num: {atom.OBAtom.GetAtomicNum()}")
        #     if atom.OBAtom.GetAtomicNum() == 0:  # *
        #         print (f"Found leftover * at index {atom.idx}, marking for deletion")
        #         template.OBMol.DeleteAtom(atom.OBAtom)
    
        # # --------------------------------------------------
        # # 6. Generate 3D and optimize
        # # --------------------------------------------------
        # template.make3D(forcefield=force_field, 
        #                 steps=force_field_steps)

        # # --------------------------------------------------
        # # 7. Save
        # # --------------------------------------------------
        # template.write("sdf", output_sdf, overwrite=True)
        # print(f"Success! Saved to {output_sdf}")
    
        # if skip_count == 4:
        #     raise ValueError("No ligand is attached to the template because 0 dummy atoms were found.")
        # TODO unhash till here
            




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
