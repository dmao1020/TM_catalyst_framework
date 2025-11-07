from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
import openbabel
import pybel
from pathlib import Path
from typing import List, Optional
from TM_catalyst_framework.ligands import Ligand
import numpy as np
from openbabel import pybel
from importlib import resources
from TM_catalyst_framework.metal_template import load_ti_template_sdf
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
        substituent_dummy_map = {
            "R1": 9,   # Fluorine
            "R2": 17,  # Chlorine
            "R3": 35   # Bromine
        }
        # --------------------------------------------------------------
        # 1. Load the template (keep hydrogens if you need them)
        # --------------------------------------------------------------
        
        template = load_ti_template_sdf(template_sdf, package = "openbabel",
                                        template_dir = "TM_catalyst_framework.template")
        if template is None:
            raise ValueError("Could not read the template SDF file.")
        
        
        for substituent_label, fragment_smiles in fragment_smiles_dict.items():
            dummy_atomic_num = substituent_dummy_map.get(substituent_label)
            # --------------------------------------------------
            # 2. Replace dummy F with fragment
            # --------------------------------------------------
            # Find dummy and anchor in template
            dummy_atom, anchor_atom = self.find_dummy_and_anchor(template, dummy_atomic_num)
            anchor_idx = anchor_atom.idx  # 1-based Open Babel index

            # --------------------------------------------------
            # 3. Load fragment and find attachment point
            # --------------------------------------------------
            # Load fragment and find attachment point
            frag = pybel.readstring("smi", fragment_smiles)
            attach_atom, connect_atom = self.find_fragment_attachment_point(frag)
            connect_idx = connect_atom.GetIdx()  # 1-based

            # --------------------------------------------------
            # 4. Delete dummy, merge, and ADD BOND
            # --------------------------------------------------
            # Delete dummy from template and fragment
            template.OBMol.DeleteAtom(dummy_atom.OBAtom)

            # Merge fragment into template
            template.OBMol += frag.OBMol

            # Now add bond: anchor (in template) → connect_atom (in fragment)
            # After merge, fragment atoms start at: mol.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
            frag_start_idx = template.OBMol.NumAtoms() - frag.OBMol.NumAtoms() + 1
            connect_idx_in_merged = frag_start_idx + connect_idx - 1

            # Add bond (1-based indices)
            template.OBMol.AddBond(anchor_idx, connect_idx_in_merged, 1)  # 1 = single bond

            #cleanup: check if there's any remaining * atoms and remove them
            delete_indices = []
            for atom in template:
                print (f"New molecule Atom idx: {atom.idx}, atomic num: {atom.OBAtom.GetAtomicNum()}")
                if atom.OBAtom.GetAtomicNum() == 0:  # *
                    print (f"Found leftover * at index {atom.idx}, marking for deletion")
                    template.OBMol.DeleteAtom(atom.OBAtom)
            print (f"Fragment attachment at index {attach_atom.idx}")
        
        # --------------------------------------------------
        # 5. Generate 3D and optimize
        # --------------------------------------------------
        template.make3D(forcefield=force_field, 
                        steps=force_field_steps)

        # --------------------------------------------------
        # 6. Save
        # --------------------------------------------------
        template.write("sdf", output_sdf, overwrite=True)
        print(f"Success! Saved to {output_sdf}")
        
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

        # --------------------------------------------------------------
        # 1. Load the template (keep hydrogens if you need them)
        # --------------------------------------------------------------

        template = load_ti_template_sdf(template_sdf, 
                                        package = "openbabel", 
                                        template_dir = "TM_catalyst_framework.metal_center_template")
        if template is None:
            raise ValueError("Could not read the template SDF file.")
        
        skip_count = 0
        ligand = pybel.readstring("smi", ligand_smiles)
        dummy_atom_dict = {}

        for dummy_atom_label, dummy_atomic_num in metal_coord_dummy_map.items():
            # --------------------------------------------------
            # 2.1. Check if template contains dummy atom
            # --------------------------------------------------

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
        # If the complex is mono-ligated
        # --------------------------------------------------
        if num_ligands == 1:
            
            

        

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
