# utils_atoms.py
from rdkit import Chem

def atom_indices_by_symbol(rdkit_mol, symbol):
    """Return list of atom indices matching element symbol."""
    return [a.GetIdx() for a in rdkit_mol.GetAtoms() if a.GetSymbol() == symbol]

def first_atom_by_symbols(rdkit_mol, symbols):
    """Return first atom index where atom symbol in symbols."""
    for a in rdkit_mol.GetAtoms():
        if a.GetSymbol() in symbols:
            return a.GetIdx()
    return None

from TM_catalyst_framework.utils_atoms import atom_indices_by_symbol

def get_placeholder_indices(mol: Chem.Mol, atom_symbol = None):
    """Return dictionary mapping placeholder symbols to their atom indices."""
    return atom_indices_by_symbol(mol, atom_symbol)[0]
