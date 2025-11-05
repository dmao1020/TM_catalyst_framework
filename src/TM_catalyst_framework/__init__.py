"""
TM_catalyst_framework
======================

A modular Python framework for constructing and automating
transition-metal (TM) catalyst complexes. Designed for integration
with RDKit, ASE, and Open Babel for structure generation,
optimization, and export to quantum-chemistry codes such as ORCA.

Author: Dawn Mao
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Dawn Mao"
__email__ = "your_email@example.com"

# Re-export key classes for convenience
from .ligands import Ligand
from .metal_center import MetalCenter
from .complex import Complex
from .io_utils import ORCA

# Optional: define what shows up under *
__all__ = ["Ligand", "MetalCenter", "ComplexBuilder", "ORCA"]
