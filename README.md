# Ti-Ureate Framework

A Python-based framework for constructing and automating Titanium Ureate catalyst complexes.

## Features
- Define ligands using SMILES or 3D structures
- Build Ti(IV) complexes with variable coordination (4 or 5)
- Automate geometry assembly and export for ORCA/DFT
- Generate force-field optimized geometries via Open Babel

## Installation
```bash
git clone https://github.com/dmao1020/TM_catalyst_framework
cd TM_catalyst_framework
pip install -e .
```

## Troubleshooting
A common installation issue arises from the Open Babel dependency, particularly when installing via pip.
The recommended solution is to install Open Babel using conda-forge:
```bash
conda install -c conda-forge openbabel
```
After installing Open Babel, reinstall the package if necessary following the instructions above.