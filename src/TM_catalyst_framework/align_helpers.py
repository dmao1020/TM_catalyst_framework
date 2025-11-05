# align_helpers.py
import numpy as np
from rdkit import Chem

def get_conf_positions(mol):
    conf = mol.GetConformer()
    n = mol.GetNumAtoms()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(n)])
    return coords

def set_conf_positions(mol, coords):
    conf = mol.GetConformer()
    for i, pos in enumerate(coords):
        conf.SetAtomPosition(i, tuple(pos))

def translate_coords(coords, translation):
    return coords + translation

def rotation_matrix_align_vectors(v_from, v_to):
    """
    Return rotation matrix that aligns v_from to v_to (both numpy arrays).
    Uses Rodrigues' rotation formula.
    """
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)
    cross = np.cross(v_from, v_to)
    dot = np.dot(v_from, v_to)
    if np.allclose(cross, 0) and dot > 0.9999:
        return np.eye(3)
    if np.allclose(cross, 0) and dot < -0.9999:
        # 180 degree rotation: pick arbitrary orthogonal axis
        axis = np.array([1.0, 0.0, 0.0])
        if np.allclose(v_from, axis):
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - axis.dot(v_from) * v_from
        axis = axis / np.linalg.norm(axis)
        theta = np.pi
        K = np.array([[0, -axis[2], axis[1]],[axis[2], 0, -axis[0]],[-axis[1], axis[0], 0]])
        return np.eye(3) + np.sin(theta)*K + (1 - np.cos(theta))*(K@K)
    K = np.array([[0, -cross[2], cross[1]],[cross[2], 0, -cross[0]],[-cross[1], cross[0], 0]])
    s = np.linalg.norm(cross)
    R = np.eye(3) + K + K@K * ((1 - dot)/(s**2))
    return R
