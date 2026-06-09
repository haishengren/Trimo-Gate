import os
import re
import torch
import numpy as np
import pandas as pd


DATA_PATH = r"QM9_dataset_with_U0_atom.xlsx"
# XYZ_DIR = r"/multimodal_modeling/QM9/QM9_xyz"
# XYZ_DIR = r"/workspace/scu/QM9/QM9"
XYZ_DIR = r"/data/chuanda/QM9_cmcr/QM9"
print("QM9 3D loader: xyz coordinates + .."
      ".xyz atom one-hot only; RDKit is not used.")


def fortran_float(value):
    """Convert QM9 Fortran-style floats, such as 2.1997*^-6, to Python floats."""
    return re.sub(r"(\d+\.?\d*)\*\^([+-]?\d+)", r"\1e\2", value)


def parse_qm9_xyz_and_center(file_path):
    """Read QM9 xyz atoms and return centered coordinates in the xyz atom order."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    n_atom = int(lines[0].strip())
    elements = []
    raw_coords = []

    for line in lines[2: 2 + n_atom]:
        parts = line.split()
        elements.append(parts[0])
        raw_coords.append([
            float(fortran_float(parts[1])),
            float(fortran_float(parts[2])),
            float(fortran_float(parts[3])),
        ])

    raw_coords = np.asarray(raw_coords, dtype=np.float32)
    centroid = np.mean(raw_coords, axis=0)
    centered_coords = raw_coords - centroid
    return centered_coords, elements


df = pd.read_excel(DATA_PATH)
smiles_list = df.iloc[:, 0]
total_files = len(smiles_list)


# Build atom vocabulary directly from xyz element symbols. This does not use
# RDKit and keeps the one-hot atom type aligned with the xyz atom order.
all_elements = []
for idx in range(1, total_files + 1):
    filename = f"dsgdb9nsd_{idx:06d}.xyz"
    filepath = os.path.join(XYZ_DIR, filename)
    try:
        _, xyz_elements = parse_qm9_xyz_and_center(filepath)
    except Exception as exc:
        print(f"Error reading element types from {filename}: {exc}")
        continue
    all_elements.extend(xyz_elements)

element_list = sorted(set(all_elements))
atom_type_to_idx = {atom: idx for idx, atom in enumerate(element_list)}
print(f"element_list: {element_list}")


point_clouds = []
point_cloud_lengths = []

for idx in range(1, total_files + 1):
    filename = f"dsgdb9nsd_{idx:06d}.xyz"
    filepath = os.path.join(XYZ_DIR, filename)

    try:
        coords, xyz_elements = parse_qm9_xyz_and_center(filepath)
    except Exception as exc:
        print(f"Error processing {filename}: {exc}")
        continue

    n_atoms = len(xyz_elements)

    #  Element Type
    atom_type_onehot = np.zeros((n_atoms, len(element_list)), dtype=np.float32)
    for atom_idx, symbol in enumerate(xyz_elements):
        type_idx = atom_type_to_idx.get(symbol)
        if type_idx is None:
            raise ValueError(f"Unknown atom type {symbol} in {filename}")
        atom_type_onehot[atom_idx, type_idx] = 1.0

    # Radial distance from the molecular centroid, Since Coordinates have already been centered, rho_i = ||r_i^c||_2.
    rho = np.linalg.norm(coords, axis=1, keepdims=True).astype(np.float32)

    point_cloud_feature = np.concatenate([coords.astype(np.float32), rho, atom_type_onehot], axis=1,)

    point_clouds.append(point_cloud_feature)
    point_cloud_lengths.append(n_atoms)

    if idx % 10000 == 0:
        print(f"Processed {idx} files...")

max_atoms = max(point_cloud_lengths)
d_input = point_clouds[0].shape[1]
print(f"d_input = {d_input}, type: {type(d_input)}")
print(f"Maximum atoms in dataset: {max_atoms}")
print(f"Feature dimension per atom: {d_input}")
print(f"Total molecules processed: {len(point_clouds)}")

padded_point_clouds = []
for pc, length in zip(point_clouds, point_cloud_lengths):
    if length < max_atoms:
        pad_size = max_atoms - length
        padding = np.zeros((pad_size, pc.shape[1]), dtype=np.float32)
        padded_pc = np.vstack([pc, padding])
    else:
        padded_pc = pc
    padded_point_clouds.append(padded_pc)

point_cloud = [torch.tensor(pc, dtype=torch.float32) for pc in padded_point_clouds]
point_cloud_lengths = torch.tensor(point_cloud_lengths, dtype=torch.long)
