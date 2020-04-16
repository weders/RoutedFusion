import os
import h5py

import numpy as np

from graphics.utils import extract_mesh_marching_cubes
from graphics.visualization import plot_mesh

with h5py.File('output/03001627.1015e71a0d21b127de03ab2a27ba7531.volume.hf5', 'r') as hf:
    scene = hf['TSDF'][:]
with h5py.File('output/03001627.1015e71a0d21b127de03ab2a27ba7531.weights.hf5', 'r') as hf:
    weights = hf['weights'][:]
with h5py.File('output/03001627.1015e71a0d21b127de03ab2a27ba7531.gt.hf5', 'r') as hf:
    groundtruth = hf['TSDF'][:]

mesh = extract_mesh_marching_cubes(scene)
plot_mesh(mesh)

mesh = extract_mesh_marching_cubes(groundtruth)
plot_mesh(mesh)
