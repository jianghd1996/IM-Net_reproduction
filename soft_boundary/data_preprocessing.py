import os
import h5py
import numpy as np
from tqdm import tqdm
import random

if __name__ == "__main__":
    category = "Chair"
    filename = os.listdir(os.path.join("../data", category))
    random.shuffle(filename)
    for file in tqdm(filename):
        x = file.split('.')[0]
        with h5py.File(os.path.join("../data", category, file), "r") as f:
            raw_voxel = np.array(f["shape_voxel64"], dtype="float32")

        if os.path.exists(os.path.join("../soft_data", category, x + ".npy")):
            continue

        voxel = np.zeros((64, 64, 64), dtype="float32")
        for i in range(64):
            for j in range(64):
                for k in range(64):
                    voxel[i][j][k] = raw_voxel[i][j][k] > 0

        weight = np.zeros((64, 64, 64), dtype="float32")

        for i in range(64):
            for j in range(64):
                for k in range(64):
                    for delt in range(3, 0, -1):
                        lx = max(0, i - delt)
                        rx = min(i + delt+1, 64)
                        ly = max(0, j - delt)
                        ry = min(j + delt+1, 64)
                        lz = max(0, k - delt)
                        rz = min(k + delt+1, 64)
                        local_vox = voxel[lx:rx, ly:ry, lz:rz]
                        if np.max(local_vox) != np.min(local_vox):
                            weight[i][j][k] = delt
                        else:
                            break
        points = []
        for i in range(64):
            for j in range(64):
                for k in range(64):
                    if weight[i][j][k] != 0:
                        points.append([(i, j, k), weight[i][j][k]])

        np.save(os.path.join("../soft_data", category, x), [voxel, points])

