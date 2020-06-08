import os
import h5py
import numpy as np
from tqdm import tqdm
import random

if __name__ == "__main__":
    category = "Chair"
    filename = os.listdir(os.path.join("../data", category))
    random.shuffle(filename)
    resolution = [16, 32, 64]
    for file in tqdm(filename):
        x = file.split('.')[0]
        with h5py.File(os.path.join("../data", category, file), "r") as f:
            raw_voxel = np.array(f["shape_voxel64"], dtype="float32")

        if os.path.exists(os.path.join("../processed_data", category, x + ".npy")):
            continue

        data = dict()
        for res in resolution:
            voxel = np.zeros((res, res, res), dtype="float32")
            length = 64 // res
            for i in range(res):
                for j in range(res):
                    for k in range(res):
                        voxel[i][j][k] = np.max(raw_voxel[i*length:(i+1)*length, j*length:(j+1)*length, k*length:(k+1)*length]) > 0
            voxel = np.repeat(voxel, length, axis=0)
            voxel = np.repeat(voxel, length, axis=1)
            voxel = np.repeat(voxel, length, axis=2)
            points = []
            for i in range(64):
                for j in range(64):
                    for k in range(64):
                        lx = max(0, i-2)
                        rx = min(i+2, 64)
                        ly = max(0, j-2)
                        ry = min(j+2, 64)
                        lz = max(0, k-2)
                        rz = min(k+2, 64)
                        local_vox = voxel[lx:rx, ly:ry, lz:rz]
                        if np.max(local_vox) != np.min(local_vox):
                            points.append([i, j, k])
            data[res] = [voxel, points]

        np.save(os.path.join("../processed_data", category, x), data)

