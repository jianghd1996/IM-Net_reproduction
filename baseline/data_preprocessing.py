import os
import h5py
import numpy as np
from tqdm import tqdm
import random

if __name__ == "__main__":
    category = "Chair"
    filename = os.listdir(os.path.join("../data", category))
    random.shuffle(filename)
    resolution = [8, 16, 32, 64]
    for file in tqdm(filename):
        x = file.split('.')[0]
        with h5py.File(os.path.join("data", category, file), "r") as f:
            raw_voxel = np.array(f["shape_voxel64"], dtype="float32")

        if os.path.exists(os.path.join("processed_data", category, x + ".npy")):
            continue

        data = dict()
        for res in resolution:
            voxel = np.zeros((res, res, res), dtype="float32")
            length = 64 // res
            for i in range(res):
                for j in range(res):
                    for k in range(res):
                        voxel[i][j][k] = np.max(raw_voxel[i*length:(i+1)*length, j*length:(j+1)*length, k*length:(k+1)*length]) > 0
            points = []
            for i in range(res):
                for j in range(res):
                    for k in range(res):
                        lx = max(0, i-2)
                        rx = min(i+2, res)
                        ly = max(0, j-2)
                        ry = max(j+2, res)
                        lz = max(0, k-2)
                        rz = max(k+2, res)
                        local_vox = voxel[lx:rx, ly:ry, lz:rz]
                        if np.max(local_vox) != np.min(local_vox):
                            points.append([i * length, j * length, k * length])
            data[res] = [voxel, points]

        np.save(os.path.join("processed_data", category, x), data)

