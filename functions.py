import numpy as np
import pandas as pd
import h5py


def load_npz(path: str):
    data = np.load(path, allow_pickle=True)
    print(f"Keys in {path}: {list(data.keys())}")
 
    X = data["maps"]
    Y = data["ground_truths"]
    y = np.column_stack([data["radii"], data["cx_shifts"], data["cy_shifts"]])
    
    print(f"Loaded  X: {X.shape}  Y: {Y.shape}  y: {y.shape}")
    return X, Y, y

def load_h5py(path: str):
    with h5py.File(path, "r") as f:
        X = f["noisy"][:]
        Y = f["clean"][:]
    return X, Y

def radial_bin_indices(nx, ny):
    x, y = np.indices((nx, ny))
    center = (nx//2, ny//2)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    max_r = min(nx, ny) // 2
    return r, max_r