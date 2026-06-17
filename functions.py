import numpy as np
import pandas as pd


def load_npz(path: str):
    data = np.load(path, allow_pickle=True)
    print(f"Keys in {path}: {list(data.keys())}")
 
    X = data["maps"]
    Y = data["ground_truths"]
    y = np.column_stack([data["radii"], data["cx_shifts"], data["cy_shifts"]])
    
    print(f"Loaded  X: {X.shape}  Y: {Y.shape}  y: {y.shape}")
    return X, Y, y