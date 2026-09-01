import numpy as np
import pandas as pd
import h5py

import datetime

import config


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

def normalize_data(data, epsilon = 1e-8):
    data_min = data.min(axis = (1, 2), keepdims = True)
    data_max = data.max(axis = (1, 2), keepdims = True)
    data = (data - data_min) / (data_max - data_min + epsilon)
    amp = data_max - data_min + epsilon
    return data, amp, data_min

def normalize_map(map, epsilon = 1e-8):
    map_min = map.min()
    map_max = map.max()
    map = (map - map_min) / (map_max - map_min + epsilon)
    amp = map_max - map_min + epsilon
    return map, amp, map_min

def radial_bin_indices(nx, ny):
    x, y = np.indices((nx, ny))
    center = (nx//2, ny//2)
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(int)
    max_r = min(nx, ny) // 2
    return r, max_r

def write_doc(run_type: str,
              doc_path: str = None,
              runtime = 0, 
              output_file: str = None
              ):
    
    lines = []
    lines.append("=" * 60)
    lines.append("DOCUMENTATION")
    lines.append("=" * 60)
    lines.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    lines.append("=" * 60)
    lines.append(f"RUNTIME: {runtime}")
    
    if run_type == "map_making":
            
        config_vars = {
            "run_type": run_type,
            "output_file": output_file,
            "save_mode": config.maps_configs.save_mode,
            "flux_mode": config.maps_configs.flux_mode,
            "number_of_images": config.maps_configs.num_images,
            "IMG_SIZE": config.maps_configs.image_size,
            "max_number_of_sources": config.maps_configs.max_source_number,
            "randamization_seed": config.maps_configs.seed,
            "rc_mean": config.maps_configs.rc_mean,
            "rc_sigma": config.maps_configs.rc_sigma,
            "I0_range": config.maps_configs.I0_range,
            "I0_fixed": config.maps_configs.I0_fixed,
            "I0_mean": config.maps_configs.I0_mean,
            "I0_sigma": config.maps_configs.I0_sigma,
            "white_noise_amplitude": config.maps_configs.white_noise_amplitude,
            "one_over_f_slope": config.maps_configs.one_over_f_slope,
            "one_over_f_amplitude": config.maps_configs.one_over_f_amplitude,
            "gaussian_smoothing_fwhm": config.maps_configs.gaussian_smoothing_fwhm,
            "output_h5": config.maps_configs.output_h5
        }
        doc_path = config.maps_configs.doc_path

    elif run_type == "training":

        config_vars = {
            "run_type": run_type,
            "output_file": output_file,
            "device": config.training_configs.DEVICE,
            "model": config.training_configs.model,
            "number_of_epochs": config.training_configs.NUM_EPOCHS,
            "batch_size": config.training_configs.BATCH_SIZE,
            "learning_rate": config.training_configs.LR,
            "image_dims": config.training_configs.IMG_DIM,
            "data_file": config.training_configs.DATA_FILE,
            "training_data_split": config.training_configs.TRAIN_SPLIT,
            "model_name": config.training_configs.trained_model_name,
            "criterium": config.training_configs.crit,
        }
        doc_path = config.training_configs.doc_path

    elif run_type == "evaluation":

        config_vars = {
            "run_type": run_type,
            "output_file": output_file,
            "device": config.evaluation_configs.DEVICE,
            "model_architecure": config.evaluation_configs.modelArchitecture,
            "model_file": config.evaluation_configs.MODEL_FILE,
            "data_file": config.evaluation_configs.DATA_FILE,
            "evaluation_split": config.evaluation_configs.EVAL_SPLIT,
            "batch_size": config.evaluation_configs.BATCH_SIZE,
            "image_size": config.evaluation_configs.IMG_SIZE,
            "max_value": config.evaluation_configs.MAX_VAL,
            "data_range": config.evaluation_configs.DATA_RANGE,
            "title": config.evaluation_configs.TITLE,
        }
        doc_path = config.evaluation_configs.doc_path

    else:
        print(f"The given step ({run_type}) does not fit the options (map_making, training, evaluation).",
              "The full config file is printed instead")
        
    for name, value in config_vars.items():
                lines.append(f"{name} = {value!r}")
            
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Documentation written to:", doc_path)
    return None