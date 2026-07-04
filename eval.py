import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py

import cmbNN_codes.functions as functions
import cmbNN_codes.models as models
import cmbNN_codes.training as training


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

modelArchitecture = None



def load_model(path: str = "../models/model_epoch_3.pth", device = DEVICE):
    if "unet" in path or "UNET" in path or modelArchitecture == "UNET":
        model, _ = training.init_UNET_model()
        print("Loading Model of UNET-architecture")
    elif "gan" in path or "GAN" in path or modelArchitecture == "GAN":
        model, _, _, _ = training.init_GAN_models()
        print("Loading Model of GAN-architecture")
    else:
        print("Model path does not specify model architecture")
        raise training.ModelArchitectureError()
    model.to(device)
    last_model_state = torch.load(path, map_location=device)
    model.load_state_dict(last_model_state["generator_state_dict"])

    print(f"Loaded model from epoch {last_model_state['epoch']} onto {DEVICE}")
    if "gan" in path or "GAN" in path or modelArchitecture == "GAN":
        print(f"  loss_G at save time: {last_model_state['loss_G']}")
        print(f"  loss_D at save time: {last_model_state['loss_D']}")

    return model

def load_random_sample(path: str, seed = 42):       #TODO: load random sample from part of the dataloader that was saved for evaluation
    np.random.seed(seed)
    with h5py.File(path, "r") as f:
        dataset = f["noisy"]
        num_maps = dataset.shape[0]
        idx = np.random.randint(0, num_maps)
        sample = dataset[idx]

        dataset_clean = f["clean"]
        sample_clean = dataset_clean[idx]

        print(f"Sample [{idx}] loaded.")

    tensor_noisy = torch.from_numpy(sample).float().unsqueeze(0).unsqueeze(0)
    tensor_clean = torch.from_numpy(sample_clean).float().unsqueeze(0).unsqueeze(0)
    return tensor_noisy, tensor_clean

def example_forward_pass(model, sample = None, path = "/home/user/Physik_Bonn/master_thesis/codes/cmbNN/data/NEW-Dset_1F-unsmooth_1k128pix_lin04.h5"):
    if sample is None:
        sample = load_random_sample(path)
    sample_on_device = sample[0].to(DEVICE)
    output = model(sample_on_device)
    ground_truth = sample[1].to(DEVICE)
    return sample_on_device, output, ground_truth

def make_figures(maps, save_path, title):
    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        img = maps[i]

        # If it's a torch tensor, convert to numpy and squeeze extra dims
        if hasattr(img, "detach"):
            img = img.detach().cpu().numpy()
        img = img.squeeze()

        im = ax.imshow(img, cmap="viridis")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.title(title)

    plt.tight_layout()

    fig.savefig(save_path, bbox_inches="tight")
    print(f"Figure saved to: {save_path}")

    plt.close(fig)
    return None



def main():
    model = load_model(path="../models/0507_unetTest_noWN_A7_epoch_1.pth")
    model.eval()
    noisy, clean = load_random_sample("../data/noWN_A7_1F-unsmooth_5k128pix_lin04.h5")
    noisy, prediction, clean = example_forward_pass(model, [noisy, clean])
    make_figures([noisy, prediction, clean], "../outputs/0507test_noWN_A7_unet_fig_unet.png", "0507_test_images_unet")
    return None

if __name__ == "__main__":
    print("Executing main() in eval.py")
    main()