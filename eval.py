import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import skimage
from skimage.metrics import structural_similarity as ssim

import cmbNN_codes.functions as functions
import cmbNN_codes.models as models
import cmbNN_codes.training as training


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

modelArchitecture = None

EVAL_SPLIT = 0.05
BATCH_SIZE = 16


IMG_SIZE = 128
MAX_VAL = 1 # for psnr
DATA_RANGE = [0, MAX_VAL]

# ---------------------------------------------------------------------------------------------------------------------------------

class ModelAccessibilityError(ValueError):
    pass

class DataAccessibilityError(ValueError):
    pass

# ---------------------------------------------------------------------------------------------------------------------------------

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

def load_eval_data(data_file):
    if 1-EVAL_SPLIT != training.TRAIN_SPLIT:
        print("TRAIN_SPLIT not consistently defined. Continuing with eval.TRAIN_SPLIT.")
    print("EVAL_SPLIT: ", EVAL_SPLIT)
    if data_file[-3:] == "npz":
        data, sol, paras = functions.load_npz(data_file)
        print("Data loading from .npz file")
    elif data_file[-2:] == "h5":
        data, sol = functions.load_h5py(data_file)
        print("Data loading from .h5 file")
    else:
        raise training.DataFileFormatError(data_file)
    
    n_total = data.shape[0]
    n_eval = int(EVAL_SPLIT*n_total)

    data_eval = data[(n_total-n_eval):]
    sol_eval = sol[(n_total - n_eval):]

    data_tensor = torch.from_numpy(data_eval).float().unsqueeze(1)
    sol_tensor = torch.from_numpy(sol_eval).float().unsqueeze(1)
    dataset = TensorDataset(data_tensor, sol_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return dataloader

def eval_MSE(model = None, data = None, model_path = None, data_file = None, batch_size = 16, device = DEVICE):
    if model is None:
        if model_path is None:
            raise ModelAccessibilityError()
        else:
            model = load_model(model_path, device)
    if data is None:
        if data_file is None:
            raise DataAccessibilityError()
        else:
            dataloader = load_eval_data(data_file)
    else:
        dataloader = data
    mse_loss = nn.MSELoss(reduction="sum")
    total_squared_error = 0
    total_elements = 0
    single_image_mse = []
    with torch.no_grad():
        for x, y in dataloader:
            noisy_batch = x.to(device, non_blocking=True)
            clean_batch = y.to(device, non_blocking=True)
            preds = model(noisy_batch)

            batch_squared_error = mse_loss(preds, clean_batch).item()
            total_squared_error += batch_squared_error
            total_elements += clean_batch.numel()

            diff_squared = (preds - clean_batch) ** 2
            per_image = diff_squared.view(diff_squared.size(0), -1).mean(dim=1)
            single_image_mse.extend(per_image.cpu().numpy().tolist())

    overall_mse = total_squared_error / total_elements
    per_image_mse = np.array(per_image_mse)

    return overall_mse, per_image_mse

def eval_PSNR(model=None, data = None, model_path = None, data_file = None, max_val = MAX_VAL, batch_size = BATCH_SIZE, device = DEVICE):
    if model is None:
        if model_path is None:
            raise ModelAccessibilityError()
        else:
            model = load_model(model_path, device)
    if data is None:
        if data_file is None:
            raise DataAccessibilityError()
        else:
            dataloader = load_eval_data(data_file)
    else:
        dataloader = data
    single_image_mse = []
    with torch.no_grad():
        for x, y in dataloader:
            noisy_batch = x.to(device, non_blocking=True)
            clean_batch = y.to(device, non_blocking=True)

            preds = model(noisy_batch)

            diff_squared = (preds - clean_batch)**2
            diff_squared_per_image = diff_squared.view(diff_squared.size(0), -1).mean(dim=1)
            single_image_mse.extend(diff_squared_per_image.cpu().numpy().tolist())
    single_image_mse = np.array(single_image_mse)
    with np.errstate(divide="ignore"):      # make sure division does not break code
        single_image_psnr = 10 * np.log10((max_val ** 2) / single_image_mse)
    single_image_psnr = np.where(np.isinf(single_image_psnr), np.nan, single_image_psnr)     #handle infinity instances seperatly

    n_perfect = np.isnan(single_image_psnr).sum()
    if n_perfect > 0:
        print(f"Warning: {n_perfect} image(s) had MSE == 0 (infinite PSNR), excluded from statistics.")

    return single_image_psnr

def eval_ssim(model=None, data = None, model_path = None, data_file = None, data_range = DATA_RANGE, batch_size = BATCH_SIZE, device = DEVICE):
    if model is None:
        if model_path is None:
            raise ModelAccessibilityError()
        else:
            model = load_model(model_path, device)
    if data is None:
        if data_file is None:
            raise DataAccessibilityError()
        else:
            dataloader = load_eval_data(data_file)
    else:
        dataloader = data
    single_image_ssim = []
    with torch.no_grad():
        for x, y in dataloader:
            noisy_batch = x.to(device, non_blocking=True)

            preds = model(noisy_batch)

            if device == "DEVICE":
                preds_cpu = preds.cpu().numpy()
            for i in range(preds_cpu[0]):
                preds_img = preds_cpu[i, 0]
                x_img = x[i, 0]

                single_image_ssim.append(ssim(x_img, preds_img, data_range=max(data_range)))
    
    single_image_ssim = np.array(single_image_ssim)
    return single_image_ssim

def radial_power_spectrum(image):
    nx = image.shape[0]
    ny = image.shape[1]

    fft = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft)
    power2d = np.abs(fft_shifted)**2

    x, y = np.indices((nx, ny))
    center = (nx//2, ny//2)
    r = np.sqrt((x - center[0])**2 + (y-center[1])**2)
    r = r.astype(int)

    max_r = min(nx, ny)//2
    k_bins = np.arange(0, max_r)
    power = np.zeros(max_r)
    for k in k_bins:
        mask = (r == k)
        if mask.sum() > 0:
            power[k] = power2d[mask].mean()
        else:
            power[k] = np.nan
    return k_bins, power

def eval_power_spectrum(model = None, data = None, model_path = None, data_file = None, batch_size = BATCH_SIZE, device = DEVICE, img_size=IMG_SIZE):

    if model is None:
        if model_path is None:
            raise ModelAccessibilityError()
        else:
            model = load_model(model_path, device)
    if data is None:
        if data_file is None:
            raise DataAccessibilityError()
        else:
            dataloader = load_eval_data(data_file)
    else:
        dataloader = data

    max_r = img_size // 2
    residuals = []
    power_groundtruths = []
    power_preds = []
    with torch.no_grad():
        for x, y in dataloader:
            noisy_batch = noisy_batch.to(device, non_blocking=True)
            preds = model(noisy_batch)
            preds_np = preds.cpu().numpy()

            for i in range(preds_np.shape[0]):
                pred_img = preds_np[i, 0]
                y_img = y.numpy()[i, 0]

                k_bins, current_power_pred = radial_power_spectrum(pred_img)
                _, current_power_groundtruth = radial_power_spectrum(y_img)

                with np.errstate(divide="ignore", invalid="ignore"):
                    current_residual = (current_power_pred - current_power_groundtruth) / current_power_groundtruth
                current_residual = np.where(np.isfinite(current_residual, current_residual, np.nan))

                residuals.append(current_residual)
                power_groundtruths.append(current_power_groundtruth)
                power_preds.append(current_power_pred)

    residuals.append(current_residual)
    power_groundtruths.append(current_power_groundtruth)
    power_preds.append(current_power_pred)

    return k_bins, residuals, power_preds, power_groundtruths

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