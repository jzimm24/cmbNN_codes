import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import skimage
from skimage.metrics import structural_similarity as ssim

import functions as functions
import models as models
import training as training


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

modelArchitecture = "UNET"
MODEL_FILE = "../models/shortUnetRun_epoch_3.pth"
DATA_FILE = training.DATA_FILE
OUTPUT_FILE = "test1508.png"

EVAL_SPLIT = 0.05
BATCH_SIZE = 16


IMG_SIZE = 128
MAX_VAL = 1 # for psnr
DATA_RANGE = [0, MAX_VAL]

TITLE = OUTPUT_FILE

# ---------------------------------------------------------------------------------------------------------------------------------

class ModelAccessibilityError(ValueError):
    pass

class DataAccessibilityError(ValueError):
    pass

# ---------------------------------------------------------------------------------------------------------------------------------

def load_model(path: str = MODEL_FILE, device: str = DEVICE):
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

def load_random_sample(path: str = DATA_FILE, seed = 42):       #TODO: load random sample from part of the dataloader that was saved for evaluation
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

def example_forward_pass(model, sample = None, path = DATA_FILE):
    if sample is None:
        sample = load_random_sample(path)
    sample_on_device = sample[0].to(DEVICE)
    output = model(sample_on_device)
    ground_truth = sample[1].to(DEVICE)
    return sample_on_device, output, ground_truth

def make_figures(maps, save_path: str = "../outputs/" + OUTPUT_FILE, title: str = TITLE):
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

def eval_MSE(model = None, data = None, model_file = MODEL_FILE, data_file = DATA_FILE, batch_size = 16, device = DEVICE):
    if model is None:
        if model_file is None:
            raise ModelAccessibilityError()
        else:
            model = load_model(model_file, device)
    model.eval()
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
    single_image_mse = np.array(single_image_mse)

    return overall_mse, single_image_mse

def eval_PSNR(model=None, data = None, model_file = MODEL_FILE, data_file = DATA_FILE, max_val = MAX_VAL, batch_size = BATCH_SIZE, device = DEVICE):
    if model is None:
        if model_file is None:
            raise ModelAccessibilityError()
        else:
            model = load_model(model_file, device)
    model.eval()
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

def eval_ssim(model=None, data = None, model_file = MODEL_FILE, data_file = DATA_FILE, data_range = DATA_RANGE, batch_size = BATCH_SIZE, device = DEVICE):
    if model is None:
        if model_file is None:
            raise ModelAccessibilityError()
        else:
            model = load_model(model_file, device)
    model.eval()
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

            if device == DEVICE:
                preds_cpu = preds.cpu().numpy()
            for i in range(preds_cpu.shape[0]):
                preds_img = preds_cpu[i, 0]
                x_img = x[i, 0].numpy()

                single_image_ssim.append(ssim(x_img, preds_img, data_range=max(data_range)))
    
    single_image_ssim = np.array(single_image_ssim)
    return single_image_ssim

def radial_power_spectrum(image):
    nx, ny = image.shape

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

def cross_correlation_coefficient(groundtruth_img, pred_img):
    nx, ny = groundtruth_img.shape

    groundtruth_fft = np.fft.fftshift(np.fft.fft2(groundtruth_img))
    pred_fft = np.fft.fftshift(np.fft.fft2(pred_img))

    groundtruth_power2d = np.abs(groundtruth_fft)**2
    pred_power2d = np.abs(pred_fft)**2

    cross_power2d = np.real(groundtruth_fft * np.conj(pred_fft))

    r, r_max = functions.radial_bin_indices(ny, nx)
    k_bins = np.arange(0, r_max)

    groundtruth_power_k = np.zeros(r_max)
    pred_power_k = np.zeros(r_max)
    cross_power_k = np.zeros(r_max)

    for k in k_bins:
        mask = (r == k)
        if mask.sum():
            groundtruth_power_k[k] = groundtruth_power2d[mask].mean()
            pred_power_k[k] = pred_power2d[mask].mean()
            cross_power_k[k] = cross_power2d[mask].mean()
        else:
            groundtruth_power_k[k] = np.nan
            pred_power_k[k] = np.nan
            cross_power_k[k] = np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        r_k = cross_power_k / np.sqrt(groundtruth_power_k * pred_power_k)
        r_k = np.where(np.isfinite(r_k), r_k, np.nan)

    return k_bins, r_k

def main():
    model = load_model()
    model.eval()
    noisy, clean = load_random_sample()
    noisy, prediction, clean = example_forward_pass(model, [noisy, clean])
    make_figures([noisy, prediction, clean])
    #
    
    #  total_mse, mse_list = eval_MSE(model_file="../models/0507_unetTest_noWN_A7_epoch_1.pth", data_file="../data/noWN_A7_1F-unsmooth_5k128pix_lin04.h5")
    # print("Total MSE: ", total_mse)
    # print("Max MSE: ", max(mse_list))
    # print("#########################################################")
    
    # psnr_list = eval_PSNR(model_file="../models/0507_unetTest_noWN_A7_epoch_1.pth", data_file="../data/noWN_A7_1F-unsmooth_5k128pix_lin04.h5")
    # print("PSNR Max: ", max(psnr_list))
    # print("PSNR Min: ", min(psnr_list))
    
    # ssim_list = eval_ssim(model_file="../models/0507_unetTest_noWN_A7_epoch_1.pth", data_file="../data/noWN_A7_1F-unsmooth_5k128pix_lin04.h5")
    # print("SSIM Max: ", max(ssim_list))
    # print("SSIM Min: ", min(ssim_list))
    # print("SSIM Avg: ", np.mean(ssim_list))
    # return None

if __name__ == "__main__":
    print("Executing main() in eval.py")
    main()