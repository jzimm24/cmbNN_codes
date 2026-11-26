import numpy as np
import matplotlib.pyplot as plt
import re
import io

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import h5py
import skimage
from skimage.metrics import structural_similarity as ssim
import time

import functions as functions
import models as models
import training as training

import config as config


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

modelArchitecture = config.evaluation_configs.modelArchitecture
MODEL_FILE = config.evaluation_configs.MODEL_FILE
DATA_FILE = config.evaluation_configs.DATA_FILE
OUTPUT_FILE = config.evaluation_configs.OUTPUT_FILE

EVAL_SPLIT = config.evaluation_configs.EVAL_SPLIT
BATCH_SIZE = config.evaluation_configs.BATCH_SIZE


pixel_n = config.evaluation_configs.pixel_n
IMG_SIZE = config.evaluation_configs.img_size
PXL_SIZE = config.evaluation_configs.pixel_size
MAX_VAL = config.evaluation_configs.MAX_VAL                     # for psnr
DATA_RANGE = config.evaluation_configs.DATA_RANGE

EVAL_FILE = config.evaluation_configs.EVAL_FILE
TITLE = config.evaluation_configs.TITLE

# ---------------------------------------------------------------------------------------------------------------------------------

class ModelAccessibilityError(ValueError):
    pass

class DataAccessibilityError(ValueError):
    pass

class UnknownFileStructureError(ValueError):
    pass

# ---------------------------------------------------------------------------------------------------------------------------------

def load_model(path: str = MODEL_FILE, device: str = DEVICE):
    """
        Function loads trained model for later evaluation. Model architecture is
        determined by config.py variable and model parameters are taken from checkpoint file, therefor
        the two models need to be identical.
    
        Parameters
        ----------
        path (str): file of the saved model with the checkpoint data
        device:
    
        Returns
        -------
        model:
        """

    denorm_vals_noisy = None
    denorm_vals_sol = None
    if  modelArchitecture == "UNET":
        model, _ = training.init_UNET_model()
        print("Loading Model of UNET-architecture.")
        model.to(device)
        last_model_state = torch.load(path, map_location=device)
        model.load_state_dict(last_model_state["model_state_dict"])
    elif "gan" in path or "GAN" in path or modelArchitecture == "GAN":
        model = training.init_GAN_model()
        print("Loading Model of GAN-architecture.")
        model.to(device)
        last_model_state = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(last_model_state["generator_state_dict"], load_optimizers=False)
        model.eval()
        denorm_vals_noisy = last_model_state["denorm_vals_noisy"]
        denorm_vals_sol = last_model_state["denorm_vals_sol"]
    elif "resnet" in path or modelArchitecture == "ResUNET":
        model, _ = training.init_ResUNET_model()
        print("Loading Model of ResUNET-architecture.")  
        model.to(device)
        last_model_state = torch.load(path, map_location=device)

        # print("path:", path)
        # print("modelArchitecture:", modelArchitecture)
        # print("keys:", list(last_model_state.keys()))

        model.load_state_dict(last_model_state["model_state_dict"])  
    else:
        print("Model path does not specify model architecture")
        raise training.ModelArchitectureError()
    if modelArchitecture == "GAN":
        model = model.generator
    print("Model loaded!")

    return model, denorm_vals_noisy, denorm_vals_sol

def load_and_prep_eval_data(data_file, epsilon = 1e-8):
    """
        Function loads images and norms the data to fit [0, 1].
    
        Parameters
        ----------
        data_file: file of data
        epsilon: small constant preventing division by zero in normalization
    
        Returns
        -------
        dataloader (dataloader):    image data wrapped in dataloader. Only the data for evaluation has been kept.
                                    Propably final 5%-10%
        data_eval_normalisationParas (list[list[floats]]): values necessary to denormalise the noisy data ([amplitude, minimum])
        sol_eval_normalisationParas (list[list[floats]]): values necessary to denormalise the clean data ([amplitude, minimum])
        """
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

    data_eval_normalized, data_eval_denorm_amp, data_eval_denorm_min = functions.normalize_data_gloabally(data_eval)
    sol_eval_normalized, sol_eval_denorm_amp, sol_eval_denorm_min = functions.normalize_data_gloabally(sol_eval)
    data_denorm_vals = [data_eval_denorm_amp, data_eval_denorm_min]
    sol_denorm_vals = [sol_eval_denorm_amp, sol_eval_denorm_min]

    data_tensor = torch.from_numpy(data_eval_normalized).float().unsqueeze(1)
    sol_tensor = torch.from_numpy(sol_eval_normalized).float().unsqueeze(1)
    dataset = TensorDataset(data_tensor, sol_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    return dataloader, data_denorm_vals, sol_denorm_vals

def load_random_samples(path: str, n = 1, seed=42, epsilon=1e-8):
    dataloader, data_paras, sol_paras = load_and_prep_eval_data(path, epsilon)

    dataset = dataloader.dataset          # TensorDataset of (noisy, clean) pairs

    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(dataset), size=n, replace=False)
    print(f"Randomly chosen index (within eval split): {idxs} of {len(dataset)}.")

    tensor_noisy, tensor_clean = dataset[idxs]   # each has shape (1, H, W)
    print(f"Sample shape: {tuple(tensor_noisy.shape)}")

    # add batch dimension -> (1, 1, H, W), same as the old function returned
    return tensor_noisy.unsqueeze(0), tensor_clean.unsqueeze(0)

def example_forward_pass(model, sample=None, path=DATA_FILE):
    if sample is None:
        sample = load_random_samples(path)
    sample_on_device = sample[0].squeeze(0).to(DEVICE)
    ground_truth = sample[1].squeeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        output = model(sample_on_device)
    return sample_on_device, output, ground_truth

def make_figures(
    maps=None,
    model=None,
    n: int = 1,
    data_path: str = DATA_FILE,
    save_path: str = "../outputs/" + OUTPUT_FILE,
    title: str = TITLE,
):
    """
    Plot n rows of (noisy | clean | predicted) triplets.

    maps: list of n triplets, i.e. [[noisy_0, clean_0, pred_0], ..., [noisy_n, clean_n, pred_n]].
          If None, n random samples are loaded from data_path and predictions are
          computed with `model` (required in that case).
    Entries can be torch tensors or numpy arrays.
    """
    if maps is None:
        assert model is not None, "model is required when no maps are passed"
        sample = load_random_samples(data_path, n=n)
        noisy, pred, clean = example_forward_pass(model, sample)
        maps = [[noisy[i], clean[i], pred[i]] for i in range(len(noisy))]

    n = len(maps)
    assert all(len(triplet) == 3 for triplet in maps), "each entry in maps must be [noisy, clean, pred]"

    col_titles = ["Noisy", "Clean", "Predicted"]

    fig, axes = plt.subplots(n, 3, figsize=(15, 5 * n), squeeze=False)

    for i, triplet in enumerate(maps):
        imgs = []
        for m in triplet:
            # Torch tensor -> numpy, drop channel/batch dims
            if hasattr(m, "detach"):
                m = m.detach().cpu().numpy()
            imgs.append(np.asarray(m).squeeze())

        # Clean and predicted share a color scale so they are directly comparable;
        # noisy gets its own, since its range is usually much larger.
        vmin, vmax = imgs[1].min(), imgs[1].max()
        limits = [(None, None), (vmin, vmax), (vmin, vmax)]

        for j, (img, (lo, hi)) in enumerate(zip(imgs, limits)):
            ax = axes[i, j]
            im = ax.imshow(
                img, cmap="viridis", vmin=lo, vmax=hi,
                extent=[0, IMG_SIZE, 0, IMG_SIZE],
                origin="lower",   # see note below
            )
            ax.set_xlabel("arcmin")
            ax.set_ylabel("arcmin")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if i == 0:
                ax.set_title(col_titles[j])

    fig.suptitle(title)
    fig.tight_layout()

    fig.savefig(save_path, bbox_inches="tight")
    print(f"Figure saved to: {save_path}")

    plt.close(fig)
    return None

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
            dataloader, _, _ = load_and_prep_eval_data(data_file)
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
            dataloader, _, _ = load_and_prep_eval_data(data_file)
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
            dataloader, _, _ = load_and_prep_eval_data(data_file)
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

def eval_power_spectrum(model = None, data = None, model_path = None, data_file = None, batch_size = BATCH_SIZE, device = DEVICE):

    if model is None:
        if model_path is None:
            raise ModelAccessibilityError()
        else:
            model, _, _ = eval.load_model(model_path, device)
    if data is None:
        if data_file is None:
            raise DataAccessibilityError()
        else:
            dataloader, noisy_denorm_paras, clean_denorm_paras = eval.load_and_prep_eval_data(data_file)
    else:
        dataloader = data

    residuals = []
    power_groundtruths = []
    power_preds = []
    with torch.no_grad():
        for x, y in dataloader:
            noisy_batch = x.to(device, non_blocking=True)
            preds = model(noisy_batch)
            preds_np = preds.cpu().numpy()

            for i in range(preds_np.shape[0]):
                pred_img = preds_np[i, 0]
                y_img = y.numpy()[i, 0]
                pred_img_denormalized = functions.denormalize_map(pred_img, clean_denorm_paras)
                y_img_denormalized = functions.denormalize_map(y_img, clean_denorm_paras)

                k_bins, current_power_pred = radial_power_spectrum(pred_img_denormalized)
                _, current_power_groundtruth = radial_power_spectrum(y_img_denormalized)

                with np.errstate(divide="ignore", invalid="ignore"):
                    current_residual = (current_power_pred - current_power_groundtruth) / current_power_groundtruth
                current_residual = np.where(np.isfinite(current_residual), current_residual, np.nan)

                residuals.append(current_residual)
                power_groundtruths.append(current_power_groundtruth)
                power_preds.append(current_power_pred)

    return k_bins, residuals, power_preds, power_groundtruths

def compute_fractional_residual(power_clean, power_pred):
    assert len(power_clean) == len(power_pred), "Number of groundtruth and prediction powerspectra differ."
    power_clean = np.array(power_clean)
    power_pred = np.array(power_pred)
    frac_res = []
    for i in range(len(power_clean)):
        frac_res.append((power_pred[i] - power_clean[i])/power_clean[i])
    return frac_res

def plot_fractional_residual(ell = None, k_bins = None, L = IMG_SIZE,  frac_res = None, power_clean = None, power_pred = None, residuals = None, ell_min = 0, ell_max = 2000, number_of_fracRes = 50, save_path = ""):
    if ell is None:
        assert k_bins is not None, "ell and k_bins cannot both be None. One has to be given."
        ell = k_bins * 2*np.pi / L
    if frac_res is None:
        assert power_clean is not None and power_pred is not None and residuals is not None, "Power Spectra needed to calculate fracional residuals"
        frac_res = compute_fractional_residual(power_clean, power_pred)
    ell = ell[2:]
    valid = (ell<ell_max)
    ell = ell[valid]
    frac_res = np.array(frac_res)[:, 2:]
    frac_res = frac_res[:, valid]
    if len(frac_res)<number_of_fracRes:
        for i in range(len(frac_res)):
            y = frac_res[i]
            plt.plot(ell, y, color = "red", alpha = 0.1)
            plt.xlim(ell_min, ell_max)
    if len(frac_res)>=number_of_fracRes:
        for i in range(number_of_fracRes):
            y = frac_res[i]
            plt.plot(ell, y, color = "red", alpha = 0.1)
            plt.xlim(ell_min, ell_max)

    plt.ylabel(r"$( \hat{C}_\ell - C_\ell )/ C_\ell$", fontsize=16)
    plt.xlabel(r"$\ell$", fontsize=16)
    #plt.legend()
    plt.savefig(save_path)
    return None

def compute_power_ratio(power_clean, power_pred):
    power_clean = power_clean[2:]
    power_pred = power_pred[2:]
    assert len(power_clean) == len(power_pred), "Number of groundtruth and prediction powerspectra differ."
    ratio = []
    for i in range(len(power_clean)):
        ratio.append(power_pred[i]/power_clean[i])
    return ratio

def plot_power_ratio(ell = None, k_bins = None, L = IMG_SIZE,  ratio = None, power_clean = None, power_pred = None, ell_min = 0, ell_max = 2000, number_of_ratios = 50, save_path = ""):
    if ell is None:
        assert k_bins is not None, "ell and k_bins cannot both be None. One has to be given."
        ell = k_bins * 2*np.pi / L
    if ratio is None:
        assert power_clean is not None and power_pred is not None, "Power Spectra needed to calculate the ratio between them."
        ratio = compute_power_ratio(power_clean, power_pred)
    ell = ell[2:]
    valid = (ell<ell_max)
    ell = ell[valid]
    ratio = np.array(ratio)[:, 2:]
    ratio = ratio[:, valid]
    if len(ratio)<number_of_ratios:
        for i in range(len(ratio)):
            y = ratio[i]
            plt.plot(ell, y, color = "red", alpha = 0.1)
            plt.xlim(ell_min, ell_max)
    if len(ratio)>=number_of_ratios:
        for i in range(number_of_ratios):
            y = ratio[i]
            plt.plot(ell, y, color = "red", alpha = 0.1)
            plt.yscale("log")
            plt.xlim(ell_min, ell_max)

    plt.ylabel(r"$\hat{C}_\ell / C_\ell$", fontsize=16)
    plt.xlabel(r"$\ell$", fontsize=16)
    #plt.legend()
    plt.savefig(save_path)
    return None

def n_modes(k_bins, N=256, dk=1.0):
    """Independent modes per radial bin (half the FFT modes, since the map is real)."""
    kx = np.fft.fftfreq(N, d=1.0 / N)                 # integer wavenumbers
    kk = np.hypot(*np.meshgrid(kx, kx))
    return np.array([((kk >= k - dk/2) & (kk < k + dk/2)).sum() for k in k_bins]) / 2

def chi2(clean, pred, nmodes, skip=2, n_max=None):
    c = np.asarray(clean)[:, skip:n_max]
    p = np.asarray(pred)[:, skip:n_max]
    N = np.asarray(nmodes)[skip:n_max]
    chi2 = np.sum(N * (p - c)**2 / (2 * c**2), axis=1)   # one value per sample
    return chi2, chi2 / c.shape[1]                        # total, reduced (dof = n_bins)

def cross_correlation_coefficient(map1, map2):

    ny, nx = map1.shape
    map1_ft = np.fft.fftshift(np.fft.fft2(map1))
    map2_ft = np.fft.fftshift(np.fft.fft2(map2))

    r, r_max = functions.radial_bin_indices(ny, nx)
    rr = r.ravel()
    cnt = np.bincount(rr, minlength=r_max)[:r_max].astype(float)
    cnt[cnt == 0] = np.nan

    def radial_mean(a):
        return np.bincount(rr, weights=a.ravel(), minlength=r_max)[:r_max] / cnt

    pg = radial_mean(np.abs(map1_ft)**2)
    pp = radial_mean(np.abs(map2_ft)**2)
    pc = radial_mean(np.real(map1_ft * np.conj(map2_ft)))
    with np.errstate(divide="ignore", invalid="ignore"):
        r_k = pc / np.sqrt(pg * pp)

    radial_bin_idxs = np.arange(r_max)
    cross_correation_coefficients = np.where(np.isfinite(r_k), r_k, np.nan)
    return radial_bin_idxs, cross_correation_coefficients


def calc_multiple_cross_correlation_coefficient(groundtruth_img=None, pred_img=None, model=None,
                                       model_path=None, data_path=None, device=DEVICE, n=50, noisy_clean_spectrum = False):
    if groundtruth_img is None or pred_img is None:
        assert model is not None or model_path is not None
    if model is None:
        model, dn_noisy, dn_clean = eval.load_model(model_path, device=device)
    model.eval()
    sample = load_random_samples(data_path, n=n)
    noisy, pred, clean = example_forward_pass(model, sample)
    groundtruth_img = [functions.to_np(functions.denormalize_map(clean[i], dn_clean)) for i in range(len(clean))]
    pred_img = [functions.to_np(functions.denormalize_map(pred[i], dn_clean)) for i in range(len(pred))]

    results = [cross_correlation_coefficient(g, p) for g, p in zip(groundtruth_img, pred_img)]
    k_bins = results[0][0]
    r_all = np.array([r for _, r in results])

    if noisy_clean_spectrum:
        noisy_img = [functions.to_np(functions.denormalize_map(noisy[i], dn_noisy)) for i in range(len(noisy))]
        results_noisy_clean = [cross_correlation_coefficient(d, g) for d, g in zip(noisy_img, groundtruth_img)]
        r_all_noisy_clean = np.array([r for _, r in results_noisy_clean])
        return k_bins, r_all, r_all_noisy_clean
    
    return k_bins, r_all, 0

def plot_cross_correlation(ell = None, k_bins = None, L = IMG_SIZE, cross_correlations = None, groundtruth_img = None, pred_img = None, ell_min = 0, ell_max = 2000, number_of_cc = 50, r_noisy_clean = None, save_path = ""):
    if ell is None:
        assert k_bins is not None, "ell and k_bins cannot both be None. One has to be given."
        ell = k_bins * 2*np.pi / L
    if cross_correlations is None:
        k_bins, cross_correlations, r_noisy_clean = calc_multiple_cross_correlation_coefficient(groundtruth_img, pred_img)
    ell = ell[2:]
    valid = (ell<ell_max)
    ell = ell[valid]
    cross_correlations = np.array(cross_correlations)[:, 2:]
    cross_correlations = cross_correlations[:, valid]
    for i in range(min(len(cross_correlations)-1, number_of_cc-1)):
        plt.plot(ell, cross_correlations[i], color="red", alpha=0.1)
        plt.ylim(0, 1.05)
        plt.ylabel(r"Cross-Correlation coefficient $r(\ell)$", fontsize=8)
        plt.xlabel(r"Multipoles $\ell$", fontsize=8)
    plt.plot(ell, cross_correlations[min(len(cross_correlations), number_of_cc)-1], color="red", alpha=0.1, label = "prediction-groundtruth")
    if r_noisy_clean.any():
        r_noisy_clean = r_noisy_clean[0]
        r_noisy_clean[2:]
        plt.plot(ell, r_noisy_clean[2:], color = "black", alpha = 0.2, label = "noisy-groundtruth")
        plt.legend()
    plt.savefig(save_path)
    return None

def _summary_lines(name, values, fmt):
    return [
        f"{name} max: {np.nanmax(values):{fmt}}",
        f"{name} min: {np.nanmin(values):{fmt}}",
        f"{name} avg: {np.nanmean(values):{fmt}}",
        f"{name} n_nan: {np.sum(np.isnan(values))}",
        "",
    ]


def _radial_profile_lines(name, k_bins, stacked_values):
    # stacked_values: shape (n_images, n_k) -- average over images, ignoring nans
    mean_profile = np.nanmean(stacked_values, axis=0)
    lines = [f"=== {name} radial profile (mean over images) ==="]
    lines.append("k\tvalue")
    for k, v in zip(k_bins, mean_profile):
        lines.append(f"{k}\t{v:.6e}")
    lines.append("")
    return lines


def write_eval_results(eval_file=EVAL_FILE, total_mse=None, mse_list=None,
                        psnr_list=None, ssim_list=None,
                        power_spectrum_result=None, cross_corr_result=None):
    # power_spectrum_result: (k_bins, residuals, power_preds, power_groundtruths) or None
    # cross_corr_result: (k_bins, r_k_list) or None -- r_k_list is a list of per-image r_k arrays

    # scalar-per-image summaries derived from the radial arrays, so they can
    # sit in the same per-image table as MSE/PSNR/SSIM
    ps_scalar = None
    cc_scalar = None
    if power_spectrum_result is not None:
        _, residuals, _, _ = power_spectrum_result
        ps_scalar = [np.nanmean(np.abs(r)) for r in residuals]
    if cross_corr_result is not None:
        _, r_k_list = cross_corr_result
        cc_scalar = [np.nanmean(r) for r in r_k_list]

    per_image = {
        name: (vals, fmt)
        for name, vals, fmt in [
            ("MSE", mse_list, ".6e"),
            ("PSNR", psnr_list, ".4f"),
            ("SSIM", ssim_list, ".4f"),
            ("PS_resid_meanabs", ps_scalar, ".4e"),
            ("CrossCorr_mean", cc_scalar, ".4f"),
        ]
        if vals is not None
    }

    lengths = {name: len(vals) for name, (vals, _) in per_image.items()}
    if len(set(lengths.values())) > 1:
        print(f"WARNING: metric lists have different lengths: {lengths}")

    with open(eval_file, "w") as f:
        f.write("=== Evaluation summary ===\n")
        f.write(f"Model file: {MODEL_FILE}\n")
        f.write(f"Data file:  {DATA_FILE}\n")
        if per_image:
            n_images = min(lengths.values())
            f.write(f"Number of eval images: {n_images}\n")
            f.write(f"Metrics computed: {', '.join(per_image)}\n\n")

        if total_mse is not None:
            f.write(f"MSE total: {total_mse:.6e}\n")
        for name, (vals, fmt) in per_image.items():
            f.write("\n".join(_summary_lines(name, vals, fmt)) + "\n")

        if per_image:
            f.write("=== Per-image results ===\n")
            f.write("index\t" + "\t".join(per_image) + "\n")
            n = min(lengths.values())
            for i in range(n):
                row = [f"{vals[i]:{fmt}}" for vals, fmt in per_image.values()]
                f.write(f"{i}\t" + "\t".join(row) + "\n")
            f.write("\n")

        # radial profiles get their own section since they're arrays, not scalars
        if power_spectrum_result is not None:
            k_bins, residuals, power_preds, power_groundtruths = power_spectrum_result
            f.write("".join(l + "\n" for l in _radial_profile_lines(
                "Power spectrum residual", k_bins, np.array(residuals))))
        if cross_corr_result is not None:
            k_bins, r_k_list = cross_corr_result
            f.write("".join(l + "\n" for l in _radial_profile_lines(
                "Cross-correlation", k_bins, np.array(r_k_list))))

def load_loss_data_per_epoch(log_path):
    with open(log_path) as f:
        text = f.read()

    patterns = {
        "loss_disc":  r"discriminator loss:\s*([-+0-9.eE]+)",
        "loss_gen":   r"generator loss:\s*([-+0-9.eE]+)",
        "loss_recon": r"reconstruction loss:\s*([-+0-9.eE]+)",
        "loss_adv":   r"adversarial loss:\s*([-+0-9.eE]+)",
    }

    losses = {
        name: np.array(re.findall(pat, text, flags=re.IGNORECASE), dtype=float)
        for name, pat in patterns.items()
    }

    loss_disc  = losses["loss_disc"]
    loss_gen   = losses["loss_gen"]
    loss_recon = losses["loss_recon"]
    loss_adv   = losses["loss_adv"]

    print({k: v.shape for k, v in losses.items()})   # each should be (64,)
    return loss_disc, loss_gen, loss_recon, loss_adv




def main():
    start_time = time.perf_counter()

    model = load_model()
    model.eval()

    data, _, _ = load_and_prep_eval_data(DATA_FILE)

    visuals = False
    metrics = False
    if visuals:
        print("Creating Figures.")
        make_figures(model = model, n = 3)
    if metrics:
        print("Calculating metrics.")

        total_mse = mse_list = psnr_list = ssim_list = None

        total_mse, mse_list = eval_MSE(model=model, data=data)
        psnr_list = eval_PSNR(model=model, data=data)
        ssim_list = eval_ssim(model=model, data=data)

        k_bins_ps, residuals, power_preds, power_groundtruths = eval_power_spectrum(model=model, data=data)
        power_spectrum_result = (k_bins_ps, residuals, power_preds, power_groundtruths)

        # cross-correlation needs per-image (pred, groundtruth) pairs; assumes
        # you have (or add) an eval_cross_correlation wrapper analogous to eval_power_spectrum
        k_bins_cc, r_k_list = eval_cross_correlation(model=model, data=data)
        cross_corr_result = (k_bins_cc, r_k_list)

        if mse_list is not None:
            print("Total MSE: ", total_mse, " Max MSE: ", max(mse_list))
        if psnr_list is not None:
            print("PSNR Max: ", max(psnr_list), " Min: ", min(psnr_list))
        if ssim_list is not None:
            print("SSIM Max: ", max(ssim_list), " Min: ", min(ssim_list),
                " Avg: ", np.mean(ssim_list))

        write_eval_results(EVAL_FILE, total_mse=total_mse, mse_list=mse_list,
                            psnr_list=psnr_list, ssim_list=ssim_list,
                            power_spectrum_result=power_spectrum_result,
                            cross_corr_result=cross_corr_result)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    functions.write_doc("evaluation",
                         runtime=elapsed_time, output_file=OUTPUT_FILE)
    return None

if __name__ == "__main__":
    print("Executing main() in eval.py")
    main()