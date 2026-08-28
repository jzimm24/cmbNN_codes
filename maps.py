import numpy as np
import matplotlib.pyplot as plt
import h5py
import csv
from datetime import datetime
import time
import torch

import functions as functions

import config as config


#-------------------------------------------------------------------------------------------------------------------------
class UnknownFileStructureError(ValueError):
    pass
#-------------------------------------------------------------------------------------------------------------------------


def make_circle_map(map_size: int = 32, radius: float = 10, cx: float = 16, cy: float = 16):
    """
    Create a simple image of shape (map_size, map_size) with a circle defined by radius and offsets (cx, cy).
    The image has zero values except for pixels belonging to the circle.

    Parameters
    ----------
    map_size : int
        Image width & height in pixels.
    radius : float
        Radius of the circle.
    cx : float
        position of circle center in the x-dim.
    cy : float
        position of circle center in the y-dim.

    Returns
    -------
    Z : 2D np.ndarray
        Map of circle.
    """
    x = np.linspace(0, map_size-1, map_size)
    y = np.linspace(0, map_size-1, map_size)
    X, Y = np.meshgrid(x, y)
    Z = (((X-cx)**2 + (Y-cy)**2) <= radius**2).astype(int)
    return Z

def make_multiple_circle_map(map_size: int = 32, number_circles: int = 1,  radius: list[float] = [10], cx: list[float] = [16], cy: list[float] = [16]):
    """
    Create image of shape (map_size, map_size) with multiple circles defined by radius and offsets (cx, cy).
    The image has zero values except for pixels belonging to the circles.

    Parameters
    ----------
    map_size : int
        Image width & height in pixels.
    number_circles : int
        number of circles in the image.
    radius : list[float]
        radii of the circles.
    cx : list[float]
        positions of circle-centers in the x-dim.
    cy : list[float]
        positions of circle-centers in the y-dim.

    Returns
    -------
    Z_final : 2D np.ndarray
        Map of multiple circles.
    """
    x = np.linspace(0, map_size-1, map_size)
    y = np.linspace(0, map_size-1, map_size)
    X, Y = np.meshgrid(x, y)
    Z_list = []
    for i in range(number_circles):
        Z = (((X-cx[i])**2 + (Y-cy[i])**2) <= radius[i]**2).astype(int)
        Z_list.append(Z)
    Z_final = sum(Z_list)
    return Z_final

def make_random_circle_maps(n: int = 1, map_size: int = 32, radius_mean: float = 10, radius_var: float = 0, radius_min: float = 2, 
                            cx_mean: float = 16, cx_var: float = 0, cy_mean: float = 16, cy_var: float = 0, max_shift: float = 10, seed: int = 42):
    """
    Create images of shape (map_size, map_size) with circle defined by random radii and offsets (cx, cy).
    The images have zero values except for pixels belonging to the circle.
    The values of the circle parameters are sampled from a gaussian distr.

    Parameters
    ----------
    n : int
        number of images
    map_size : int
        Image width & height in pixels.
    radius_mean : float
        Mean of the gauss distr. of which the radii are sampled.
    radius_var  : float
        Variance of the gauss distr. of which the radii are sampled.
    radius_min  : float
        minimal radius a circle must have before resampling is triggered.
    cx_mean : float
        Mean of the gauss distr. of which the cx values are sampled.
    cx_var  : float
        Variance of the gauss distr. of which the cx values are sampled.
    cy_mean : float
        Mean of the gauss distr. of which the cy values are sampled.
    cy_var  : float
        Variance of the gauss distr. of which the cy values are sampled.
    max_shift  : float
        Maximal positional shift a circle-center can have before resampling is triggered.
    seed    : int
        seed for the randomization in parameter sampling.

    Returns
    -------
    Z_final : 2D np.ndarray
        Map of multiple circles.
    """
    rng = np.random.default_rng(seed)
    radii = rng.normal(radius_mean, radius_var, n).clip(min = radius_min)
    cx_shifts = rng.normal(cx_mean, cx_var, n).clip(max=max_shift)
    cy_shifts = rng.normal(cy_mean, cy_var, n).clip(max=max_shift)

    circle_images = np.zeros((n, map_size, map_size))
    doc = []
    for i in range(n):
        circle_images[i, :, :] = make_circle_map(map_size, radii[i], cx_shifts[i], cy_shifts[i])
        doc.append({"index": i, "radius": round(radii[i], 3), "cx_shift": round(cx_shifts[i], 3), "cy_shift": round(cy_shifts[i], 3)})

    return circle_images, doc

def make_random_multiple_circle_maps(n: int = 1, map_size: int = 32, max_number_circles: int = 1, radius_mean: float = 10, radius_var: float = 0, radius_min: float = 2, 
                            cx_mean: float = 16, cx_var: float = 0, cy_mean: float = 16, cy_var: float = 0, max_shift: float = 10, seed: int = 42):
    """
    Create images of shape (map_size, map_size) with multiple circles defined by random radii and offsets (cx, cy).
    The images have zero values except for pixels belonging to the circles.
    The values of the circle parameters are sampled from a gaussian distr.

    Parameters
    ----------
    n : int
        number of images
    map_size : int
        Image width & height in pixels.
    max_number_circles: int
        Maximal number of circles one image can have. The number of circle
        is the sample from a linear distribution between zero and said maximum.
    radius_mean : float
        Mean of the gauss distr. of which the radii are sampled.
    radius_var  : float
        Variance of the gauss distr. of which the radii are sampled.
    radius_min  : float
        minimal radius a circle must have before resampling is triggered.
    cx_mean : float
        Mean of the gauss distr. of which the cx values are sampled.
    cx_var  : float
        Variance of the gauss distr. of which the cx values are sampled.
    cy_mean : float
        Mean of the gauss distr. of which the cy values are sampled.
    cy_var  : float
        Variance of the gauss distr. of which the cy values are sampled.
    max_shift  : float
        Maximal positional shift a circle-center can have before resampling is triggered.
    seed    : int
        seed for the randomization in parameter sampling.

    Returns
    -------
    circle_images : 2D np.ndarray
        Maps of multiple circles.
    doc : 
        dictionary with list of circle parameters
    """
    rng = np.random.default_rng(seed)

    circle_images = np.zeros((n, map_size, map_size))
    doc = []

    for i in range(n):
        number_circles = rng.integers(1, max_number_circles, endpoint=True)

        radii = rng.normal(radius_mean, radius_var, number_circles).clip(min=radius_min)
        cx_shifts = rng.normal(cx_mean, cx_var, number_circles).clip(max=max_shift)
        cy_shifts = rng.normal(cy_mean, cy_var, number_circles).clip(max=max_shift)

        circle_images[i, :, :] = make_multiple_circle_map(map_size, number_circles, radii, cx_shifts, cy_shifts)
        doc.append({"index": i, "radius": radii, "cx_shift": cx_shifts, "cy_shift": cy_shifts})

    return circle_images, doc

def make_noise_maps(n: int = 1, map_size: int = 32, noise_max: float = 10, seed: int = 42):
    """
    Create images of shape (map_size, map_size) with white noise.

    Parameters
    ----------
    n : int
        number of images
    map_size : int
        Image width & height in pixels.
    noise_max   : float
        Amplitude of the white noise.
    seed    : int
        seed for the randomization in parameter sampling.

    Returns
    -------
    Z_final : 2D np.ndarray
        Maps of white noise.
    """
    noise_maps = np.zeros((n, map_size, map_size))
    rng = np.random.default_rng(seed)
    for i in range(n):
        noise_maps[i, :, :] = rng.uniform(0, noise_max, size=(map_size, map_size))

    return noise_maps

def save_maps(maps, ground_truths, doc, file_name: str = "circle_maps"):
    """
    Saves noisy images their ground_truths and their circle parameters in a npz file.

    Parameters
    ----------
    maps : 
        Array of 2d maps (noisy maps)
    ground_truths : 
        Array of 2d maps (no noise maps/ground_truths)
    file_name   : str
        Path for npz file.

    Returns
    ---------
    
    """
    radii = [m["radius"] for m in doc]
    cx_shifts = [m["cx_shift"] for m in doc]
    cy_shifts = [m["cy_shift"] for m in doc]

    np.savez(
    file_name + ".npz",
    maps=maps,
    ground_truths=ground_truths,
    radii=np.array(radii, dtype=object),
    cx_shifts=np.array(cx_shifts, dtype=object),
    cy_shifts=np.array(cy_shifts, dtype=object)
)
    print("File saved as ", file_name, ".npz")
    return None

# real noise map functions -----------------------------------------------------------------------------------------------------------

# Original code taken from Dr. Kaustuv Basu. Expanded to include randomization of source position and source number.

save_mode = config.maps_configs.save_mode   # choose 'hdf5' or 'csv'
flux_mode = config.maps_configs.flux_mode    # 'lin', 'log', or 'gauss' fluxes

# === User‐defined parameters ===
num_images               = config.maps_configs.num_images      # how many images to generate
image_size               = config.maps_configs.image_size         # pixels (height=width)

max_source_number = config.maps_configs.max_source_number            # max number of sources an image can have. Every image has 0-max_source_number of sources distributed linearly
x_shift_mean = image_size/4     # positional offset (pixels) variables for randomly positioned (gaussian distr.) sources
x_shift_var = image_size/4
x_shift_max = image_size/2
y_shift_mean = image_size/4
y_shift_var = image_size/4
y_shift_max = image_size/2

seed = config.maps_configs.seed                   #seed for random value selection


rc_mean  = config.maps_configs.rc_mean                       # r_c in the β‐model (pixels)   --> DEFAULT 7 PIXELS
rc_sigma = config.maps_configs.rc_sigma                       # std dev in r_c (in pixels)   --> DEFAULT sigma=2 PIXELS


I0_range                 = config.maps_configs.I0_range    # if I0_fixed is None, draw uniformly/log-uniformly from this
                                         # (used when flux_mode='lin' or 'log')
I0_fixed                 = config.maps_configs.I0_fixed          # set to a float to force same I0 each time
I0_mean                  = config.maps_configs.I0_mean           # mean of Gaussian flux prior   (used when flux_mode='gauss')
I0_sigma                 = config.maps_configs.I0_sigma           # std-dev of Gaussian flux prior (used when flux_mode='gauss')


white_noise_amplitude    = config.maps_configs.white_noise_amplitude        # σ for white Gaussian noise (~2 when with 1/f, ~7 when WN only)
one_over_f_slope         = config.maps_configs.one_over_f_slope        # power‐law slope (1.5=pink, 3.0=red)
one_over_f_amplitude     = config.maps_configs.one_over_f_amplitude        # scaling for 1/f noise (~1 when slope 3.0, ~2 when slope 1.5)

gaussian_smoothing_fwhm  = config.maps_configs.gaussian_smoothing_fwhm        # if >0, smooth final image with this FWHM   --> DEFAULT 5 PIXELS


output_h5               = config.maps_configs.output_h5

visualization_path = config.maps_configs.visualization_path

def generate_beta_model_image(size, 
                              rc_mean, 
                              rc_sigma,
                              x_shift = 0,
                              y_shift = 0, 
                              I0_range=(0.5,1.0), 
                              I0_fixed=None):
    """
    Create a β‐model image of shape (size,size):
      I(r) = I0 * [1 + (r/rc)^2]^(-1.5)

    Now rc is not fixed but ~ Normal(rc_mean, rc_sigma),
    truncated at >0.

    Parameters
    ----------
    size : int
        Image width & height in pixels.
    rc_mean : float
        Mean core radius (pixels).
    rc_sigma : float
        Std-dev for core radius (pixels).
    I0_range : tuple of float
        (min, max) for uniform draw of I0 if I0_fixed is None.
    I0_fixed : float or None
        If given, use this central peak value every time;
        otherwise draw I0 ~ Uniform(*I0_range).

    Returns
    -------
    image : 2D np.ndarray
        The generated β‐model image.
    I0 : float
        The central peak intensity used.
    rc_used : float
        The actual core radius used.
    """

    # Special case: empty image if I0_fixed == 0
    if I0_fixed == 0:
        return np.zeros((size, size)), 0.0, 0.0

    # 1) draw rc, retry if non-positive
    rc = np.random.normal(loc=rc_mean, scale=rc_sigma)
    while rc <= 0:
        rc = np.random.normal(loc=rc_mean, scale=rc_sigma)

    # 2) draw I0 if not fixed
    if I0_fixed is None:
        I0 = np.random.uniform(*I0_range)
    else:
        I0 = I0_fixed

    # 3) build the grid & compute the profile
    y, x = np.indices((size, size))
    center = (size//2, size//2)
    r = np.sqrt((x-center[1]-x_shift)**2 + (y-center[0]-y_shift)**2)
    image = I0 * (1.0 + (r/rc)**2)**(-1.5)

    return image, I0, rc

def generate_beta_model_image_logI0(size, 
                                    rc_mean, 
                                    rc_sigma,
                                    x_shift = 0,
                                    y_shift = 0,
                                    I0_range=(0.1, 1.0), 
                                    I0_fixed=None):
    """
    Create a β‐model image of shape (size,size):
      I(r) = I0 * [1 + (r/rc)^2]^(-1.5)

    Now rc is not fixed but ~ Normal(rc_mean, rc_sigma),
    truncated at >0.

    Parameters
    ----------
    size : int
        Image width & height in pixels.
    rc_mean : float
        Mean core radius (pixels).
    rc_sigma : float
        Std-dev for core radius (pixels).
    I0_range : tuple of float
        (min, max) for *log-uniform* draw of I0 if I0_fixed is None.
    I0_fixed : float or None
        If given, use this central peak value every time;
        if 0, return empty image.
        otherwise draw I0 ~ LogUniform(*I0_range).

    Returns
    -------
    image : 2D np.ndarray
        The generated β‐model image.
    I0 : float
        The central peak intensity used.
    rc_used : float
        The actual core radius used.
    """

    # Special case: empty image if I0_fixed == 0
    if I0_fixed == 0:
        return np.zeros((size, size)), 0.0, 0.0

    # 1) draw rc, retry if non-positive
    rc = np.random.normal(loc=rc_mean, scale=rc_sigma)
    while rc <= 0:
        rc = np.random.normal(loc=rc_mean, scale=rc_sigma)

    # 2) draw I0 if not fixed (log-uniform)
    if I0_fixed is None:
        log_min, log_max = np.log10(I0_range[0]), np.log10(I0_range[1])
        I0 = 10 ** np.random.uniform(log_min, log_max)
    else:
        I0 = I0_fixed

    # 3) build the grid & compute the profile
    y, x = np.indices((size, size))
    center = (size//2, size//2)
    r = np.sqrt((x-center[1]-x_shift)**2 + (y-center[0]-y_shift)**2)
    image = I0 * (1.0 + (r/rc)**2)**(-1.5)

    return image, I0, rc

def generate_beta_model_image_gaussI0(size,
                                      rc_mean,
                                      rc_sigma,
                                      x_shift = 0,
                                      y_shift = 0,
                                      I0_mean=1.0,
                                      I0_sigma=0.2,
                                      I0_fixed=None):
    """
    Create a β-model image of shape (size, size):
      I(r) = I0 * [1 + (r/rc)^2]^(-1.5)

    rc is drawn from Normal(rc_mean, rc_sigma), truncated at > 0.

    I0 is drawn from a Gaussian Normal(I0_mean, I0_sigma).  Draws that yield
    I0 <= 0 are rejected and redrawn (rejection / acceptance sampling), which
    effectively produces a one-sided truncated-normal distribution on (0, ∞).

    NOTE on non-Gaussianity: rejection sampling introduces a slight positive
    bias whenever I0_mean / I0_sigma is small (i.e. the untruncated Gaussian
    has non-negligible probability mass below zero).  If you need an exact
    truncated-normal distribution, replace the while-loop draw with:

        from scipy.stats import truncnorm
        a = -I0_mean / I0_sigma     # lower clip in units of sigma
        I0 = truncnorm.rvs(a=a, b=np.inf, loc=I0_mean, scale=I0_sigma)

    This is mathematically exact and only marginally slower than the
    rejection-sampling approach for typical (I0_mean >> I0_sigma) cases.

    Parameters
    ----------
    size : int
        Image width & height in pixels.
    rc_mean : float
        Mean core radius (pixels).
    rc_sigma : float
        Std-dev for core radius (pixels).
    I0_mean : float
        Mean of the Gaussian peak-flux prior.
    I0_sigma : float
        Std-dev of the Gaussian peak-flux prior.
    I0_fixed : float or None
        If given, use this central peak value every time;
        if 0, return empty image.
        Otherwise draw I0 ~ TruncatedNormal(I0_mean, I0_sigma, lower=0).

    Returns
    -------
    image : 2D np.ndarray
        The generated β-model image.
    I0 : float
        The central peak intensity used.
    rc_used : float
        The actual core radius used.
    """

    # Special case: empty image if I0_fixed == 0
    if I0_fixed == 0:
        return np.zeros((size, size)), 0.0, 0.0

    # 1) draw rc, retry if non-positive
    rc = np.random.normal(loc=rc_mean, scale=rc_sigma)
    while rc <= 0:
        rc = np.random.normal(loc=rc_mean, scale=rc_sigma)

    # 2) draw I0 if not fixed (Gaussian, rejection-sampled to keep I0 > 0)
    if I0_fixed is None:
        I0 = np.random.normal(loc=I0_mean, scale=I0_sigma)
        while I0 <= 0:
            I0 = np.random.normal(loc=I0_mean, scale=I0_sigma)
    else:
        I0 = I0_fixed

    # 3) build the grid & compute the profile
    y, x = np.indices((size, size))
    center = (size//2, size//2)
    r = np.sqrt((x-center[1]-x_shift)**2 + (y-center[0]-y_shift)**2)
    image = I0 * (1.0 + (r/rc)**2)**(-1.5)

    return image, I0, rc

def generate_white_noise(shape, amplitude):
    """White Gaussian noise with std=amplitude."""
    return amplitude * np.random.normal(size=shape)

def generate_1overf_noise(shape, slope, amplitude):
    """
    Generate 1/f^slope noise:
      - Build power spectrum ∝ 1/f^slope
      - Randomize phases, inverse FFT, normalize, scale.
    """
    ny, nx = shape
    y = np.fft.fftfreq(ny).reshape(-1,1)
    x = np.fft.fftfreq(nx).reshape(1,-1)
    f = np.sqrt(x**2 + y**2)
    f[0,0] = np.min(f[f>0])  # avoid div-by-zero
    P = 1 / (f**slope)
    phases = np.exp(2j*np.pi*np.random.random(f.shape))
    fft_n = np.sqrt(P) * phases
    noise = np.fft.ifft2(fft_n).real
    noise -= noise.mean()
    
    ## THIS NORMALIZES THE NOISE RMS, CHANGING NOISE STATISTICS! SWITCHING OFF!
    # noise /= noise.std()
    return amplitude * noise

def strict_1overf_noise(shape, slope, amplitude, rng=None):
    """
    Generate a single realization of 1/f^slope (red/power-law) noise using a
    proper Hermitian-symmetric Gaussian construction in Fourier space.

    Unlike the random-phase approach — which sets |F[k]| = sqrt(P[k]) exactly and
    randomizes only the phase — this samples real and imaginary parts of each Fourier
    coefficient independently from N(0, P[k]/2), yielding a true Gaussian random field
    whose per-realization power spectrum fluctuates around P(f) ∝ 1/f^slope.
    Hermitian symmetry F[-k] = conj(F[k]) is enforced explicitly so that ifft2
    returns a purely real map.

    Parameters
    ----------
    shape     : (ny, nx) tuple — output map shape in pixels
    slope     : float — spectral index; 3.0 gives 1/f^3 (steep red) noise
    amplitude : float — overall scaling applied after mean subtraction
    rng       : np.random.Generator, optional
                Pass a seeded generator for reproducibility, e.g.
                rng=np.random.default_rng(42). If None, a fresh generator
                is created (non-reproducible).

    Returns
    -------
    noise : np.ndarray of shape (ny, nx), dtype float64
    """
    if rng is None:
        # Use the new-style Generator API (NumPy >=1.17); superior to legacy
        # np.random.* functions in statistical quality and reproducibility.
        rng = np.random.default_rng()

    ny, nx = shape

    # --- 2D frequency grid (cycles per pixel, range [-0.5, 0.5)) ---
    # fftfreq returns frequencies in standard FFT order: [0, 1/N, ..., Nyquist, -Nyquist+1/N, ..., -1/N]
    fy = np.fft.fftfreq(ny).reshape(-1, 1)   # shape (ny, 1)
    fx = np.fft.fftfreq(nx).reshape(1, -1)   # shape (1, nx)
    f  = np.sqrt(fx**2 + fy**2)              # radial frequency, shape (ny, nx)

    # Replace DC bin (f=0) with the smallest nonzero frequency to avoid 1/0.
    # This is consistent with the original function's convention.
    f[0, 0] = np.min(f[f > 0])

    P = 1.0 / (f ** slope)   # power spectrum: P(f) ∝ f^{-slope}

    # --- Gaussian complex draw ---
    # Each coefficient: Re, Im ~ N(0, P/2) independently.
    # => E[|F[k]|^2] = P[k]/2 + P[k]/2 = P[k].  Correct mean power.
    # rng.standard_normal() draws from N(0,1); scaling by sqrt(P/2) gives N(0, P/2).
    fft_n = np.sqrt(P / 2.0) * (
        rng.standard_normal(size=P.shape) + 1j * rng.standard_normal(size=P.shape)
    )

    # --- Enforce Hermitian symmetry: F[-k] = conj(F[k]) ---
    # For a (ny, nx) array, [-j, -k] wraps to [ny-j, nx-k].
    # [::-1, ::-1] maps [j,k] -> [ny-1-j, nx-1-k].
    # roll(+1, axis=0) then roll(+1, axis=1) shifts by +1 in each axis,
    # mapping [j,k] -> [ny-j, nx-k] = [-j, -k] mod (ny, nx).  Correct.
    fft_conj = np.conj(
        np.roll(np.roll(fft_n[::-1, ::-1], 1, axis=0), 1, axis=1)
    )  # this is conj(F[-k]) at every [j,k]

    # Projection onto Hermitian subspace: F_sym[k] = 0.5*(F[k] + conj(F[-k]))
    # After this step: F_sym[-k] = conj(F_sym[k]) holds exactly.
    # Note: expected power per mode is halved to P[k]/2, but the slope is preserved.
    fft_n = 0.5 * (fft_n + fft_conj)

    # --- IFFT and mean subtraction ---
    # ifft2 of a Hermitian-symmetric array is real to machine precision;
    # taking .real discards any residual floating-point imaginary noise.
    noise = np.fft.ifft2(fft_n).real

    ## REMOVAL OF NOISE DC OFFSET <--- THIS VIOLATED MF OPTIMALITY AND GAVE CNN DC-CHANNEL ADVANTAGE !!
    ## noise -= noise.mean()   # remove DC offset (mean of a zero-mean field)

    return amplitude * noise

def smooth_image_fourier(image, fwhm_pixels):
    """
    Smooth with an exact Gaussian beam via circular convolution in the
    Fourier domain (no truncation, no boundary-mode artifacts).

    Unlike scipy.ndimage.gaussian_filter (which uses a TRUNCATED kernel with
    reflect boundary conditions), this function applies exp(-2π²σ²k²) to the
    2D FFT, giving a truly periodic / circular convolution. As a result, the
    power spectrum of the smoothed image follows the analytical formula

        P(k) = P_pre(k) × exp(-4π²σ²k²)

    exactly, with no spurious high-k power floor from edge effects.

    Parameters
    ----------
    image : 2D ndarray
    fwhm_pixels : float
        Gaussian FWHM in pixels. 0 → returns image unchanged.

    Returns
    -------
    smoothed : 2D ndarray (real)
    """
    if fwhm_pixels <= 0:
        return image.copy()
    sigma = fwhm_pixels / 2.355
    ny, nx = image.shape
    ky = np.fft.fftfreq(ny).reshape(-1, 1)
    kx = np.fft.fftfreq(nx).reshape(1, -1)
    beam_amp = np.exp(-2.0 * np.pi**2 * sigma**2 * (kx**2 + ky**2))
    return np.fft.ifft2(np.fft.fft2(image) * beam_amp).real

def apply_gaussian_smoothing(image, fwhm_pixels):
    """
    Smooth with Gaussian kernel of given FWHM using FFT-domain circular
    convolution (calls smooth_image_fourier).

    The previous implementation used scipy.ndimage.gaussian_filter with the
    default 'reflect' boundary condition, which introduces a large spurious
    high-k power floor in the Fourier domain (> 10^4× larger than the ideal
    Gaussian prediction at k > 0.3 cycles/pixel for FWHM = 5 px, N = 128).
    This made it impossible to match the image power spectrum with an
    analytical model.  The FFT-domain approach eliminates this artifact.
    """
    return smooth_image_fourier(image, fwhm_pixels)

def save_dataset_hdf5_SLOW(outpath, noisy_images, clean_images, I0s, compression='gzip', compression_opts=4):
    """Save arrays to HDF5 file.
    Parameters
    ----------
    outpath : str
        Output .h5 path.
    noisy_images, clean_images : list or np.ndarray of shape (N, H, W)
    I0s : list or np.ndarray of length N
    """
    noisy = np.stack(noisy_images).astype(np.float32)
    clean = np.stack(clean_images).astype(np.float32)
    I0s_arr = np.array(I0s).astype(np.float32)
    with h5py.File(outpath, 'w') as f:
        f.create_dataset('noisy', data=noisy, compression=compression, compression_opts=compression_opts)
        f.create_dataset('clean', data=clean, compression=compression, compression_opts=compression_opts)
        f.create_dataset('I0', data=I0s_arr, compression='gzip', compression_opts=1)
    print(f"Wrote HDF5 dataset with shapes noisy={noisy.shape}, clean={clean.shape}, I0={I0s_arr.shape} -> {outpath}")

def save_dataset_hdf5(outpath, noisy_images, clean_images, I0s, 
                       compression='lzf', compression_opts=None):
    """
    Save image dataset to HDF5 with optimal chunking for PyTorch DataLoader.
    
    This function creates an HDF5 file optimized for random access during training.
    Key optimization: Per-image chunking ensures each image is stored in its own
    chunk, enabling fast random access without reading unnecessary data.
    
    Parameters
    ----------
    outpath : str
        Output .h5 file path.
    noisy_images, clean_images : list or np.ndarray of shape (N, H, W) or (N, C, H, W)
        Image arrays. Will be stacked if provided as list.
    I0s : list or np.ndarray of length N
        Target values (regression labels).
    compression : str or None, optional
        Compression algorithm. Options:
        - None: No compression (fastest, ~2-4 GB for 20k images)
        - 'lzf': Fast compression, good balance (recommended, ~1-2 GB)
        - 'gzip': Better compression, slower (not recommended for training data)
        Default: 'lzf' (good speed/size balance)
    compression_opts : int or None, optional
        Compression level for gzip (1-9). Ignored for lzf.
        Default: None
    
    Returns
    -------
    dict
        Dictionary with dataset statistics and file info.
    
    Notes
    -----
    Critical optimization: Uses per-image chunking (chunks=(1, H, W)) so each
    image is stored independently. This enables fast random access during training.
    
    Without proper chunking, random access can be 10-100x slower!
    
    Examples
    --------
    # Fastest (no compression, ~2x larger files)
    save_dataset_hdf5('data.h5', noisy, clean, I0s, compression=None)
    
    # Recommended (good balance)
    save_dataset_hdf5('data.h5', noisy, clean, I0s, compression='lzf')
    
    # Smaller files (slower training)
    save_dataset_hdf5('data.h5', noisy, clean, I0s, compression='gzip', compression_opts=1)
    """
    
    # Stack and convert to float32
    noisy = np.stack(noisy_images).astype(np.float32)
    clean = np.stack(clean_images).astype(np.float32)
    #I0s_arr = np.array(I0s).astype(np.float32)
    
    # Get dataset info
    n_samples = noisy.shape[0]
    image_shape = noisy.shape[1:]  # (H, W) or (C, H, W)
    
    # ============================================================
    # CRITICAL: Define per-image chunks for fast random access
    # ============================================================
    # For shape (N, H, W): chunks = (1, H, W)
    # For shape (N, C, H, W): chunks = (1, C, H, W)
    # This ensures each image is stored in its own chunk!
    image_chunks = (1,) + image_shape
    
    # For 1D arrays (targets), use reasonable chunk size
    target_chunks = (min(1000, n_samples),)
    
    # Calculate chunk sizes for reporting
    chunk_bytes_image = np.prod(image_chunks) * 4  # float32 = 4 bytes
    chunk_mb_image = chunk_bytes_image / (1024 * 1024)
    
    print(f"Creating HDF5 with optimized per-image chunking...")
    print(f"  Image shape: {image_shape}")
    print(f"  Chunk shape: {image_chunks}")
    print(f"  Chunk size: {chunk_mb_image:.3f} MB")
    print(f"  Compression: {compression if compression else 'None (fastest)'}")
    
    with h5py.File(outpath, 'w') as f:
        # ============================================================
        # Create datasets with EXPLICIT per-image chunking
        # ============================================================
        
        # Noisy images (input data)
        ds_noisy = f.create_dataset(
            'noisy',
            data=noisy,
            chunks=image_chunks,        # CRITICAL: Per-image chunks!
            compression=compression,
            compression_opts=compression_opts,
            shuffle=(compression is not None),  # Improves compression, no speed penalty
            dtype=np.float32
        )
        
        # Clean images (ground truth)
        ds_clean = f.create_dataset(
            'clean',
            data=clean,
            chunks=image_chunks,        # CRITICAL: Per-image chunks!
            compression=compression,
            compression_opts=compression_opts,
            shuffle=(compression is not None),
            dtype=np.float32
        )
        
        # Target values (regression labels)
        # ds_I0 = f.create_dataset(
        #     'I0',
        #     data=I0s_arr,
        #     chunks=target_chunks,
        #     compression='gzip' if compression else None,  # Targets are small, gzip is fine
        #     compression_opts=1,  # Fast gzip
        #     dtype=np.float32
        # )
        
        # ============================================================
        # Store metadata as attributes (useful for later)
        # ============================================================
        
        # Dataset-level metadata
        f.attrs['n_samples'] = n_samples
        f.attrs['image_shape'] = image_shape
        f.attrs['creation_date'] = datetime.now().isoformat()
        f.attrs['dtype'] = str(noisy.dtype)
        
        # Per-dataset metadata
        ds_noisy.attrs['description'] = 'Noisy input images'
        ds_noisy.attrs['min'] = float(noisy.min())
        ds_noisy.attrs['max'] = float(noisy.max())
        ds_noisy.attrs['mean'] = float(noisy.mean())
        ds_noisy.attrs['std'] = float(noisy.std())
        
        ds_clean.attrs['description'] = 'Clean ground truth images'
        ds_clean.attrs['min'] = float(clean.min())
        ds_clean.attrs['max'] = float(clean.max())
        ds_clean.attrs['mean'] = float(clean.mean())
        ds_clean.attrs['std'] = float(clean.std())
        
        # ds_I0.attrs['description'] = 'Regression target values (I0)'
        # ds_I0.attrs['min'] = float(I0s_arr.min())
        # ds_I0.attrs['max'] = float(I0s_arr.max())
        # ds_I0.attrs['mean'] = float(I0s_arr.mean())
        # ds_I0.attrs['std'] = float(I0s_arr.std())
    
    # Calculate file size
    import os
    file_size_mb = os.path.getsize(outpath) / (1024 * 1024)
    
    # Prepare summary
    # summary = {
    #     'file_path': outpath,
    #     'file_size_mb': file_size_mb,
    #     'n_samples': n_samples,
    #     'image_shape': image_shape,
    #     'noisy_range': (float(noisy.min()), float(noisy.max())),
    #     'clean_range': (float(clean.min()), float(clean.max())),
    #     'I0_range': (float(I0s_arr.min()), float(I0s_arr.max())),
    #     'compression': compression if compression else 'none',
    #     'chunk_shape': image_chunks,
    # }

    summary = {
        'file_path': outpath,
        'file_size_mb': file_size_mb,
        'n_samples': n_samples,
        'image_shape': image_shape,
        'noisy_range': (float(noisy.min()), float(noisy.max())),
        'clean_range': (float(clean.min()), float(clean.max())),
        'compression': compression if compression else 'none',
        'chunk_shape': image_chunks,
    }
    
    print(f"\n✓ Successfully created optimized HDF5 file:")
    print(f"  Path: {outpath}")
    print(f"  File size: {file_size_mb:.2f} MB")
    #print(f"  Datasets: noisy={noisy.shape}, clean={clean.shape}, I0={I0s_arr.shape}")
    print(f"  Datasets: noisy={noisy.shape}, clean={clean.shape}")
    print(f"  Per-image chunking: ENABLED (fast random access)")
    print(f"  Expected random access time: <5 ms/sample")
    
    return summary

def save_dataset_csv(outpath, noisy_images, clean_images, I0s):
    """Save as CSV: each row contains flattened noisy pixels, flattened clean pixels, and I0 at the end.
    NOTE: CSVs are larger and slower; use HDF5 when possible.
    """
    n = len(noisy_images)
    H, W = noisy_images[0].shape
    with open(outpath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # header
        writer.writerow([f'noisy_pix_{{i}}' for i in range(H*W)] +
                        [f'clean_pix_{{i}}' for i in range(H*W)] +
                        ['I0'])
        for i in range(n):
            row = list(np.ravel(noisy_images[i])) + list(np.ravel(clean_images[i])) + [float(I0s[i])]
            writer.writerow(row)
    print(f"Wrote CSV dataset -> {outpath} (rows={n}, cols={2*H*W+1})")

def visualize_maps(datafile, save_path, n: int = 1, ground_truth: bool = True):
    if datafile[-3:] == ".h5":
        with h5py.File(datafile, "r") as f:
            data_noisy = f["noisy"]
            num_maps = data_noisy.shape[0]
            idx = np.random.randint(0, num_maps, size=n)
            idx_sorted = np.sort(idx)
            samples_noisy = data_noisy[idx_sorted]

            data_clean = f["clean"]
            samples_clean = data_clean[idx_sorted]

            print(f"Samples [{idx_sorted}] loaded.")
    elif datafile[-3:] == "npz":
        data_noisy, data_clean, _ = functions.load_npz(path=datafile)
        num_maps = data_noisy.shape[0]
        idx = np.random.randint(0, num_maps, n)
        idx_sorted = np.sort(idx)

        samples_noisy = data_noisy[idx_sorted]
        samples_clean = data_clean[idx_sorted]
    else:
        raise UnknownFileStructureError(datafile[-7:])
    
    if ground_truth:
        fig, axes = plt.subplots(n, 2, figsize=(5, 5))
        axes = np.atleast_2d(axes)
        for k in range(n):
            img_noisy = samples_noisy[k]
            img_clean = samples_clean[k]

            im_noisy = axes[k, 0].imshow(img_noisy, cmap="viridis")
            im_clean = axes[k, 1].imshow(img_clean, cmap="viridis")

            axes[k, 0].set_title(f"Noisy #{k}")
            axes[k, 1].set_title(f"Clean #{k}")

            fig.colorbar(im_noisy, ax=axes[k, 0], fraction=0.046, pad=0.04)
            fig.colorbar(im_clean, ax=axes[k, 1], fraction=0.046, pad=0.04)
    else:
        fig, axes = plt.subplots(1, n, figsize=(5, 5))
        axes = np.atleast_2d(axes)
        for k in range(n):
            img_noisy = samples_noisy[k]
            im_noisy = axes[k].imshow(img_noisy, cmap="viridis")
            axes[k].set_title("Noisy")
            fig.colorbar(im_noisy, ax=axes[k], fraction=0.046, pad=0.04)

    title = datetime.now().isoformat() + "-" + datafile
    fig.suptitle(title)

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Figure saved to: {save_path}")

    plt.close(fig)


    return None


def main():

    start_time = time.perf_counter()

    images_noisy = []
    images_clean = []
    doc   = []

    rng = np.random.default_rng(seed)

    intermediate_imgs = []

    for i in range(num_images):

        source_number = rng.integers(0, max_source_number, endpoint=True)

        # 1) β‐model
        #img, I0 = generate_beta_model_image(image_size, core_radius,
        #                                    I0=I0_fixed)

        if flux_mode == 'lin':
            img = np.zeros((image_size, image_size))
            I0 = []
            rc_used = []
            for k in range(source_number):
                x_shift = min(rng.normal(x_shift_mean, x_shift_var), x_shift_max)
                y_shift = min(rng.normal(y_shift_mean, y_shift_var), y_shift_max)
                img_current, I0_current, rc_used_current = generate_beta_model_image(
                    size=image_size,
                    rc_mean=rc_mean,
                    rc_sigma=rc_sigma,
                    x_shift=x_shift,
                    y_shift=y_shift,
                    I0_range=I0_range,      # e.g. (0.5,1.0)
                    I0_fixed=I0_fixed       # or None
                )
                intermediate_imgs.append(img_current)
                img += img_current
                I0.append(I0_current)
                rc_used.append(rc_used_current)

        elif flux_mode == 'log':
            img = np.zeros((image_size, image_size))
            I0 = []
            rc_used = []
            for k in range(source_number):
                x_shift = min(rng.normal(x_shift_mean, x_shift_var), x_shift_max)
                y_shift = min(rng.normal(y_shift_mean, y_shift_var), y_shift_max)
                img_current, I0_current, rc_used_current = generate_beta_model_image_logI0(
                    size=image_size,
                    rc_mean=rc_mean,
                    rc_sigma=rc_sigma,
                    x_shift=x_shift,
                    y_shift=y_shift,
                    I0_range=I0_range,      # e.g. (0.1,1.0)
                    I0_fixed=I0_fixed       # or None
                )
                img += img_current
                I0.append(I0_current)
                rc_used.append(rc_used_current)

        elif flux_mode == 'gauss':
            img = np.zeros((image_size, image_size))
            I0 = []
            rc_used = []
            for k in range(source_number):
                x_shift = min(rng.normal(x_shift_mean, x_shift_var), x_shift_max)
                y_shift = min(rng.normal(y_shift_mean, y_shift_var), y_shift_max)
                img_current, I0_current, rc_used_current = generate_beta_model_image_gaussI0(
                    size=image_size,
                    rc_mean=rc_mean,
                    rc_sigma=rc_sigma,
                    x_shift=x_shift,
                    y_shift=y_shift,
                    I0_mean=I0_mean,        # Gaussian mean
                    I0_sigma=I0_sigma,      # Gaussian std-dev
                    I0_fixed=I0_fixed       # or None
                )
                img += img_current
                I0.append(I0_current)
                rc_used.append(rc_used_current)
        else:
            raise ValueError(f"Unknown flux_mode '{flux_mode}'. Choose 'lin', 'log', or 'gauss'.")
        
        # 2) noise
        w = generate_white_noise(img.shape, white_noise_amplitude)

        # #THIS ONE WITH PHASE RANDOMIZATION
        # n1f = generate_1overf_noise(img.shape, one_over_f_slope,
        #                             one_over_f_amplitude)
        
        ## THIS ONE WITH STRICT HERMITIAN SYMMETRY
        n1f = strict_1overf_noise(img.shape, one_over_f_slope,
                                    one_over_f_amplitude)

        # 3) optional smoothing:
        #    White noise originates in the detector timestreams and is therefore
        #    NOT convolved with the telescope beam. Only the signal and the
        #    red (1/f, atmospheric) noise component are beam-smoothed.
        smoothed = apply_gaussian_smoothing(img + n1f, gaussian_smoothing_fwhm)
        final = smoothed + w   # add unsmoothed white noise after beam convolution

        ## ADDING AN EXTRA STEP TO REMOVE ENTIRE IMAGE MEAN <-- FOR TESTING !!
        ## final -= final.mean()

        # append to lists
        images_clean.append(img.astype(np.float32))
        images_noisy.append(final.astype(np.float32))
        doc.append({"index": i, "rc_used": rc_used, "number_sources": source_number})

    #------------------------------------------------------------------------------------------------------------------------------

    # save_images_to_csv(images, labels, output_csv)
    if save_mode == 'hdf5':
        # save_dataset_hdf5(output_h5, images_noisy, images_clean, labels_I0)
        
        # Option 1: Recommended - LZF compression (good balance)
        save_dataset_hdf5(
            output_h5, 
            images_noisy, 
            images_clean, 
            doc,
            compression='lzf'  # Fast compression, ~50% size reduction
        )
        
        # # Option 2: Maximum speed - No compression (if disk space is plentiful)
        # save_dataset_hdf5(
        #     output_h5, 
        #     images_noisy, 
        #     images_clean, 
        #     labels_I0,
        #     compression=None  # No compression, fastest I/O
        # )
        
        # # Option 3: Smaller files - Light gzip (slightly slower training)
        # save_dataset_hdf5(
        #     output_h5, 
        #     images_noisy, 
        #     images_clean, 
        #     labels_I0,
        #     compression='gzip',
        #     compression_opts=1  # Level 1 = fast, level 9 = slow
        # )
        
    else:
        save_dataset_csv(output_csv, images_noisy, images_clean, doc)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    functions.write_doc("map_making", runtime=elapsed_time, output_file=output_h5)

    print("Plotting example maps.")    
    visualize_maps(output_h5, visualization_path, n=4)
    print("Done!")
    return None

# def main():

#     #number of images
#     n = 10000
#     #directory of data
#     filename = "../data/10k_64_multicircle_32xs32xy40"

#     multiple_circles_test_maps, multiple_circles_test_doc = make_random_multiple_circle_maps(n, 64, 3, 10, 5, 5, 32, 32, 32, 32, 40, 42)
#     noise_maps = make_noise_maps(n, 64, 3, 42)
#     multiple_circles_test_maps_withNoise = noise_maps + multiple_circles_test_maps
#     save_maps(multiple_circles_test_maps_withNoise, multiple_circles_test_maps, multiple_circles_test_doc, filename)
#     return None

if __name__ == "__main__":
    print("Executing main() in maps.py")
    main()