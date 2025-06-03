import matplotlib.pyplot as plt
import tifffile as tiff # Loads .tiff as a NumPy array
import os

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

def find_fourier_peaks(
    img: np.ndarray,
    *,
    sigma: float = 3,
    min_distance: int = 10,
    threshold_rel: float = 0.3
):
    """
    Locate the centre (0‑th order) and the ±1‑st‑order peaks
    in a 2‑D Fourier‑spectrum image.

    Parameters
    ----------
    img : 2‑D ndarray
        Grayscale Fourier‑spectrum (uint8/uint16/float).
    sigma : float
        Gaussian smoothing prior to peak detection.
    min_distance : int
        Minimum pixel distance between detected peaks
        (passed to skimage.feature.peak_local_max).
    threshold_rel : float
        Relative intensity threshold for peak detection.

    Returns
    -------
    dict
        {
          'center': (row, col),
          'plus1' : (row, col),   # x > centre  (‑> +1 order)
          'minus1': (row, col)    # x < centre  (‑> −1 order)
        }

    Notes
    -----
    * The function looks for **three** bright spots:
        the one closest to the geometric centre is the 0‑th order;
        of the remaining peaks it finds the pair that are most
        nearly opposite each other (180 ± a few degrees).
    * Works best when the ±1 peaks are the brightest objects
      after the central spot (typical for diffraction/Fourier patterns).
    """
    if img.ndim != 2:
        raise ValueError("`img` must be a 2‑D grayscale array")
    if np.iscomplexobj(img):
        img = np.abs(img)
    else:
        img = img.astype(float)
    # --- 1. smooth + normalise -------------------------------------------
    img_f = gaussian_filter(img.astype(float), sigma=sigma)
    if img_f.max() > 0:
        img_f /= img_f.max()

    # --- 2. find candidate peaks -----------------------------------------
    peaks = peak_local_max(
        img_f,
        min_distance=min_distance,
        threshold_rel=threshold_rel,
        num_peaks=9               # a few more than we actually need
    )

    if len(peaks) < 3:
        raise RuntimeError("Could not find at least three peaks")

    h, w = img.shape
    centre_geom = np.array([h / 2, w / 2])

    # --- 3. pick the centre peak: closest to image centre ----------------
    dists = np.linalg.norm(peaks - centre_geom, axis=1)
    centre_idx = np.argmin(dists)
    centre_peak = peaks[centre_idx]
    remaining   = np.delete(peaks, centre_idx, axis=0)

    # --- 4. from remaining peaks, choose the most opposite pair ----------
    # Score each pair by the magnitude of (v1 + v2); the smaller, the more opposite.
    best_pair   = None
    best_score  = np.inf
    for i in range(len(remaining)):
        for j in range(i + 1, len(remaining)):
            v1 = remaining[i] - centre_peak
            v2 = remaining[j] - centre_peak
            score = np.linalg.norm(v1 + v2)    # 0 when perfectly opposite
            if score < best_score:
                best_score = score
                best_pair  = (remaining[i], remaining[j])

    if best_pair is None:
        raise RuntimeError("Failed to identify ±1 order peaks")

    # --- 5. label pair so that plus1 has x > centre ----------------------
    p1, p2 = best_pair
    if p1[1] > centre_peak[1]:       # compare column (x) coordinate
        plus1, minus1 = p1, p2
    else:
        plus1, minus1 = p2, p1

    return {
        "center": tuple(int(v) for v in centre_peak),
        "plus1" : tuple(int(v) for v in plus1),
        "minus1": tuple(int(v) for v in minus1),
    }




## -- import .tiff image ---
base_dir_1 = r"C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\H0_3_28_25\(1) 50-1000ns"
file_path_1 = os.path.join(base_dir_1, "M2_X11.tif")
base_dir_2 = r"C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\H0_3_28_25\reference x4"
file_path_ref = os.path.join(base_dir_2, "REF_X7.tif")
img1 = tiff.imread(file_path_1)  
img_ref = tiff.imread(file_path_ref)


## -- for loop to get phase difference ---
for i in range(2):
    if i == 0:
        img = img1
    else:
        img = img_ref
    # crop image
    #img_crop = img[116:238, 289:519] 
    # padding image
    img_padded = np.pad(img, pad_width = 15, mode = 'constant') # constant, wrap, symmetric

    ## -- Fourier transform ---
    # Compute the FFT and shift the zero frequency component to the center
    fft_image = np.fft.fft2(img_padded)
    fft_shifted = np.fft.fftshift(fft_image)

    coords = find_fourier_peaks(fft_shifted)

    # magnitude spectrum |fft_shifted| - (log-scaled for visibility)
    magnitude_spectrum = np.log1p(np.abs(fft_shifted))
    plt.imshow(magnitude_spectrum)
    plt.show()
    # magnitude_spectrum = np.abs(fft_shifted)
    # phase spectrum angle(fft_shifted)
    phase_spectrum = np.angle(fft_shifted)


    ## -- Fourier transform ---
    rows, cols = magnitude_spectrum.shape
    # create elliptical gaussian filter placed in [x, y] coordinate
    # Get Fourier center
    y_center, x_center = rows // 2, cols // 2 # rounded "floor" division operator
    # Apply the shift to the Fourier center


    # —— pick out individual tuples ——
    center  = coords['center']   # (137, 522)
    plus1   = coords['plus1']    # (121, 522)
    minus1  = coords['minus1']   # (153, 522)
    # —— unpack into two separate numbers ——
    shifted_peak_y, shifted_peak_x = plus1


    ## --- create elliptical gaussian mask ---

    if i == 0:
        # Define the elliptical mask centered at the shifted location
        a, b = cols // 9, rows // 9  # Semi-axes of the ellipse
        sgm = 0
    else:
        a, b = cols // 11, rows // 11
        sgm = 0
    # Create a coordinate 1D array...meshgrid
    y, x = np.ogrid[:rows, :cols] # y = vektor hodnot 0 až 144 (výška obr), x = vektor hodnot 0 až 250 (šířka obr), 
    # Define the elliptical mask equation at the new location
    ellipse_mask = (((x - shifted_peak_x) / a) ** 2 + ((y - shifted_peak_y) / b) ** 2) <= 1
    # Generate a Gaussian fade-out mask at the new location
    gaussian_fade = np.zeros_like(ellipse_mask, dtype=float) # zeros matrix with the size of ellipse_mask
    gaussian_fade[ellipse_mask] = 1
    gaussian_fade = gaussian_filter(gaussian_fade, sigma=sgm)  # Smooth transition

    ## --- Extract phase from 2D FFT ---
    # Apply the shifted mask to the Fourier spectrum
    filtered_fft = fft_shifted * gaussian_fade
    # Compute the inverse FFT
    ifft_shifted = np.fft.ifftshift(filtered_fft)
    reconstructed_image = np.fft.ifft2(ifft_shifted)
    phase_im = np.angle(reconstructed_image)  # Extract phase
    if i == 0:
        phase_im1 = phase_im
    else:
        phase_im_ref = phase_im

## --- plot ---
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
im1 = ax[0].imshow(phase_im1, cmap='cividis')
ax[0].set_title("Phase image 1")
fig.colorbar(im1, ax=ax[0])
im_ref = ax[1].imshow(phase_im_ref, cmap='cividis')
ax[1].set_title("Phase image reference")
fig.colorbar(im_ref, ax=ax[1])
plt.show()

## --- Phase shift image ---
# substract phases
phase_shift = phase_im1 - phase_im_ref
phase_shift = np.mod(phase_shift, 2 * np.pi) # modulo 2pi
plt.imshow(phase_shift, cmap='winter')
plt.title("Modulo Phase shift")
plt.colorbar(label="phase")
plt.show()

## --- Phase UNWRAP with step function algorithms ---
# openCV library
# Skimage library
from skimage.restoration import unwrap_phase
image_unwrapped = unwrap_phase(phase_shift, wrap_around=(False, False))

plt.imshow(image_unwrapped, cmap='jet')
plt.colorbar()
plt.title('Unwrapped Phase Map')
plt.show()

########################################################
#### Abel equation solving algorithm implementation ####
########################################################
## IDEA
# 1) Fourier Method -> v článku Gorbushkin, Shabanov
# 2) f-Interpolation
# 3) Backus-Gilbert-Method
## Other algorithms:
# Fourier-Hankel transform
# https://pyabel.readthedocs.io/en/v0.8.2/transform_methods/fh.html
# Pyable package:
# https://pyabel.readthedocs.io/en/latest/abel.html