import numpy as np
from skimage import io, exposure
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cmasher as cmr


def detect_bit_depth(arr: np.ndarray) -> int:
    """Return the bit depth (8, 16, etc.) of an integer image array."""
    if np.issubdtype(arr.dtype, np.integer):
        return arr.dtype.itemsize * 8
    raise ValueError(f"Unsupported dtype {arr.dtype}")

def stretch_contrast(arr: np.ndarray, clip_percent: float = 0.0) -> np.ndarray:
    """
    Linearly stretch the input image array to its full integer range,
    optionally clipping the lowest/highest clip_percent% of pixels first.
    
    Parameters
    ----------
    arr : np.ndarray
        Input integer image array.
    clip_percent : float
        Percentage of pixels to clip at low and high end (e.g. 1.0 for 1%).
    
    Returns
    -------
    np.ndarray
        Contrast‑stretched array in same dtype as input.
    """
    b = detect_bit_depth(arr)
    vmax = 2**b - 1

    if clip_percent > 0:
        # compute percentiles and clip those ranges before stretching
        p_low, p_high = np.percentile(arr, (clip_percent, 100.0 - clip_percent))
        in_range = (p_low, p_high)
    else:
        in_range = 'image'

    stretched = exposure.rescale_intensity(
        arr,
        in_range=in_range,
        out_range=(0, vmax)
    ).astype(arr.dtype)
    return stretched

def gamma_correction(arr: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """
    Apply gamma correction to an integer image array.
    
    1) Normalize to [0,1]
    2) apply gamma
    3) remap to original integer range
    """
    b = detect_bit_depth(arr)
    vmax = 2**b - 1
    norm = arr.astype(np.float64) / vmax
    corrected = exposure.adjust_gamma(norm, gamma=gamma)
    return (corrected * vmax).astype(arr.dtype)

if __name__ == "__main__":
    # ─── CONFIG ────────────────────────────────────────────────
    path = r'C:\\Users\\User\\z\\Desktop\\WUT\\Diplomka\\DATA\\Images\\H0_3_28_25\\50-1000ns\\M1_X14.tif' # MI_X14.tif
    clip_percent = 2    # clip 1% at low/high before stretching
    gamma_value = 1.2     # e.g. <1 to brighten shadows
    sgm = 1               # denoise
    # ────────────────────────────────────────────────────────────

    # 0) Load original (auto-dtype)
    orig = io.imread(path)

    # 1) Noise filter
    filtered = gaussian_filter(orig, sigma=sgm)  # Smooth transition

    # 2) Contrast stretch with percentile clipping
    stretched = stretch_contrast(filtered, clip_percent=clip_percent)

    # 3) Gamma correction on the stretched result
    gamma_img = gamma_correction(stretched, gamma=gamma_value)

    # 4) Display side‑by‑side
    b = detect_bit_depth(orig)
    vmax = 2**b - 1

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    images = [orig, filtered, stretched, gamma_img]
    titles = ['Original',f'Gauss filt (sigma={sgm})', f'Stretched (clip={clip_percent}%)', f'After Gamma (γ={gamma_value})']
    colormap = getattr(cmr.cm, 'nuclear')
    
    for i, img in enumerate(images):
        # Display image
        ax_img = axes[0, i]
        ax_img.imshow(img, cmap=colormap, vmin=0, vmax=vmax)
        ax_img.set_title(titles[i])
        ax_img.axis('off')
        
        # Display histogram
        ax_hist = axes[1, i]
        counts, bin_edges = np.histogram(img.ravel(), bins=2**16, range=(0, vmax))
        ax_hist.hist(img.ravel(), bins=256)
        ax_hist.set_ylim(0, counts.max() * 1.05)
        ax_hist.set_xlim(2, vmax)
        ax_hist.set_xlabel('Intensity')
        ax_hist.set_ylabel('Count')
        ax_hist.set_title('Histogram')

plt.tight_layout()
plt.show()