import numpy as np
from typing import Tuple

from scipy.signal import find_peaks
from scipy.ndimage import binary_closing, gaussian_filter

from skimage.filters import threshold_otsu, threshold_multiotsu
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops


def masked_average(image: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute the average of `image` over the region where `mask == 1`.

    Parameters
    ----------
    image : np.ndarray
        2D (or ND) array of values.
    mask : np.ndarray
        Same shape as image; values between 0 and 1. Only pixels
        where mask == 1 are counted.

    Returns
    -------
    avg : float
        The mean of image[mask == 1]. Returns nan if no pixels selected.
    """
    if image.shape != mask.shape:
        raise ValueError("`image` and `mask` must have the same shape")
    # Boolean array of the selected region
    sel = (mask == 1)
    # If nothing selected, avoid division by zero
    if not np.any(sel):
        return float('nan')
    # Compute and return the mean over that region
    return image[sel].mean()


def otsu_binary_mask_simple(
        img: np.ndarray,
        invert: bool = False,
        min_size: int = 0,
        min_hole_size: int = 0,
        keep_largest: bool = False
) -> Tuple[np.ndarray, float]:
    """
    Compute and clean a binary mask of `img` by Otsu’s threshold.

    Steps
    -----
    1) Compute Otsu threshold.
    2) Build initial mask (img > thresh or img < thresh).
    3) Remove small objects (< min_size).
    4) Fill small holes (< min_hole_size).
    5) Optionally keep only the largest connected component.

    Parameters
    ----------
    img : 2D array
        Input image (grayscale).
    invert : bool
        If False, mask = (img > thresh); if True, mask = (img < thresh).
    min_size : int
        Remove any foreground islands smaller than this (in pixels).
    min_hole_size : int
        Fill any background holes smaller than this (in pixels).
    keep_largest : bool
        If True, after cleaning, only the single largest component is retained.

    Returns
    -------
    mask : 2D bool array
        Cleaned binary mask (True = foreground).
    thresh : float
        The Otsu‐computed intensity threshold.
    """
    # 1) compute threshold
    thresh = threshold_otsu(img)

    # 2) initial mask
    if invert:
        mask = img < thresh
    else:
        mask = img > thresh

    # 3) remove small objects
    if min_size > 0:
        mask = remove_small_objects(mask, min_size)

    # 4) fill small holes
    if min_hole_size > 0:
        mask = remove_small_holes(mask, area_threshold=min_hole_size)

    # 5) keep only the largest component
    if keep_largest:
        labeled = label(mask)
        props = regionprops(labeled)
        if props:
            largest = max(props, key=lambda p: p.area).label
            mask = (labeled == largest)

    return mask, float(thresh)


def otsu_binary_mask(
        img: np.ndarray,
        method: str = 'multiotsu',
        classes: int = 3,
        remove_small: int = 10,
        close_size: int = 0,
        gaussian_sigma: float = 0,
        keep_largest: bool = False
) -> Tuple[np.ndarray, float]:
    """
    Extract the very bright peak region from `img`, clean it, and optionally blur.

    Steps:
      1) Threshold (Otsu or multi-Otsu).
      2) Remove small objects.
      3) Optionally binary-close with a square structuring element.
      4) Optionally Gaussian-blur the mask and re-binarize.
      5) Optionally keep only the largest connected component.

    Parameters
    ----------
    img : 2D array
        Input image.
    method : {'otsu','multiotsu'}
        Thresholding method.
    classes : int
        Number of classes for multi-Otsu.
    remove_small : int
        Remove islands smaller than this.
    close_size : int
        If >0, square closing of this size.
    gaussian_sigma : float
        If >0, sigma for Gaussian blur applied to the binary mask,
        then re-thresholded at 0.5 back to binary.
    keep_largest : bool
        If True, only keep the largest connected component.

    Returns
    -------
    mask : 2D bool array
        Final cleaned (and optionally blurred) bright-peak mask.
    thr : float
        The threshold used (Otsu value or multi-Otsu top-class edge).
    """
    # 1) threshold
    if method == 'otsu':
        thr = threshold_otsu(img)
        mask = img > thr
    elif method == 'multiotsu':
        threshs = threshold_multiotsu(img, classes=classes)
        thr = float(threshs[-1])
        regions = np.digitize(img, bins=threshs)
        mask = (regions == classes - 1)
    else:
        raise ValueError("method must be 'otsu' or 'multiotsu'")

    # 2) remove small islands
    if remove_small > 0:
        mask = remove_small_objects(mask, min_size=remove_small)

    # 3) binary closing
    if close_size > 0:
        struct = np.ones((close_size, close_size), dtype=bool)
        mask = binary_closing(mask, structure=struct)

    # 4) optional Gaussian blur on the mask
    if gaussian_sigma > 0:
        # blur the mask (as float), then threshold at 0.5
        blurred = gaussian_filter(mask.astype(float), sigma=gaussian_sigma)
        mask = blurred >= 0.5

    # 5) keep largest component
    if keep_largest:
        lbl = label(mask)
        props = regionprops(lbl)
        if props:
            largest = max(props, key=lambda p: p.area).label
            mask = (lbl == largest)

    return mask, thr
