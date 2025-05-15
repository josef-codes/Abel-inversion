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


def otsu_binary_mask(
        img: np.ndarray,
        method: str = 'multiotsu',
        n_classes: int = 3,
        threshold_index: int = -1,
        remove_small: int = 10,
        close_size: int = 0,
        gaussian_sigma: float = 0,
        keep_largest: bool = False
) -> Tuple[np.ndarray, float]:
    """
    Extract a binary mask via Otsu/multi-Otsu, clean it, and optionally
    keep only the largest component.

    Parameters
    ----------
    img : 2D np.ndarray
        Input image (phase or density map).
    method : {'otsu','multiotsu'}
        Single- or multi-level Otsu thresholding.
    n_classes : int
        Number of classes for multi-Otsu.
    threshold_index : int
        Which threshold boundary to pick from `threshold_multiotsu`:
           -1 = highest (keep only top class),
           -2 = second highest (keep top two classes), etc.
    remove_small : int
        Minimum object size (px) to keep.
    close_size : int
        Kernel size for binary closing (px). 0 = skip.
    gaussian_sigma : float
        If >0, Gaussian-smooth the mask by this σ before final threshold.
    keep_largest : bool
        If True, retain only the single largest connected component.

    Returns
    -------
    mask : 2D bool array
        The cleaned binary mask.
    thr  : float
        The numeric threshold used.
    """
    if method == 'otsu':
        thr = threshold_otsu(img)
        mask = img > thr
    elif method == 'multiotsu':
        # Look at how many unique values we really have:
        unique_vals = np.unique(img.ravel())
        # If there are fewer values than classes, do a single Otsu:
        if unique_vals.size < n_classes:
            thr = float(threshold_otsu(img))
            mask = img > thr
        else:
            try:
                threshs = threshold_multiotsu(img, classes=n_classes)
            except ValueError:
                # fallback if skimage still complains
                thr = float(threshold_otsu(img))
                mask = img > thr
            else:
                thr = float(threshs[threshold_index])
                regions = np.digitize(img, bins=threshs)
                mask = (regions == n_classes - 1)
    else:
        raise ValueError("method must be 'otsu' or 'multiotsu'")

    if remove_small > 0:
        mask = remove_small_objects(mask, min_size=remove_small)

    if close_size > 0:
        struct = np.ones((close_size, close_size), dtype=bool)
        mask = binary_closing(mask, structure=struct)

    if gaussian_sigma > 0:
        blurred = gaussian_filter(mask.astype(float), sigma=gaussian_sigma)
        mask = blurred >= 0.5

    if keep_largest:
        lbl = label(mask)
        props = regionprops(lbl)
        if props:
            largest = max(props, key=lambda p: p.area).label
            mask = (lbl == largest)

    return mask, thr


def extract_main_positive_region(img: np.ndarray) -> np.ndarray:
    """
    Return a mask of the largest contiguous region where img > 0.

    Parameters
    ----------
    img : 2D np.ndarray
        Input array (e.g. phase or density) containing positive, zero, and negative values.

    Returns
    -------
    mask : 2D bool ndarray
        True for pixels in the largest connected component where img > 0; False elsewhere.
    """
    # 1) initial positive‐value mask
    pos_mask = img > 0

    # 2) label connected regions of positive pixels
    lbl = label(pos_mask)

    # 3) find the largest region by area
    props = regionprops(lbl)
    if not props:
        return np.zeros_like(pos_mask, dtype=bool)

    largest_label = max(props, key=lambda p: p.area).label
    # 4) return mask of that region
    return lbl == largest_label


def fill_bottom_rectangle(mask: np.ndarray) -> np.ndarray:
    """
    Given a 2D boolean mask, find the horizontal span and bottom edge
    of the True region, and then extend that span down to the bottom of
    the image in a solid rectangle.

    Parameters
    ----------
    mask : 2D bool array, shape (H, W)
        Your initial mask (True = foreground).

    Returns
    -------
    new_mask : 2D bool array, shape (H, W)
        A copy of `mask` but with everything in the rectangle
        spanning [left..right] × [bottom..H-1] set to True, where:
          • left   = first column containing any True
          • right  = last  column containing any True
          • bottom = largest row index containing any True
    """
    H, W = mask.shape
    # find all True coordinates
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        # nothing to fill
        return mask.copy()

    # 1) leftmost and rightmost columns
    left = xs.min()
    right = xs.max()

    # 2) bottom-most row
    bottom = ys.max()

    # 3) build new mask
    new_mask = mask.copy()
    new_mask[bottom:H, left:right + 1] = True

    return new_mask
