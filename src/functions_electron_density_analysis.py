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
        remove_small: int = 10,
        close_size: int = 0,
        gaussian_sigma: float = 0,
        keep_largest: bool = False
) -> Tuple[np.ndarray, float]:
    """
    Extract the bright‐peak mask via Otsu or multi‐Otsu, clean it, and
    optionally blur

    Returns
    -------
    mask : 2D bool array
    thr  : float
    """
    if method == 'otsu':
        thr = threshold_otsu(img)
        mask = img > thr

    elif method == 'multiotsu':
        # <-- must use 'classes', not 'n_classes'
        threshs = threshold_multiotsu(img, classes=n_classes)
        thr = float(threshs[-1])
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
