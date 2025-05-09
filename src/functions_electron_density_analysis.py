import numpy as np


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