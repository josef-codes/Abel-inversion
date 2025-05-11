import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.restoration import unwrap_phase
from typing import Optional, Tuple


def find_fourier_peaks(
    img: np.ndarray,
    exclude_radius: int = 8,
    smooth_sigma: float = 2
) -> dict:
    """
    Simple detection of the DC and first-order peaks in the FFT magnitude by:
      1. Computing the centered FFT magnitude.
      2. Optionally smoothing it.
      3. Zeroing out a small circular region around DC.
      4. Picking the two brightest remaining pixels as ±1 orders.

    Parameters
    ----------
    img : 2D ndarray
        Spatial-domain (padded) image.
    exclude_radius : int
        Radius (pixels) around the center to mask out DC.
    smooth_sigma : float
        If >0, Gaussian-smooth the magnitude before peak selection.

    Returns
    -------
    dict with keys:
      'center' : (row, col) of DC component,
      'plus1'  : (row, col) of the +1 order peak,
      'minus1' : (row, col) of the -1 order peak.
    """
    # 1. FFT and shift
    F = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(F)

    # 2. optional smoothing
    if smooth_sigma > 0:
        mag = gaussian_filter(mag, sigma=smooth_sigma)

    # 3. center coordinates
    rows, cols = mag.shape
    cy, cx = rows // 2, cols // 2

    # 4. mask out DC region
    y, x = np.ogrid[:rows, :cols]
    dc_mask = (y - cy)**2 + (x - cx)**2 <= exclude_radius**2
    mag_nodc = mag.copy()
    mag_nodc[dc_mask] = 0

    # 5. find two brightest peaks
    flat = mag_nodc.ravel()
    idx1 = np.argmax(flat)
    flat[idx1] = 0  # suppress first peak
    idx2 = np.argmax(flat)

    peak1 = divmod(idx1, cols)
    peak2 = divmod(idx2, cols)

    # classify by y < center (rows index)
    if peak1[0] < cy:
        plus1, minus1 = peak1, peak2
    else:
        plus1, minus1 = peak2, peak1

    return {
        'center':  (cy, cx),
        'plus1':   plus1,
        'minus1':  minus1
    }


def create_ellipse_mask(
    shape: tuple,
    x_center: float,
    y_center: float,
    a: float,
    b: float,
    sigma: float = 0.0
) -> np.ndarray:
    """
    Create an elliptical mask (optionally Gaussian‐smoothed) for an image.

    Parameters
    ----------
    shape : tuple of int
        The image shape as (rows, cols):
        rows = number of vertical pixels (height),
        cols = number of horizontal pixels (width).
    x_center : float
        X‐coordinate of the ellipse center, i.e. the column index.
    y_center : float
        Y‐coordinate of the ellipse center, i.e. the row index.
    a : float
        Semi‐axis length along the x‐direction (horizontal radius in pixels).
    b : float
        Semi‐axis length along the y‐direction (vertical radius in pixels).
    sigma : float, optional
        If zero, returns a boolean mask. If > 0, returns a float mask
        smoothed by a Gaussian of this standard deviation.

    Returns
    -------
    mask : ndarray
        Elliptical mask of shape (rows, cols). Boolean if sigma == 0,
        float in [0, 1] if sigma > 0.
    """
    rows, cols = shape
    # y is row indices (0…rows-1), x is column indices (0…cols-1)
    y, x = np.ogrid[:rows, :cols]

    # binary ellipse equation ((x-x0)/a)^2 + ((y-y0)/b)^2 <= 1
    mask = ((x - x_center) / a) ** 2 + ((y - y_center) / b) ** 2 <= 1

    if sigma > 0:
        # convert to float and apply Gaussian smoothing
        mask = mask.astype(float)
        mask = gaussian_filter(mask, sigma=sigma, mode='constant')

    return mask


def circular_exclusion_mask(
    shape: Tuple[int, int],
    center: Tuple[int, int],
    radius: int,
    dtype: np.dtype = float
) -> np.ndarray:
    """
    Create a binary mask of given `shape` where pixels inside `radius`
    of `center` are 0, and all others are 1.

    Parameters
    ----------
    shape : (H, W)
        Size of the mask to create.
    center : (y0, x0)
        Coordinates of the circle center.
    radius : int
        Radius (in pixels) of the exclusion zone.
    dtype : data-type
        Desired data-type of the mask (e.g. bool, float).

    Returns
    -------
    mask : ndarray of shape (H, W)
    """
    H, W = shape
    y0, x0 = center
    y, x = np.ogrid[:H, :W]
    mask = np.ones((H, W), dtype=dtype)
    mask[((y - y0)**2 + (x - x0)**2) <= radius**2] = 0
    return mask


def extract_phase_from_mask(
    img_padded: np.ndarray,
    mask: np.ndarray,
    peak_coord: Tuple[int, int],
    pad: int = 0,
    inner_radius_mask: Optional[int] = None
) -> np.ndarray:
    """
    Apply a pre-computed elliptical mask (and optionally a binary circular
    mask at the spectrum center) in the Fourier domain and return the phase,
    optionally cropping away the padded border.

    Parameters
    ----------
    img_padded : 2D ndarray
        Zero-padded image (rows = height, cols = width).
    mask : 2D ndarray of same shape
        Elliptical mask (boolean or float) created around peak_coord.
    peak_coord : (y, x)
        The row (y) and column (x) that defined the center of the ellipse.
    pad : int, optional
        Number of pixels of padding on each side to remove from the output.
        If zero (default), no cropping is performed.
    inner_radius_mask : int, optional
        If provided, radius (in pixels) of an additional binary circular mask
        centered on the Fourier-domain center (zero-frequency). Inside the
        circle: 1, outside: 0. The final mask = elliptical mask * circular mask.

    Returns
    -------
    phase_im : 2D ndarray
        Phase of the inverse FFT of the masked spectrum, cropped to
        `phase_im[pad:-pad, pad:-pad]` if `pad > 0`.
    """

    # 1. Forward FFT & shift zero‐freq to center
    f = np.fft.fftshift(np.fft.fft2(img_padded))

    # 2. Build the combined mask
    combined_mask = mask.copy()
    if inner_radius_mask is not None:
        H, W = mask.shape
        cy, cx = H // 2, W // 2

        # create a binary circular mask: 1 inside, 0 outside
        y, x = np.ogrid[:H, :W]
        circular_mask = np.ones_like(mask, dtype=mask.dtype)
        circular_mask[((y - cy)**2 + (x - cx)**2) <= inner_radius_mask**2] = 0

        # now multiply with the elliptical mask
        combined_mask = combined_mask * circular_mask

    # 3. Apply mask in Fourier domain
    f_masked = f * combined_mask

    # 4. Inverse shift & inverse FFT, then extract phase
    rec = np.fft.ifft2(np.fft.ifftshift(f_masked))
    phase = np.angle(rec)

    # 5. Crop away padding border if requested
    if pad > 0:
        return phase[pad:-pad, pad:-pad]
    return phase


def compute_phase_from_padded(
    img_padded: np.ndarray,
    order: str = 'plus1',
    exclude_radius: int = 10,
    smooth_sigma: float = 0.0,
    a_div: int = 9,
    b_div: int = 9,
    mask_sigma: float = 0.0,
    pad: int = 0
) -> np.ndarray:
    """
    Locate a given Fourier peak in a zero-padded image (using simple top-two method),
    build an elliptical mask around it, and extract the phase image.

    Parameters
    ----------
    img_padded : 2D ndarray
        Zero-padded input image.
    order : {'center','plus1','minus1'}
        Which peak to isolate.
    exclude_radius : int
        Radius around DC to mask out before picking peaks.
    smooth_sigma : float
        Gaussian sigma for pre-peak FFT-magnitude smoothing.
    a_div, b_div : int
        Divisors for ellipse semi-axes: a = cols//a_div, b = rows//b_div.
    mask_sigma : float
        If >0, Gaussian-smooth the binary mask by this sigma.
    pad : int
        Number of pixels of padding on each side to remove from the output.

    Returns
    -------
    phase_image : 2D ndarray
        Phase of the inverse FFT of the masked Fourier component,
        cropped to original size if pad>0.
    """
    try:
        coords = find_fourier_peaks(
            img=img_padded,
            exclude_radius=exclude_radius,
            smooth_sigma=smooth_sigma
        )
        y0, x0 = coords[order]
        rows, cols = img_padded.shape
        a = cols // a_div
        b = rows // b_div
        mask = create_ellipse_mask(
            shape=(rows, cols),
            x_center=x0,
            y_center=y0,
            a=a,
            b=b,
            sigma=mask_sigma
        )
        return extract_phase_from_mask(
            img_padded=img_padded,
            mask=mask,
            peak_coord=(y0, x0),
            pad=pad,
            inner_radius_mask=3
        )
    except ZeroDivisionError:
        print("Warning: Division by zero in compute_phase_from_padded. Skipping.")
        try:
            if order == 'minus1':
                order = 'plus1'
            else:
                order = 'minus1'
            y0, x0 = coords[order]
            rows, cols = img_padded.shape
            a = cols // a_div
            b = rows // b_div
            mask = create_ellipse_mask(
                shape=(rows, cols),
                x_center=x0,
                y_center=y0,
                a=a,
                b=b,
                sigma=mask_sigma
            )
            return extract_phase_from_mask(
                img_padded=img_padded,
                mask=mask,
                peak_coord=(y0, x0),
                pad=pad,
                inner_radius_mask=3
            )
        except:
            return img_padded  # Or np.zeros_like(img_padded), or raise, depending on downstream needs



def unwrap_phase_image(
    phase_image: np.ndarray
) -> np.ndarray:
    """
    Perform a 2D phase unwrapping on the input wrapped-phase image.

    Parameters
    ----------
    phase_image : np.ndarray, shape (H, W)
        Wrapped phase map (values in the range [-π, π] or similar).

    Returns
    -------
    unwrapped : np.ndarray, shape (H, W)
        Unwrapped phase map, continuous over the image.
    """
    # ensure array
    arr = np.asarray(phase_image, dtype=float)
    # call scikit-image’s unwrap
    unwrapped = unwrap_phase(arr)
    return unwrapped


def crop_img_base(image: np.ndarray, n_bottom: int) -> np.ndarray:
    """
    Crop a given number of pixels from the bottom of an image.

    Parameters
    ----------
    image : np.ndarray
        Input image array of shape (H, W) or (H, W, C).
    n_bottom : int
        Number of rows to remove from the bottom. Must be ≥ 0.

    Returns
    -------
    cropped : np.ndarray
        Image array of shape (H−n_bottom, W) or (H−n_bottom, W, C).
    """
    if n_bottom < 0:
        raise ValueError(f"n_bottom must be non-negative, got {n_bottom}")
    H = image.shape[0]
    if n_bottom >= H:
        # return an empty array with zero height
        return image[:0].copy()
    return image[:H - n_bottom].copy()

