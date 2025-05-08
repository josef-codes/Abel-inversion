import numpy as np
from scipy.ndimage import gaussian_filter
from abel.basex import basex_transform
from abel.hansenlaw import hansenlaw_transform


def make_semicircle_mask(shape: tuple[int, int],
                         center: tuple[float, float],
                         radius: float,
                         smooth_sigma: float | None = None) -> np.ndarray:
    """
    Create a bottom-centered semicircle mask.

    Parameters
    ----------
    shape : (H, W)
        Size of the output mask.
    center : (y0, x0)
        Coordinates of the semicircle center.
    radius : float
        Radius of the semicircle in pixels.
    smooth_sigma : float | None
        If given, apply a Gaussian filter with this sigma to soften the
        mask edges. Default is None (no smoothing).

    Returns
    -------
    mask : np.ndarray, shape (H, W)
        Float mask, 1 inside the semicircle (y <= y0 and (x−x0)^2+(y−y0)^2 ≤ R^2),
        0 outside. If `smooth_sigma` is set, values will taper between 0 and 1.
    """
    H, W = shape
    y0, x0 = center
    yy, xx = np.indices((H, W))
    # full circle condition
    circle = (xx - x0)**2 + (yy - y0)**2 <= radius**2
    # semicircle: pixels above or at center row
    semi = yy <= y0
    mask = (circle & semi).astype(float)
    if smooth_sigma:
        # blur the hard edge, then re-normalize to [0,1]
        mask = gaussian_filter(mask, sigma=smooth_sigma)
        mask = (mask - mask.min()) / (mask.max() - mask.min())
    return mask


# -----------------------------v1
def symmetrize_via_fft(profile: np.ndarray) -> np.ndarray:
    """
    Enforce even symmetry by zeroing the odd Fourier components.

    Parameters
    ----------
    profile : 1D array
        Original, possibly asymmetric data.

    Returns
    -------
    h_sym : 1D array
        Symmetrized profile: h_sym(x)=½[h(x)+h(-x)].
    """
    # 1) forward FFT
    H = np.fft.fft(profile)
    # 2) keep only the real part (even component)
    H_even = np.real(H)
    # 3) inverse FFT back to real space
    h_sym = np.fft.ifft(H_even)
    return np.real(h_sym)


# -----------------------------v2
def remove_linear_tilt(profile: np.ndarray) -> np.ndarray:
    """
    Subtract the straight line through the end points from the data.

    Parameters
    ----------
    profile : 1D array of length n

    Returns
    -------
    detrended : 1D array of length n
    """
    n = profile.size
    x = np.arange(n)
    # fit slope and intercept through first and last point
    m, c = np.polyfit([0, n - 1], [profile[0], profile[-1]], 1)
    return profile - (m * x + c)


def symmetrize_profile(
        profile: np.ndarray,
        side: str = "both"
) -> np.ndarray:
    """
    Vrací symetrický profil, ale shiftnutý na kraj.
    Enforce symmetry about the midpoint by copying or averaging.

    Parameters
    ----------
    profile : 1D array of length n
    side : 'left', 'right', or 'both'
      'left'  : mirror left half onto right
      'right' : mirror right half onto left
      'both'  : average left and right about center

    Returns
    -------
    sym : 1D array of length n
    """
    n = profile.size
    mid = n // 2
    left = profile[:mid]
    right = profile[-mid:][::-1]
    sym = profile.copy()

    if side == "left":
        sym[mid:mid + mid] = left[::-1]
    elif side == "right":
        sym[:mid] = right[::-1]
    else:  # both
        avg = 0.5 * (left + right)
        sym[:mid] = avg[::-1]
        sym[-mid:] = avg

    return sym


# -------- Automated image symmetrisation -------
def mirror_array(
    arr: np.ndarray,
    flip_sign: bool = False
) -> np.ndarray:
    """
    Mirror a 1D array about its first element.

    Parameters
    ----------
    arr : 1D array
        Values from index 0 outward.
    flip_sign : bool
        If True, the mirrored (left) part will be negated.
        Use this for coordinate axes; leave False for data.

    Returns
    -------
    full : 1D array
        Concatenation of mirrored left + original right.
        Length = 2*len(arr) - 1.
    """
    left = arr[1:][::-1]
    if flip_sign:
        left = -left
    return np.concatenate([left, arr])


def mirror_image(half_image: np.ndarray) -> np.ndarray:
    """
    Mirror a 2D half-image about its first column to produce a full symmetric image.

    Parameters
    ----------
    half_image : 2D np.ndarray, shape (H, W)
        The “right-hand” half of your image, where column 0 is on the symmetry axis.

    Returns
    -------
    full_image : 2D np.ndarray, shape (H, 2*W-1)
        Concatenation of the mirrored left half (excluding the axis column)
        with the original half_image.
    """
    # take all columns except the first (the axis), and flip them left-right
    left = half_image[:, 1:][:, ::-1]
    # stitch the left mirror and the original half together
    full_image = np.concatenate([left, half_image], axis=1)
    return full_image


def symmetrize_plasma_img(
    image: np.ndarray,
    n_rows: int
) -> np.ndarray:
    """
    Apply a 1D transform to the bottom `n_rows` of each row in `image`,
    zeroing out the rows above.

    Parameters
    ----------
    image : np.ndarray, shape (H, W)
        Input image.
    n_rows : int
        Number of rows (from the bottom) to process.
    Returns
    -------
    out : np.ndarray, shape (H, W)
        Output image where:
          - rows H-n_rows ... H-1 are replaced by symmetrize_func(row)
          - rows   0     ... H-n_rows-1 are all zeros
    """
    H, W = image.shape
    if not (0 <= n_rows <= H):
        raise ValueError(f"n_rows must be between 0 and {H}, got {n_rows}")

    out = np.zeros_like(image, dtype=float)

    # Process bottom n_rows
    for i in range(n_rows):
        row_idx = H - 1 - i
        profile = image[row_idx, :].astype(float)
        transformed = symmetrize_via_fft(profile)
        if transformed.shape != (W,):
            raise ValueError(
                f"symmetrize_func must return length {W}, "
                f"but got {transformed.shape}"
            )
        out[row_idx, :] = transformed

    return out


# -------- Automated Abel inversion -------

def inverse_abel(
    profile: np.ndarray,
    method: str,
    dr: float = 1.0,
    # BASEX parameters
    sigma: float = 2.0,
    reg: float = 0.0,
    correction: bool = True,
    # Hansen–Law parameters
    hold_order: int = 0
) -> np.ndarray:
    """
    Compute the inverse Abel transform of a 1D or 2D half-profile using either
    the BASEX or Hansen–Law algorithm.

    Parameters
    ----------
    profile : np.ndarray
        1D or 2D array (half-profile) from axis → edge.
    method : str
        'basex'       → use abel.basex.basex_transform
        'hansenlaw'   → use abel.hansenlaw.hansenlaw_transform
    dr : float
        Pixel spacing (for spatial scaling).
    sigma : float
        BASEX Gaussian basis width.
    reg : float
        BASEX Tikhonov regularization strength.
    correction : bool
        Whether BASEX applies the post‐processing correction.
    hold_order : int
        Hansen–Law zero‐order (0) or first‐order (1) hold interpolation.

    Returns
    -------
    recon : np.ndarray
        The reconstructed 1D array f(r) of the same length as `profile`.
    """
    method = method.lower()
    if method == 'basex':
        recon = basex_transform(
            profile,
            sigma=sigma,
            reg=reg,
            correction=correction,
            dr=dr,
            direction='inverse'
        )
    elif method == 'hansenlaw':
        recon = hansenlaw_transform(
            profile,
            dr=dr,
            direction='inverse',
            hold_order=hold_order
        )
    else:
        raise ValueError(f"Unknown method '{method}'. Choose 'basex' or 'hansenlaw'.")
    # return
    if profile.ndim == 2:
        return recon
    else:
        return np.asarray(recon, dtype=float)


