from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
import numpy as np  # low level image manipulation (matrix)


def get_horizontal_intensity_sum(image):
    """
    Makes sum in the horizontal direction of an image intensity.
    :param image: numpy
    """
    return image.sum(axis=0)


def smooth_profile(profile, sigma=2):
    return gaussian_filter1d(profile, sigma)


def calculate_1d_gradient(profile, sigma):
    smoothed = smooth_profile(profile, sigma)
    return np.gradient(smoothed)


def detect_edges(profile, smoothing_sigma=5, threshold_ratio=0.3):
    """
    Detect the left and right edges in a 1D intensity profile.

    Parameters
    profile : 1D array of summed intensities
    smoothing_sigma : float smoothing parameter for gaussian_filter1d
    threshold_ratio : float fraction of peak gradient to use as threshold

    Returns
    left_edge, right_edge : integer indices of the first rising edge
                            and the last falling edge
    """
    # smooth the profile to suppress noise
    smooth = gaussian_filter1d(profile, sigma=smoothing_sigma)

    # compute its gradient
    grad = np.gradient(smooth)

    # dynamic thresholds
    pos_thr = grad.max() * threshold_ratio
    neg_thr = grad.min() * threshold_ratio

    # first rising edge
    rising = np.where(grad > pos_thr)[0]
    left_edge = int(rising[0]) if rising.size else 0

    # last falling edge
    falling = np.where(grad < neg_thr)[0]
    right_edge = int(falling[-1]) if falling.size else profile.size - 1

    return left_edge, right_edge


def center_by_symmetry_1d(profile, smoothing_sigma=5):
    """
    Find the horizontal center of maximal left-right symmetry in a 1D profile.
    """
    # smooth the profile to suppress noise
    smooth = gaussian_filter1d(profile, sigma=smoothing_sigma)
    # full cross correlation with its own reversed copy
    corr = correlate(smooth, smooth[::-1], mode='full')
    # length of the original profile
    n = smooth.size
    # index of maximum correlation
    lag = np.argmax(corr)
    # convert lag to center coordinate
    # for profile length n the formula is (2n - 2 - lag) / 2
    center = int((2 * n - 2 - lag) / 2)
    return center


def crop_from_center(image, sx, width, height, sy=None):
    """
    Crop img around horizontal center Sx and vertical center Sy.
    If Sy is None the crop is taken from the top (y=0).
    """
    h, w = image.shape[:2]
    x0 = max(sx - width//2, 0)
    x1 = min(sx + width//2, w)
    if sy is None:
        y0, y1 = 0, min(height, h)
    else:
        y0 = max(sy - height//2, 0)
        y1 = min(sy + height//2, h)
    return image[y0:y1, x0:x1]