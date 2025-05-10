import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Optional


def plot_1d_profile(
    profile,
    title: str = "1D Profile",
    xlabel: str = "Index",
    ylabel: str = "Value",
    figsize: tuple[float, float] = (10, 4),
    grid: bool = True
):
    """
    Plot a 1D profile with customizable labels.

    Parameters
    ----------
    profile : array-like
        The 1D data to plot.
    title : str
        The title of the plot.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    figsize : tuple of two floats
        Width and height of the figure in inches.
    grid : bool
        Whether to show a grid.
    """
    plt.figure(figsize=figsize)
    plt.plot(profile, lw=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if grid:
        plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()

def plot_filtered_fft_spectrum(
    img_padded: np.ndarray,
    mask: np.ndarray,
    extra_mask: Optional[np.ndarray] = None,
    cmap_mag: str = 'gray'
) -> plt.Figure:
    """
    Plot the log-scaled magnitude of a masked FFT spectrum, optionally
    combining two masks, and return the Figure.

    Parameters
    ----------
    img_padded : 2D ndarray
        Zero-padded image.
    mask : 2D ndarray
        Primary mask in the Fourier domain (same shape as img_padded).
    extra_mask : 2D ndarray, optional
        If provided, this mask (e.g. a central-exclusion mask) will
        be multiplied with `mask` before filtering.
    cmap_mag : str, optional
        Colormap to use for displaying the magnitude spectrum.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The Figure object containing the plot.
    """
    # 1. FFT and center
    f_shifted = np.fft.fftshift(np.fft.fft2(img_padded))

    # 2. Combine your masks (if you passed a second one)
    combined_mask = mask if extra_mask is None else mask * extra_mask

    # 3. Apply combined mask and compute log-magnitude
    f_masked = f_shifted * combined_mask
    magnitude = np.log1p(np.abs(f_masked))

    # 4. Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(magnitude, cmap=cmap_mag)
    ax.set_title("Filtered FFT Magnitude Spectrum")
    ax.set_xlabel("Frequency (x)")
    ax.set_ylabel("Frequency (y)")

    # 5. Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Log Magnitude", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.show()

    return fig

def plot_fft_spectra(
    image,
    log_scale=True,
    cmap_mag='inferno',
    cmap_phase='twilight',
    figsize=(13, 6)
):
    """
    Compute FFT magnitude and phase spectra of an image and display them side by side.

    Parameters
    ----------
    image : array_like
        Original image array.
    log_scale : bool, optional
        If True, apply log scaling to magnitude spectrum.
    cmap_mag : str, optional
        Colormap for magnitude spectrum.
    cmap_phase : str, optional
        Colormap for phase spectrum.
    figsize : tuple, optional
        Figure size for the plots.
    """

    # Compute FFT and shift zero frequency component to center
    fft_image = np.fft.fft2(image)
    fft_shifted = np.fft.fftshift(fft_image)

    # Compute magnitude spectrum
    magnitude = np.log1p(np.abs(fft_shifted)) if log_scale else np.abs(fft_shifted)

    # Compute phase spectrum
    phase = np.angle(fft_shifted)

    # Plot both spectra
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Magnitude Spectrum
    im_m = axes[0].imshow(magnitude, cmap=cmap_mag)
    axes[0].set_title("Magnitude Spectrum (log scaled)" if log_scale else "Magnitude Spectrum")
    axes[0].axis('off')
    divider_m = make_axes_locatable(axes[0])
    cax_m = divider_m.append_axes("right", size="5%", pad=0.05)
    cbar_m = fig.colorbar(im_m, cax=cax_m)
    cbar_m.set_label("|A|", rotation=90, labelpad=15)

    # Phase Spectrum
    im_p = axes[1].imshow(phase, cmap=cmap_phase)
    axes[1].set_title("Phase Spectrum")
    axes[1].axis('off')
    divider_p = make_axes_locatable(axes[1])
    cax_p = divider_p.append_axes("right", size="5%", pad=0.05)
    cbar_p = fig.colorbar(im_p, cax=cax_p)
    cbar_p.set_label("Phase $\Phi$", rotation=90, labelpad=15)

    plt.tight_layout()
    plt.show()

def plot_image_row(img: np.ndarray, row: int, title: str = None) -> None:
    """
    Plot the intensity profile of a single row of a 2D image.

    Parameters
    ----------
    img : 2D ndarray
        Input image (rows = height, cols = width).
    row : int
        Zero-based index of the row to plot (vertical coordinate).
    title : str, optional
        Plot title.
    """
    profile = img[row, :]            # grab the row (y = row, all x)
    plt.figure(figsize=(10, 4))
    plt.plot(profile)
    if title:
        plt.title(title)
    plt.xlabel("Column index (x)")
    plt.ylabel("Phase shift $\Delta\Phi$")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def crop_three_sides(img: np.ndarray, r: int) -> np.ndarray:
    """
    Crop an image from the left, right, and top, keeping the bottom intact.

    Parameters
    ----------
    img : np.ndarray
        Input image, shape (H, W[, ...]).
    r : int
        Radius to keep around the center column and above the bottom:
          • width of output = 2*r + 1  (center_col - r  … center_col + r)
          • height of output = 2*r + 1 (bottom_row - 2*r  … bottom_row)

    Returns
    -------
    cropped : np.ndarray
        The cropped image of shape (2*r+1, 2*r+1[, ...]).
    """
    H, W = img.shape[:2]
    if not isinstance(r, int) or r < 0:
        raise ValueError("r must be a non-negative integer")

    center = W // 2
    left   = max(0, center - r)
    right  = min(W, center + r + 1)

    top    = max(0, H - (2*r + 1))
    bottom = H

    # slice rows then cols; preserve any extra channels
    cropped = img[top:bottom, left:right, ...]
    return cropped


