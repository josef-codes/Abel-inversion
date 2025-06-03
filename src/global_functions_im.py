# global_functions_im.py

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from skimage import io, exposure
from skimage import io as skio
# 
from PIL import ImageDraw, ImageFont, Image
from scipy.ndimage import gaussian_filter
import cmasher as cmr
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.image import AxesImage
from typing import Union, Tuple
from matplotlib.colors import Colormap


def stretch_contrast_np(arr, clip_percent=2):
    """Percentile stretch that works on *any* dtype, incl. float32."""
    lo, hi = np.percentile(arr, [clip_percent, 100 - clip_percent])
    out = (arr - lo) / (hi - lo)
    return np.clip(out, 0, 1)


def detect_bit_depth(arr: np.ndarray) -> int:
    """Return the bit depth (8, 16, etc.) of an integer image array."""
    if np.issubdtype(arr.dtype, np.integer):
        return arr.dtype.itemsize * 8
    raise ValueError(f"Unsupported dtype {arr.dtype}")


def gamma_correction(arr: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    """
    Apply gamma correction to an image array.

    • If `arr` is an integer type, it is first normalised to [0, 1]
      using the dtype’s max value, then re‑scaled back to the same dtype.
    • If `arr` is floating‑point, it is *assumed* to already be in
      [0, 1] (values outside will be clipped) and the result is returned
      in the same float dtype.

    Parameters
    ----------
    arr : np.ndarray
        Image array, uint8 / uint16 / float32 / float64 …
    gamma : float
        Gamma exponent ( <1 brightens, >1 darkens ).

    Returns
    -------
    np.ndarray
        Gamma‑corrected image, same dtype as input.
    """
    if np.issubdtype(arr.dtype, np.floating):
        # Float image: clip to [0,1], apply gamma, keep float dtype
        norm = np.clip(arr, 0.0, 1.0)
        corrected = np.power(norm, gamma, dtype=arr.dtype)
        return corrected.astype(arr.dtype)
    else:
        # Integer image: normalise, gamma, rescale to original integer range
        vmax = np.iinfo(arr.dtype).max  # 255 for uint8, 65535 for uint16, …
        norm = arr.astype(np.float32) / vmax
        corrected = np.power(norm, gamma, dtype=np.float32)
        return (corrected * vmax).round().astype(arr.dtype)


def annotate_image(
        img,
        text,
        position=(10, 10),
        font_path=None,
        font_size=24,
        color=255
):
    """
    Draws `text` onto a PIL Image `img` at `position`.

    Parameters
    ----------
    img : PIL.Image.Image
        The image to annotate (modified in place).
    text : str
        The text to draw.
    position : tuple(int, int)
        (x, y) pixel coordinates for the text.
    font_path : str or None
        Path to a .ttf font file. If None or load fails, uses default font.
    font_size : int
        Size of the font.
    color : int or tuple
        Grayscale value (0–255) or RGB tuple for text color.

    Returns
    -------
    PIL.Image.Image
        The annotated image.

        # 4) Add a text annotation
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=24)
    except IOError:
        font = ImageFont.load_default()
    text = text
    # Position text at (10, 10)
    draw.text((10, 10), text, fill=255, font=font)
    """

    draw = ImageDraw.Draw(img)
    # Load the specified font or fallback
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
    except IOError:
        font = ImageFont.load_default()
    # Draw the text
    draw.text(position, text, fill=color, font=font)
    return img


def edit_save_tiff(
        input_path: str,
        output_path: str,
        sgm: float = 1.0,
        clip_percent: float = 2.0,
        gamma_value: float = 1.2,
        cmap_name: str = 'nuclear'
):
    """
    Open a TIFF image, apply edits (filtering, contrast stretch, gamma correction,
    apply a cmasher colormap) and save to a new colored TIFF.

    Parameters
    ----------
    input_path : str
        Path to source TIFF.
    output_path : str
        Path to save processed TIFF.
    sgm : float
        Sigma for Gaussian smoothing.
    clip_percent : float
        Percentile clipping for contrast stretch (e.g., 2 -> clip 2% tails).
    gamma_value : float
        Gamma correction exponent.
    cmap_name : str
        Name of a cmasher colormap to apply (e.g., 'nuclear', 'cmr.ember').
    """
    # 1) Open the source image
    img = skio.imread(input_path).astype(np.float32)
    img /= np.iinfo(np.uint16).max  # 65535 → 1.0
    # 2) Edits
    img = gaussian_filter(img, sigma=sgm)
    img = stretch_contrast_np(img)
    # img = stretch_contrast(img, clip_percent=clip_percent)
    img = gamma_correction(img, gamma=gamma_value)
    # 3) get cmasher colormap
    cmap = getattr(cmr.cm, cmap_name)  # use cmasher’s colormap object
    # plt.figure()
    # plt.imshow(img, cmap=cmap)   # img is 2‑D, cmap supplied here
    # plt.axis("off")
    # plt.title("Coloured with nuclear colormap")
    # plt.show()
    # 4) colourise & convert to 8‑bit RGB
    rgb = (cmap(img)[..., :3] * 255).astype(np.uint8)
    # 4) Save as TIFF
    imageio.imwrite(output_path, rgb, format="png")
    print(f"Saved coloured PNG to:\n  {output_path}")


# --- REDUNDANT FUNCTIONS -------------------
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
    vmax = 2 ** b - 1

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


def plot_image(
        img: np.ndarray,
        title: str = '',
        cmap=cmr.nuclear,
        colorbar_label: str = 'Intensity',
        ignore_top_percent: float = 0.0,
        match_scale: Union[AxesImage, np.ndarray, Tuple[float, float], None] = None,
        fontsize: int = 15
) -> tuple[Figure, Axes, AxesImage]:
    """
    Display an image with a colorbar, custom title, and flexible color scaling.

    Parameters
    ----------
    img : 2D array
        Image data to display.
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap name.
    colorbar_label : str
        Label for the colorbar.
    ignore_top_percent : float, 0–100
        If >0, set vmax to the (100 - ignore_top_percent) percentile of img.
    match_scale : AxesImage or 2D array or (vmin, vmax)
        If AxesImage, use its color limits.
        If 2D array, use its min/max.
        If tuple, interpret as (vmin, vmax).
        If None, auto-scale (with optional ignore_top_percent).
    fontsize : int
        Font size for label and ticks.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    im : matplotlib.image.AxesImage
        The image object (useful for retrieving color limits later).
    """
    # Determine vmin/vmax
    if isinstance(match_scale, AxesImage):
        vmin, vmax = match_scale.get_clim()
    elif isinstance(match_scale, np.ndarray):
        vmin, vmax = float(match_scale.min()), float(match_scale.max())
    elif isinstance(match_scale, tuple) and len(match_scale) == 2:
        vmin, vmax = match_scale
    else:
        vmin = float(np.nanmin(img))
        if ignore_top_percent > 0:
            vmax = float(np.nanpercentile(img, 100.0 - ignore_top_percent))
        else:
            vmax = float(np.nanmax(img))

    # Plot
    fig, ax = plt.subplots()
    im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis('off')

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(colorbar_label, rotation=90, labelpad=10, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize - 2)

    plt.tight_layout()
    plt.show()

    return fig, im


def plot_two_images(
    img1: np.ndarray,
    img2: np.ndarray,
    title1: str = 'Image 1',
    title2: str = 'Image 2',
    cmap1: Union[str, Colormap] = 'cmr.nuclear',
    cmap2: Union[str, Colormap, None] = None,
    colorbar: bool = False,
    colorbar_label: str = '',
    share_color_scale: bool = True,
    vmin: float | None = None,
    vmax: float | None = None
) -> plt.Figure:
    """
    Plot two images side by side with flexible colorbar options and manual vmin/vmax.

    Parameters
    ----------
    img1, img2 : 2D np.ndarray
        The images to display.
    title1, title2 : str
        Subplot titles.
    cmap1 : str or Colormap
        Colormap for the first image.
    cmap2 : str, Colormap, or None
        Colormap for the second image. If None, uses cmap1.
    colorbar : bool
        If True, draw colorbar(s).
    colorbar_label : str
        Label for the colorbar(s). If provided, applied to all colorbars.
    share_color_scale : bool
        If True, uses a single color scale (vmin/vmax) for both images.
        If False, each image autoscaled or uses provided vmin/vmax.
    vmin, vmax : float or None
        Minimum and maximum data values for the colormap. If None, autoscale
        or shared scale logic applies.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # resolve colormaps
    def _resolve_cmap(c):
        return plt.get_cmap(c) if isinstance(c, str) else c

    cm1 = _resolve_cmap(cmap1)
    cm2 = _resolve_cmap(cmap2) if cmap2 is not None else cm1

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # determine color scale
    if colorbar and share_color_scale:
        if cm1.name != cm2.name:
            raise ValueError("Cannot share one color scale across different colormaps.")
        # compute global vmin/vmax if not provided
        global_vmin = vmin if vmin is not None else min(np.nanmin(img1), np.nanmin(img2))
        global_vmax = vmax if vmax is not None else max(np.nanmax(img1), np.nanmax(img2))
        im1 = axs[0].imshow(img1, cmap=cm1, vmin=global_vmin, vmax=global_vmax)
        im2 = axs[1].imshow(img2, cmap=cm2, vmin=global_vmin, vmax=global_vmax)
    else:
        # individual scaling
        im1 = axs[0].imshow(img1, cmap=cm1, vmin=vmin, vmax=vmax)
        im2 = axs[1].imshow(img2, cmap=cm2, vmin=vmin, vmax=vmax)

    # titles and remove axes ticks
    for ax, title in zip(axs, (title1, title2)):
        ax.set_title(title)
        ax.axis('off')

    # add colorbars if requested
    if colorbar:
        if share_color_scale:
            # one shared bar on the right of the second subplot
            divider = make_axes_locatable(axs[1])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(im2, cax=cax)
            if colorbar_label:
                cbar.set_label(colorbar_label, rotation=90, labelpad=10)
        else:
            # one per image
            for im, ax in zip((im1, im2), axs):
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = fig.colorbar(im, cax=cax)
                if colorbar_label:
                    cbar.set_label(colorbar_label, rotation=90, labelpad=10)

    plt.tight_layout()
    plt.show()
    return fig

def plot_two_images_v2(
    img1: np.ndarray,
    img2: np.ndarray,
    title1: str = 'Image 1',
    title2: str = 'Image 2',
    cmap1: Union[str, Colormap] = 'cmr.nuclear',
    cmap2: Union[str, Colormap, None] = None,
    colorbar: bool = False,
    colorbar_labels: Union[str, Tuple[str, str]] = ('', ''),
    share_color_scale: bool = True
) -> plt.Figure:
    """
    Plot two images side by side with flexible colorbar options.

    Parameters
    ----------
    img1, img2 : 2D np.ndarray
        Images to display.
    title1, title2 : str
        Subplot titles.
    cmap1 : str or Colormap
        Colormap for the first image.
    cmap2 : str, Colormap, or None
        Colormap for the second image. If None, uses cmap1.
    colorbar : bool
        If True, draw colorbar(s).
    colorbar_labels : str or tuple of str
        Label(s) for the colorbar(s). If a single string is provided,
        both colorbars use the same label. If a tuple, it must be (label1, label2).
    share_color_scale : bool
        If True, use the same vmin/vmax for both images; otherwise autoscale each.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # resolve colormaps
    def _resolve_cmap(c):
        return plt.get_cmap(c) if isinstance(c, str) else c

    cm1 = _resolve_cmap(cmap1)
    cm2 = _resolve_cmap(cmap2) if cmap2 is not None else cm1

    # prepare colorbar labels
    if isinstance(colorbar_labels, str):
        label1 = label2 = colorbar_labels
    else:
        label1, label2 = colorbar_labels

    # set up figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # determine scaling
    if share_color_scale:
        if cm1.name != cm2.name:
            raise ValueError("Cannot share one color scale across different colormaps.")
        vmin = min(np.nanmin(img1), np.nanmin(img2))
        vmax = max(np.nanmax(img1), np.nanmax(img2))
        im1 = axs[0].imshow(img1, cmap=cm1, vmin=vmin, vmax=vmax)
        im2 = axs[1].imshow(img2, cmap=cm2, vmin=vmin, vmax=vmax)
    else:
        im1 = axs[0].imshow(img1, cmap=cm1)
        im2 = axs[1].imshow(img2, cmap=cm2)

    # titles and remove axes ticks
    for ax, title in zip(axs, (title1, title2)):
        ax.set_title(title)
        ax.axis('off')

    # add colorbars if requested
    if colorbar:
        # first image colorbar
        divider1 = make_axes_locatable(axs[0])
        cax1 = divider1.append_axes("right", size="5%", pad=0.05)
        cbar1 = fig.colorbar(im1, cax=cax1)
        if label1:
            cbar1.set_label(label1, rotation=90, labelpad=10)

        # second image colorbar
        divider2 = make_axes_locatable(axs[1])
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cbar2 = fig.colorbar(im2, cax=cax2)
        if label2:
            cbar2.set_label(label2, rotation=90, labelpad=10)

    plt.tight_layout()
    plt.show()
    return fig


def save_tiff(data, filename):
    """
    Save a 2D array (list of lists or NumPy array) to a TIFF image.

    Parameters
    ----------
    data : array-like
        2D data to save (e.g. nested list or NumPy array).
    filename : str
        Path to the output .tiff file.

    Raises
    ------
    ValueError
        If `data` is not 2-dimensional.
    """
    # Convert to NumPy array
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D data, but got array with ndim={arr.ndim}")

    # Pillow expects integer types for most TIFFs;
    # auto-cast floats into 16-bit or 8-bit as needed
    if np.issubdtype(arr.dtype, np.floating):
        # scale floats to the full uint16 range
        arr = (arr - arr.min()) / (arr.max() - arr.min())  # normalize 0–1
        arr = (arr * 65535).astype(np.uint16)
    elif np.issubdtype(arr.dtype, np.integer):
        # ensure it’s a supported size
        if arr.dtype.itemsize > 2:
            # downcast large ints to uint16
            arr = arr.astype(np.uint16)
    else:
        # for any other type, just cast to uint8
        arr = arr.astype(np.uint8)

    # Create and save the image
    img = Image.fromarray(arr)
    img.save(filename, format='tiff')

# if __name__ == "__main__":
# inp = r"C:/Users/User/z/Desktop/WUT/Diplomka/DATA/Images/H6_3_31_25/(1) 50-1000ns/M1_X20.tif"
# inp = r"C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\H6_3_31_25\(2) 1000-2000ns\M10_X6.tif"
# inp = r"C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\H6_3_31_25\(3) 2-5us\M8_X8.tif"
# inp = r"C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\H6_3_31_25\(4) 5-15us\M4_X10.tif"
# inp = r"C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\H6_3_31_25\(5) 15-100us\M9_X7.tif"
# inp = r"C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\H6_3_31_25\(6 )100us-2ms\M10_X16.tif"
# inp = r"C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\H6_3_31_25\reference x4\REFx4_X12.tif"
# H0
# inp = r"C:/Users/User/z/Desktop/WUT/Diplomka/DATA/Images/H0_3_28_25/(1) 50-1000ns/M2_X2.tif"
# inp = r"C:/Users/User/z/Desktop/WUT/Diplomka/DATA/Images/H0_3_28_25/(1) 50-1000ns/M2_X20.tif"
# inp = r"C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\H0_3_28_25\(2) 1000 - 2000 ns\M6_X12.tif"
# inp = r"C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\H0_3_28_25\(3) 2-5us\M2_X16.tif"
# inp = r"C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\H0_3_28_25\(5) 15-100us\M1_X1.tif"
# inp = r"C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\H0_3_28_25\(5) 15-100us\M1_X18.tif"
# inp = r"C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\H0_3_28_25\(5) 15-100us\M3_X16.tif"

# inp = r"C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\2090 nm\H6\(5) 15-100us\M1_X8.tif"
# out = r"C:\Users\User\z\Desktop\WUT\Diplomka\RESULTS\Img - interference/95us_H6_2090.png"
# edit_save_tiff(inp, out)
