# global_functions_spectra.py
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional



def load_spectral_data(file_path):
    """
    Load a pure‑numeric .asc file (delimiter=';') into a pandas DataFrame.
    Handles lines wrapped in double‑quotes and trailing semicolons.
    
    Returns:
        df (pd.DataFrame): columns = ['wavelength', 'intensity(1)', …, 'intensity(n)']
    """
    data = []
    with open(file_path, 'r') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            # Strip surrounding quotes if present
            if line.startswith('"') and line.endswith('"'):
                line = line[1:-1]
            # Remove any trailing semicolons
            line = line.rstrip(';')
            # Split on ';' and convert every field to float
            parts = line.split(';')
            floats = [float(cell) for cell in parts]
            data.append(floats)

    # Build DataFrame
    df = pd.DataFrame(data)
    # Name columns: first = wavelength, rest = intensity(1), intensity(2), …
    ncols = df.shape[1]
    df.columns = ['wavelength'] + [f'intensity({i})' for i in range(1, ncols)]
    return df

def extract_column(df, col):
    """
    Extract a column by index (int) or name (str).
    """
    if isinstance(col, int):
        return df.iloc[:, col]
    elif isinstance(col, str):
        return df[col]
    else:
        raise ValueError("Column must be an integer index or a column name string.")

def normalize_intensities(df):
    """
    Min–max normalize all intensity columns in a spectral DataFrame.
    
    Assumes df.columns = ['wavelength', 'intensity(1)', 'intensity(2)', …].
    Returns a new DataFrame with the same columns; wavelength is untouched.
    """
    df_norm = df.copy()
    # Skip the first column (wavelength)
    for col in df_norm.columns[1:]:
        y = df_norm[col]
        y_min, y_max = y.min(), y.max()
        # avoid divide-by-zero if flat
        if y_max > y_min:
            df_norm[col] = (y - y_min) / (y_max - y_min)
        else:
            df_norm[col] = 0.0
    return df_norm

def normalize_intensities_by_ref_max(df, ref_max):
    """
    Normalize all intensity columns by a provided reference maximum value.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input spectral DataFrame with columns ['wavelength', 'intensity(1)', …].
    ref_max : float
        The maximum intensity value (from whichever measurement) to normalize to 1.
    
    Returns
    -------
    pd.DataFrame
        A new DataFrame with the same structure where each intensity is divided by ref_max.
    """
    df_norm = df.copy()
    for col in df_norm.columns[1:]:
        df_norm[col] = df_norm[col] / ref_max
    return df_norm

def get_intensity_maximum(df):
    """
    Return the single highest intensity value across all intensity columns.
    """
    # compute per‑column maxima, then take the max of those
    return df.iloc[:, 1:].max().max()

def spectrum_normalised(df: pd.DataFrame, col: int = 1) -> np.ndarray:
    """
    Min–max-normalise a single intensity column from a spectral DataFrame
    and return it as a 1-D NumPy array.

    Parameters
    ----------
    df  : pandas.DataFrame
        First column must be wavelength; intensity columns follow.
    col : int, default 1
        Column index of the intensity spectrum to normalise.

    Returns
    -------
    np.ndarray
        1-D array of the normalised spectrum (values in [0, 1]).
    """
    # extract the chosen intensity column as a float array
    y = df.iloc[:, col].to_numpy(dtype=float)

    y_min = y.min()
    y_max = y.max()

    if y_max > y_min:                # normal case
        return (y - y_min) / (y_max - y_min)
    else:                            # flat spectrum → all zeros
        return np.zeros_like(y, dtype=float)

def plot_intensity(
    df,
    intensity_col_index,
    color='C0',
    fill_color=None,
    fill_alpha=0.3,
    linewidth=1.0,
    style='seaborn-darkgrid',
    font_family='sans-serif',
    font_size=12,
    title_fontsize=14,
    show_plot = 0
):
    """
    Plot wavelength vs intensity with customizable style and filled area under the curve.
    
    Parameters
    ----------
    df : pd.DataFrame
    intensity_col_index : int
        Column index of the intensity to plot (1..n)
    color : str
        Matplotlib color name or code for the line.
    fill_color : str or None
        Color for the fill area. If None, uses `color`.
    fill_alpha : float
        Opacity for the fill (0 transparent, 1 opaque).
    linewidth : float
        Thickness of the line.
    style : str
        Matplotlib style sheet name.
    font_family : str
        Font family for text.
    font_size : int
        Base font size for labels and ticks.
    title_fontsize : int
        Font size for the title.
    show_plot = 0 : bool 1 or 0
        Shows plot in figure
    """
    # 1) Style & fonts
    plt.style.use(style)
    plt.rcParams['font.family'] = font_family
    plt.rcParams['font.size'] = font_size

    # 2) Extract data
    x = df.iloc[:, 0]
    y = df.iloc[:, intensity_col_index]

    # 3) Plot
    fig, ax = plt.subplots()
    
    # Fill under curve
    fc = fill_color or color
    ax.fill_between(x, y, color=fc, alpha=fill_alpha)
    
    # Plot line on top
    ax.plot(x, y, color=color, linewidth=linewidth)

    # 4) Labels & title
    ax.set_xlabel("wavelength (nm)") # df.columns[0]
    ax.set_ylabel("Intensity (a.u.)") # df.columns[intensity_col_index]
    # ax.set_title(f"{df.columns[intensity_col_index]} vs {df.columns[0]}",fontsize=title_fontsize)

    # 5) axis limits
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(-0.05, 1)
    fig.tight_layout()

    if show_plot != 0:
        plt.show()

    return fig


def integrate_uniform(intensity: np.ndarray, delta: float) -> float:
    """
    Integrate a uniformly sampled 1-D spectrum by a simple Σ I · Δλ.

    Parameters
    ----------
    intensity : 1-D array-like
        Intensity values I₀, I₁, …, Iₙ sampled on a uniform grid.
    delta : float
        Wavelength step Δλ (nm) between consecutive samples.

    Returns
    -------
    float
        Integral  Σ I · Δλ  (units: intensity × nm).
    """
    intensity = np.asarray(intensity, dtype=float).ravel()
    return intensity.sum() * float(delta)


def integrate_peak_sum(
    target_wl: float,          # wavelength you care about   (nm)
    wl: np.ndarray,            # 1-D wavelength vector       (nm)
    I: np.ndarray,             # 1-D intensity vector        (same length)
    *,
    half_width: float = 1.0,   # ± window half-width         (nm)
    left: Optional[float] = None,
    right: Optional[float] = None
) -> float:
    """
    Simple Σ I·Δλ integration over a narrow band in a uniformly-sampled spectrum.

    Parameters
    ----------
    target_wl : float
        Centre of the peak (ignored if `left`/`right` are given).
    wl : ndarray
        Wavelength array (uniform grid assumed).
    I : ndarray
        Intensity array aligned with `wl`.
    half_width : float
        Half-width of the window when `left`/`right` are omitted.
    left, right : float, optional
        Explicit wavelength limits.  If either is None the window is
        [target_wl − half_width, target_wl + half_width].

    Returns
    -------
    float
        Σ I·Δλ  →  “intensity × nm”.
    """
    if wl.shape != I.shape:
        raise ValueError("wl and I must have the same length")

    # decide the window
    if left is None or right is None:
        left  = target_wl - half_width
        right = target_wl + half_width
    if left >= right:
        raise ValueError("left must be smaller than right")

    # boolean mask for points inside the window
    m = (wl >= left) & (wl <= right)
    if not m.any():
        raise RuntimeError("No wavelength points inside the chosen window")

    # uniform step size (assumed)
    delta = np.mean(np.diff(wl))
    # warn if grid is noticeably non-uniform
    if np.ptp(np.diff(wl)) > 1e-6 * delta:
        print("⚠️  Grid is not perfectly uniform; sum-rule may be approximate")

    # simple Σ I * Δλ
    return I[m].sum() * delta




