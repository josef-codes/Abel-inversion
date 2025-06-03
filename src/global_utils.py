# global_utils.py

import os
import re
from typing import List, Union
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional


def get_folder_names(root_folder):
    """
    Return a list of all directories in a folder.
    """
    return [
        name for name in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, name))
    ]


def get_file_names(root_folder):
    """
    Return a list of all files in a folder.
    """
    # return [
    #    name for name in os.listdir(folder)
    #    if os.path.isfile(os.path.join(folder, name))
    # ]
    return [
        name
        for name in os.listdir(root_folder)
        if os.path.isfile(os.path.join(root_folder, name))
    ]


def filter_by_M_exact(
    names: List[str],
    exact: Union[int, List[int]]
) -> List[str]:
    """
    From a list of filenames, return only those matching "M<number>_…"
    where <number> == exact (or is in the list exact), sorted by their
    numeric components.

    Parameters
    ----------
    names : List[str]
        Filenames to filter.
    exact : int or List[int]
        The value(s) that the M<number> must equal.

    Returns
    -------
    List[str]
        Filenames whose number matches exactly, sorted by (M, then subsequent numbers).
    """
    # match the M<number> prefix
    pat = re.compile(r"^M(\d+)_")
    targets = {exact} if isinstance(exact, int) else set(exact)

    # first filter
    filtered = []
    for nm in names:
        m = pat.match(nm)
        if not m:
            continue
        if int(m.group(1)) in targets:
            filtered.append(nm)

    # now sort by all numeric substrings in the filename
    # e.g. "M4_X10.tif" → [4, 10]
    def numeric_key(nm: str):
        nums = list(map(int, re.findall(r"\d+", nm)))
        return tuple(nums)

    return sorted(filtered, key=numeric_key)


def filter_by_X_exact(
    names: List[str],
    exact: Union[int, List[int]]
) -> List[str]:
    """
    From a list of filenames, return only those containing "_X<number>"
    where <number> == exact (or is in the list exact), sorted by the
    numeric value before and after "_X".

    Parameters
    ----------
    names : List[str]
        Filenames to filter.
    exact : int or List[int]
        The X<number> value(s) that must match.

    Returns
    -------
    List[str]
        Filenames whose "_X<number>" matches exactly, sorted naturally
        by the leading 'M' number.
    """
    pat = re.compile(r"^M(\d+)_X(\d+)\b")
    targets = {exact} if isinstance(exact, int) else set(exact)

    filtered = []
    for nm in names:
        m = pat.match(nm)
        if not m:
            continue
        x_val = int(m.group(2))
        if x_val in targets:
            filtered.append(nm)

    def numeric_key(nm: str):
        # extract M-number then X-number
        m = pat.match(nm)
        m_num = int(m.group(1))
        x_num = int(m.group(2))
        return (m_num, x_num)

    return sorted(filtered, key=numeric_key)


def natural_key(s: str):
    """
    Turn a string into a list of ints and lower‑case text,
    so that numeric parts sort by value, not lexically.
    E.g. "M3_X16.tif" → ["m", 3, "_x", 16, ".tif"]
    """
    parts = re.split(r'(\d+)', s)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def save_plot_as_png(
    filename: str,
    fig: Optional[Figure] = None,
    dpi: int = 300,
    transparent: bool = False,
    bbox_inches: str = 'tight'
) -> None:
    """
    Save a Matplotlib figure to a PNG file.

    Parameters
    ----------
    filename : str
        Path (including .png extension) where the image will be saved.
    fig : matplotlib.figure.Figure, optional
        The Figure object to save. If None, uses the current figure (plt.gcf()).
    dpi : int, optional
        Dots per inch for the output image (controls resolution).
    transparent : bool, optional
        If True, the background will be transparent.
    bbox_inches : str, optional
        Bounding box setting passed to savefig (e.g., 'tight', 'standard').
    """
    if fig is None:
        fig = plt.gcf()

    fig.savefig(
        fname=filename,
        dpi=dpi,
        transparent=transparent,
        bbox_inches=bbox_inches,
        format='png'
    )


# ── Usage ──
#root = r"C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\H0_3_28_25"
#batches = get_folder_names(root)
#print(batches)
#files = get_file_names(root + \\ + batches[0])
#print(files)

# ─── USAGE regex ────────────────────────────────────────────────────────────
#print(filter_by_M_exact(files, 1))
#print(filter_by_M_exact(files, [5,4]))

#print(filter_by_X_exact(files, [1]))
