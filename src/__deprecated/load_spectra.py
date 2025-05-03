import pandas as pd
import matplotlib.pyplot as plt

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
    title_fontsize=14
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

    # 5) X‑axis limits
    ax.set_xlim(x.min(), x.max())

    fig.tight_layout()
    plt.show()
    return fig

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



# Example usage:
df = load_spectral_data(r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Spectra\H0_2025_3_28\A_50-1000ns\M1_50-1000ns_H0.asc')
# print(df.shape)      # e.g. (24816, 21)
# print(df.shape[1])   # numbe rof items in one row
#print(df.columns)    # wavelength, intensity(1) … intensity(20)
# print(df.head())

# plot_intensity(df, 2)  # plots the first intensity measurement vs. wavelength
plot_intensity(
    normalize_intensities(df),
    2,
    color='darkblue',
    fill_color=None,
    fill_alpha=0,
    linewidth=0.2,
    style='fivethirtyeight', #fivethirtyeight
    font_family='serif'
)

# import functions_spectra
import utils
if __name__ == "__main__":
    # ─── GIF settings ────────────────────────────────────────────────
    gif_folder = r"C:\Users\User\z\Desktop\WUT\Diplomka\PREZENTACE\image process\gif"
    gif_name = "gif_H0_spectr_try.gif"
    fps = 0.007
    loop = 0 # infinite loop/one pass
    duration = 1.0 / fps
    # writer = imageio.get_writer(os.path.join(gif_folder, gif_name), mode='I', duration=duration, loop=loop)
    # ─── CONFIG ────────────────────────────────────────────────
    root = r"C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Spectra\H0_2025_3_28"
    # ─── LOAD ──────────────────────────────────────────────────
    folders = utils.get_folder_names(root)
    no_meas_batch = 4 # number of measurement in the batch
    count = 0
    for i in folders:
        if count > 7: # number of measurements
            continue
        files = utils.get_file_names(root + "\\" + i)
        print(files)
        files_filtered = utils.filter_by_M_exact(files, no_meas_batch)
        print(files_filtered)
        sorted_files = sorted(files_filtered, key = utils.natural_key)
        print(sorted_files)
    #writer.close()