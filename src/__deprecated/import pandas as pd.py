import pandas as pd

def load_spectral_data_partial(file_path, skip_rows=0, num_rows=100):
    """
    Load a small portion of a pure‑numeric .asc spectral file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the .asc file.
    skip_rows : int
        Number of initial lines to skip before reading data.
    num_rows : int
        Number of data rows to read.
    
    Returns
    -------
    pd.DataFrame
        Columns named ['wavelength', 'intensity(1)', ...].
    """
    # Read only the specified block
    df = pd.read_csv(
        file_path,
        sep=';',
        header=None,
        skiprows=skip_rows,
        nrows=num_rows,
        quotechar='"',     # strips surrounding quotes
        engine='python'
    )
    # Drop any all-NaN columns (from trailing semicolons)
    df = df.dropna(axis=1, how='all')
    
    # Rename columns: first is wavelength, the rest are intensities
    ncols = df.shape[1]
    df.columns = ['wavelength'] + [f'intensity({i})' for i in range(1, ncols)]
    return df

# Example: load rows 0–99
file_path = r"C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Spectra\H0_2025_3_28\D_5-15us\M1_5-15us.asc"
df_small = load_spectral_data_partial(file_path, skip_rows=0, num_rows=20000)
print(df_small.head())
print(df_small.shape)