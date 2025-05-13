# -------------- constants for main script ----------------------------
path_confing_files = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\image_processing_param_json'
# 1) exclusion matrix
p = 15  # value for padding an image before fourier transform
# 2) Fourier transform:
# exclude surroundings of 0th diffraction order spectra -> should be less than the number of fringes:
exclude_radius_peaks = 8
smooth_spectra_sigma = 2  # gaussian blur spectra for better findings
treshold_phase_ratio = 1.6
# mask param (do not touch)
a = 100  # x-axis mask
# b = spectr_peak_distance  # y-axis mask
inner_radius_mask = 3  # mask out middle
# 3) Abel inversion: a) mask_radius b) symmetrize image (gauss sigma) c1) radius2 c2) Otsus method param
dr = 1  # step = px
reg = 10  # precission
sigma_abel = 3  # idk - don't touch
# 4) Masking electron density profile
crop_edge_artefacts = 10  # crop artefacts on the mask radius first