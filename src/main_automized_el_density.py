"""
Main code to run the Abel inversion.
"""

# ---- IMPORTS from my functions ----
# add path to make sure the imports happen:
import sys
import os

project_root = os.path.abspath(
    os.path.join(os.getcwd(), '..', '..'))  # Go up two levels from current path (from src/ to project root)
sys.path.append(project_root)
# or: sys.append(r"C:\Users\User\z\Desktop\WUT\Diplomka\ZPRACOVÁNÍ\Data testing\processing_project")
import global_utils
import global_constants
import constants
import global_functions_im
import utils
import functions_image_crop
import functions_phase_shift
import functions_abel
import functions_electron_density_analysis

# ---- other IMPORTS ----
import matplotlib.pyplot as plt
import tifffile as tiff  # Loads .tiff as a NumPy array
import numpy as np  # low level image manipulation (matrix)
import cmasher as cmr  # extra colormaps
import json

# ------ DATA LOAD HERE -> each time folder evaluated independently --------
# load:
# 0a) load reference image
# 0b) Crop constant based on magnification -> x4 = -40px, x1 and x2 = -20px
# 1) exclusion matrix
# 2) Fourier transform: a) 'plus1' or 'minus1' b) y_peak_shift + value c) mask_sigma
# 3) base crop
# 4) Abel inversion: a) mask_radius b) symmetrize image (gauss sigma) c1) radius2 c2) Otsus method param
# 5) Masking electron density profile

# -------------- LOAD configurations ----------------------------
# here I save images, .json etc...:
output_dir = r'C:\Users\User\z\Desktop\WUT\Diplomka\ZPRACOVÁNÍ\Data testing\processing_project\Abel-inversion\output\el_density_files_v2'

path_json = r'C:\Users\User\z\Desktop\WUT\Diplomka\DATA\Images\image processing param json\2090nm\(4) 2090nm_Cu_6-15us.json'
with open(path_json) as f:
    data = json.load(f)

# Main variables
px_crop = data['px_crop']
peak_name =  data['peak_name']
mask_sigma = data['mask_sigma']
y_peak_shift = data['y_peak_shift']
shift_val = data['shift_val']
base_crop = data['base_crop']
mask_radius = data['mask_radius']
bool_remove_tilt = data['bool_remove_tilt']
bool_tilt_masked = data['bool_tilt_masked']
bool_normalise_phase = data['bool_normalise_phase']
symmetrize_gauss = data['symmetrize_gauss']
r_electron_density = data['r_electron_density']
bool_nonzero_mask = data['bool_nonzero_mask']
missing_bottom = data['missing_bottom']
usable_data = data["usable"]

# Otsu parameters
otsu = data['otsu']
method = otsu['method']
n_classes = otsu['n_classes']
threshold_index = otsu['threshold_index']
remove_small = otsu['remove_small']
close_size = otsu['close_size']
otsu_sigma = otsu['otsu_sigma']
keep_largest = otsu['keep_largest']

"""
# 0)
px_crop = 40  # crop data by additional value
# 2)
peak_name = 'plus1'  # or 'plus1' -> above center, or 'minus1' -> below center
mask_sigma = 5  # smooth out filtration mask
y_peak_shift = True  # 1064 nm = True, 2090 nm = False
shift_val = 20
# 3)
base_crop = 15
# 4)
mask_radius = 185  # input into Abel inversion should have clearly defined edge + helps with symmetrization
symmetrize_gauss = 5  # smooth symmetrisation (input should be smooth) -> might help with H0 and H6 for 1064 nm
# 5)
# if r_electron_density != 0 -> will crop with circular mask, else otsus method
r_electron_density = 0  # 62
# Otsu method parameters
missing_bottom = True  # fills in rectangle mask if bottom is missing
method = 'multiotsu'
n_classes = 3  # must be integer and greater than 3! (how many classes the histogram will be divided into)
remove_small = 5  # remove small regions from the mask
close_size = 15  # use a 10×10 closing structuring element
otsu_sigma = 3   # mask gaussian sigma
keep_largest = True
"""
# -------------- constants load ----------------------------
p = constants.p
exclude_radius_peaks = constants.exclude_radius_peaks
smooth_spectra_sigma = constants.smooth_spectra_sigma
treshold_phase_ratio = constants.treshold_phase_ratio
# mask param (do not touch)
a = constants.a  # x-axis mask
# b = spectr_peak_distance  # y-axis mask
inner_radius_mask = constants.inner_radius_mask
# x0 = cols // 2 # middle of an image -> if needed change here
dr = constants.dr
reg = constants.reg
sigma_abel = constants.sigma_abel
crop_edge_artefacts = constants.crop_edge_artefacts
# -----------------------------------------------------

"""
# ----------------------------------------------------------------
# path_ref_files = os.path.join(path4, measurement_folders[9])
# for meas in path_ref_files: # for each reference file:
#    ref_path = os.path.join(path_ref_files, files_ref[i]) # path to reference image

files_ref = global_utils.get_file_names(path_ref_files)
path_ref_files = os.path.join(path4, measurement_folders[9])
ref_path = os.path.join(path_ref_files, files_ref[19]) # path to reference image
ref = tiff.imread(ref_path)

path_img_files = os.path.join(path4, measurement_folders[0])
files_img = global_utils.get_file_names(path_img_files) # files to iterate over

# number of rows is total files divided by 10
n_rows = len(files_img) // 10

# Loop over X positions 1…10
for col, exact in enumerate(range(1, 11)):
    files_x = global_utils.filter_by_X_exact(files_img, exact=exact)
    # Loop over n positions 1…n # number of M
    for i, fname in enumerate(files_x[:n_rows]):
        img = tiff.imread(os.path.join(path_img_files, fname))
        # compare to ref_img instead of absolute metric

# ----------------------------------------------------------------
"""
# ------------- DATA PATH --------------
# this stays constant:
path = global_constants.base_dir
data_folders = global_utils.get_folder_names(path)  # ['IDEA program+ cvicna data', 'Images', 'Spectra']
path2 = os.path.join(path, data_folders[1])
# this will change:
wavelength_folders = global_utils.get_folder_names(path2)

path3 = os.path.join(path2, wavelength_folders[1])  # [0 = '1064 nm', 1 = '2090 nm']
sample_folders = global_utils.get_folder_names(path3)
# 0) [0 = 'H0_3_28_25', 1 = 'H6_3_31_25']
# 1) ['Cu', 'H0', 'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'wrong']
path4 = os.path.join(path3, sample_folders[0])
measurement_folders = global_utils.get_folder_names(path4)
print(data_folders)
print(wavelength_folders)
print(sample_folders)
print(measurement_folders)
# -------------------------------------
if __name__ == "__main__":
    # average elecron density:
    avg_el_density = []
    row_el_density = []
    # reference
    # ['(1) 50-1000ns', '(2) 1000 - 2000 ns', '(3) 2-5us', '(4) 5-14.5us', '(5) 15-100us', '(6) 100-525us', '(7) 500us-2ms', 'reference x1', 'reference x2', 'reference x4', 'wrong beginning']
    # (8) = x2, (9) = x4 -> 1064 H0, (7) = 2x H6
    # 2090: (6) x4 (9) x2
    path_ref_files = os.path.join(path4, measurement_folders[9])
    files_ref = global_utils.get_file_names(path_ref_files)
    print(files_ref)
    ref_path = os.path.join(path_ref_files, files_ref[0])  # path to reference image
    ref = tiff.imread(ref_path)
    # image files
    # ['(1) 50-1000ns', '(2) 1000 - 2000 ns', '(3) 2-5us', '(4) 6-15us', '(5) 15-100us', '(6) 100-525us', '(7) 500us-2ms', 'reference x1', 'reference x2', 'reference x4', 'wrong beginning']
    path_img_files = os.path.join(path4, measurement_folders[3]) #
    files_img = global_utils.get_file_names(path_img_files)  # files to iterate over
    print(path_img_files)
    print(ref_path)
    # number of rows is total files divided by 10
    n_rows = len(files_img) // 10
    print(f'n_rows: {n_rows}')
    # Loop over X positions 1…10
    for col in range(n_rows):
        print('-------')
        files_x = global_utils.filter_by_X_exact(files_img, exact=col+1)
        print(files_x)
        if col != 0:
            avg_el_density.append(row_el_density)
        row_el_density = []
        # Loop over n positions 1…n # number of M
        for i in range(len(files_x)):
            if not usable_data[col][i]:
                row_el_density.append(0)
                continue
            print(files_x[i])
            img = tiff.imread(os.path.join(path_img_files, files_x[i]))

            # ---- CROP DATA ----
            # detect edges
            sx, wid, hei = functions_image_crop.compute_crop_params(img, px_crop)
            crop_img = functions_image_crop.crop_from_center(img, sx, wid, hei)
            crop_ref = functions_image_crop.crop_from_center(ref, sx, wid, hei)
            # global_functions_im.plot_two_images(crop_img, crop_ref, title1='crop Image', title2='crop Reference')
            # ---- PADDING IMAGE ----
            padded_img = np.pad(crop_img, pad_width=p, mode='constant')  # constant, wrap, symmetric
            padded_ref = np.pad(crop_ref, pad_width=p, mode='constant')  # constant, wrap, symmetric

            # ---- FOURIER TRANSFORM ----
            # isolate the -1st-order peak and get its phase map
            spectr_peaks = functions_phase_shift.find_fourier_peaks(padded_img, exclude_radius=exclude_radius_peaks,
                                                                    smooth_sigma=smooth_spectra_sigma)
            # extract the first coordinate of ‘center’ and ‘plus1’
            if peak_name == 'plus1':
                y_peak = int(spectr_peaks['plus1'][0])
                x_peak = int(spectr_peaks['plus1'][1])
            elif peak_name == 'minus1':
                y_peak = int(spectr_peaks['minus1'][0])
                x_peak = int(spectr_peaks['minus1'][1])
            else:
                raise ValueError(f"Unknown peak_name: {peak_name!r}")
            y_peak_c = int(spectr_peaks['center'][0])
            x_peak_c = int(spectr_peaks['center'][1])

            spectr_peak_distance = abs(y_peak_c - y_peak) - 1  # distance from center = b value of the mask
            b = spectr_peak_distance  # second (elliptical mask) axis

            # shift mask from the diffraction order (optional)
            if y_peak_shift:
                if peak_name == 'plus1':
                    shift_val = -shift_val  # shift mask up (-)#num
                elif peak_name == 'minus1':
                    shift_val = shift_val  # down (+)#num
                y_peak = y_peak + shift_val

            phase_map_im = functions_phase_shift.compute_phase_from_padded(
                img_padded=padded_img,
                order=peak_name,
                exclude_radius=exclude_radius_peaks,  # to find peaks outside
                smooth_sigma=smooth_spectra_sigma,  # smooth spectrum to find peaks better
                a_div=a,
                b_div=b,
                mask_sigma=mask_sigma,  # gauss filter mask
                pad=p  # padding crop after computation
            )
            phase_map_ref = functions_phase_shift.compute_phase_from_padded(
                img_padded=padded_ref,
                order=peak_name,
                exclude_radius=exclude_radius_peaks,
                smooth_sigma=smooth_spectra_sigma,
                a_div=a,
                b_div=b,
                mask_sigma=mask_sigma,
                pad=p
            )

            # ---- PHASE SHIFT and PHASE UNWRAP ----
            phase_shift = phase_map_ref - phase_map_im
            phase_shift = np.mod(phase_shift, 2 * np.pi)  # modulo 2pi
            phase_shift_crop = functions_phase_shift.crop_img_base(phase_shift, base_crop)  # crop base of the image
            phase_shift_crop = utils.crop_lrt(phase_shift_crop, n=10)  # crops image left, right, top by 10 px to get rid of edge artefacts
            image_unwrapped = functions_phase_shift.unwrap_phase_image(phase_shift_crop)


            # ---- PRE-PROCESS INVERSE ABEL TRANSFORM (1)----
            # y0 at the last row (base of the image), x0 at the horizontal midpoint
            rows, cols = image_unwrapped.shape
            y0 = rows - 1  # bottom row of an image
            x0 = cols // 2  # middle column of an image

            semicircle_mask = functions_abel.make_semicircle_mask((rows, cols),
                                                                  center=(y0, x0),
                                                                  radius=mask_radius,
                                                                  smooth_sigma=None)
            if bool_remove_tilt:
                # removing tilt in background
                if bool_tilt_masked:
                    mask_flipped = 1 - semicircle_mask.astype(bool)  # I need the outside
                    bg, coeffs = functions_phase_shift.fit_plane_background(image_unwrapped, mask=mask_flipped)
                else:
                    bg, coeffs = functions_phase_shift.fit_plane_background(image_unwrapped)

                image_unwrapped = image_unwrapped - bg  # remove background

            if bool_normalise_phase:  # normalise to 0
                image_unwrapped = functions_abel.normalize_image(image_unwrapped, force_zero=True)  # normalise


            semicircle_mask2 = functions_abel.make_semicircle_mask((rows, cols),
                                                                  center=(y0, x0),
                                                                  radius=mask_radius//2,
                                                                  smooth_sigma=None)

            """
            # check if plasma is positive, else calculate again
            invert = functions_electron_density_analysis.masked_average(image_unwrapped, semicircle_mask2)
            max_val = np.max(image_unwrapped)
            min_val = np.min(image_unwrapped)
            threshold_phase = min_val + (max_val - min_val) / treshold_phase_ratio  # treshold 1/3 above minimum
            if invert < threshold_phase:
                if peak_name == 'plus1':
                    peak_name = 'minus1'
                else:
                    peak_name = 'plus1'
                # shift mask from the diffraction order (optional)
                if y_peak_shift:
                    if peak_name == 'plus1':
                        shift_val = -shift_val  # shift mask up (-)#num
                    elif peak_name == 'minus1':
                        shift_val = shift_val  # down (+)#num
                    y_peak = y_peak + shift_val

                phase_map_im = functions_phase_shift.compute_phase_from_padded(
                    img_padded=padded_img,
                    order=peak_name,
                    exclude_radius=exclude_radius_peaks,  # to find peaks outside
                    smooth_sigma=smooth_spectra_sigma,  # smooth spectrum to find peaks better
                    a_div=a,
                    b_div=b,
                    mask_sigma=mask_sigma,  # gauss filter mask
                    pad=p  # padding crop after computation
                )
                phase_map_ref = functions_phase_shift.compute_phase_from_padded(
                    img_padded=padded_ref,
                    order=peak_name,
                    exclude_radius=exclude_radius_peaks,
                    smooth_sigma=smooth_spectra_sigma,
                    a_div=a,
                    b_div=b,
                    mask_sigma=mask_sigma,
                    pad=p
                )

                # ---- PHASE SHIFT and PHASE UNWRAP ----
                phase_shift = phase_map_ref - phase_map_im
                phase_shift = np.mod(phase_shift, 2 * np.pi)  # modulo 2pi
                phase_shift_crop = functions_phase_shift.crop_img_base(phase_shift, base_crop)  # crop base of the image
                phase_shift_crop = utils.crop_lrt(phase_shift_crop, n=10)  # crops image left, right, top by 10 px to get rid of edge artefacts
                image_unwrapped = functions_phase_shift.unwrap_phase_image(phase_shift_crop)
                # removing tilt in background
                if bool_remove_tilt:
                    if bool_tilt_masked:
                        mask_flipped = 1 - semicircle_mask.astype(bool)  # I need the outside
                        bg, coeffs = functions_phase_shift.fit_plane_background(image_unwrapped, mask=mask_flipped)
                    else:
                        bg, coeffs = functions_phase_shift.fit_plane_background(image_unwrapped)
                    image_unwrapped = image_unwrapped - bg  # remove background

                if bool_normalise_phase:  # normalise to 0
                    image_unwrapped = functions_abel.normalize_image(image_unwrapped, force_zero=True)  # normalise
                """

            # mask plasma:
            masked_plasma = image_unwrapped * semicircle_mask
            # ---- PRE-PROCESS ABEL TRANSFORM (2)----
            symmetrize_plasma = functions_abel.symmetrize_plasma_img(masked_plasma, n_rows=mask_radius, gaussian_sigma=symmetrize_gauss,
                                                                     smooth_per_row=True)
            row_center = symmetrize_plasma.shape[1] // 2  # Half of the image size
            # right side so that half[:,0] is on the axis, row_center = index column of symmetry
            # TODO ---- ZDE MOHU VYČÍSLIT SYMETRIČNOST! -> masked_plasma[:, row_center:]; masked_plasma[:, :row_center] ----
            half_img = symmetrize_plasma[:, row_center:]

            # ---- INVERSE ABEL TRANSFORM = INDEX OF REFRACTION ----
            # Now run the inverse‐Abel on that half-profile:
            inv_abel = functions_abel.inverse_abel(half_img, method='basex', sigma=sigma_abel,
                                                    reg=reg, dr=dr)
            # Mirror data to get full 2D profile:
            inv_abel_full = functions_abel.mirror_image(inv_abel)

            # ---- INDEX OF REFRACTION -> ELECTRON DENSITY ----
            # if negative values, shift them so there is only positive vals (min = 0):
            # inv_abel_shifted = functions_abel.normalize_negative_image(inv_abel_full)
            electron_density = functions_abel.compute_electron_density(inv_abel_full)

            # ---- ELECTRON DENSITY MASKING ----
            rows, cols = electron_density.shape
            y0 = rows - 1
            x0 = cols // 2

            if r_electron_density != 0:
                mask = functions_abel.make_semicircle_mask((rows, cols),
                                                                        center=(y0, x0),
                                                                        radius=r_electron_density,
                                                                        smooth_sigma=None)
                electron_density_masked = electron_density * mask
            elif bool_nonzero_mask:
                # crop edge artefacts
                r_no = mask_radius - crop_edge_artefacts
                semicircle_mask = functions_abel.make_semicircle_mask((rows, cols),
                                                                        center=(y0, x0),
                                                                        radius=r_no,
                                                                        smooth_sigma=None)
                electron_density_pre_masked = electron_density * semicircle_mask
                mask = functions_electron_density_analysis.extract_main_positive_region(electron_density_pre_masked)
                electron_density_masked = electron_density_pre_masked * mask
            else:
                # crop edge artefacts
                r_no = mask_radius - crop_edge_artefacts
                semicircle_mask = functions_abel.make_semicircle_mask((rows, cols),
                                                                        center=(y0, x0),
                                                                        radius=r_no,
                                                                        smooth_sigma=None)
                electron_density_pre_masked = electron_density * semicircle_mask
                # Otsus tresholding
                mask, threshold = functions_electron_density_analysis.otsu_binary_mask(
                    electron_density_pre_masked,
                    method='multiotsu',
                    n_classes=n_classes,
                    threshold_index=threshold_index,
                    remove_small=remove_small,
                    close_size=close_size,  # use a 10×10 closing structuring element
                    gaussian_sigma=otsu_sigma,
                    keep_largest=keep_largest
                )
                if missing_bottom:  # adds rectangle mask to the bottom if missing
                    mask = functions_electron_density_analysis.fill_bottom_rectangle(mask)
                electron_density_masked = electron_density_pre_masked * mask

            # calculate average electron density
            integrated_ne = functions_electron_density_analysis.masked_average(electron_density_masked, mask)
            row_el_density.append(integrated_ne)
            print(integrated_ne)

#            fig, im = global_functions_im.plot_image(image_unwrapped, title='Unwrapped phase',
#                                           colorbar_label='$\Delta\Phi$', cmap=cmr.cosmic)
#            fname = f"unwrapped_phase({files_x[i]}).png"
#            out_path = os.path.join(output_dir, fname)
#            global_utils.save_plot_as_png(out_path, fig)
#            plt.close(fig)

#            fig, im = global_functions_im.plot_image(electron_density, title='Electron density',
#                                           colorbar_label=r'$n_e\ \mathrm{(cm^{-3})}$', cmap='inferno')
#            fname = f"electron_density({files_x[i]}).png"
#            out_path = os.path.join(output_dir, fname)
#            global_utils.save_plot_as_png(out_path, fig)
#            plt.close(fig)

#            fig, im = global_functions_im.plot_image(electron_density_masked, title='Output cropped', colorbar_label=r'$n_e\ \mathrm{(cm^{-3})}$', cmap='inferno')
#            fname = f"electron_density_cropped({files_x[i]}).png"
#            out_path = os.path.join(output_dir, fname)
#            global_utils.save_plot_as_png(out_path, fig)
#            plt.close(fig)

    # now append last column’s data
#    avg_el_density.append(row_el_density)
#    with open(os.path.join(output_dir, "El_density_Cu_2090_6-15us.json"), "w") as f:
#        json.dump(avg_el_density, f)

# crop optional:
# plasma_shape_mask_cropped = utils.crop_three_sides(electron_density_masked, r=mask_radius)



# ---- plots ----
# fig = global_functions_im.plot_two_images(img, ref, title1='raw Image', title2='raw Reference')
# global_functions_im.plot_two_images(crop_img, crop_ref, title1='crop Image', title2='crop Reference')
# utils.plot_fft_spectra(padded_img, cmap_mag = 'gray')
"""
maskino_spectrino = functions_phase_shift.create_ellipse_mask(
        shape=(rows, cols),
        x_center=x_peak,
        y_center=y_peak,
        a= a,
        b= b,
        sigma=mask_sigma
    )
maskino_spectrino_center = functions_phase_shift.circular_exclusion_mask(
    shape=(rows, cols),
    center=(y_peak_c,x_peak_c),
    radius=3
)

# inferno_cmap = plt.get_cmap('gray')  # this is the Colormap instance
# global_functions_im.plot_image(maskino_spectrino_center, cmap=cmr.cosmic)
fig = utils.plot_filtered_fft_spectrum(img_padded=padded_img, mask=maskino_spectrino, extra_mask=maskino_spectrino_center)
"""
# fig = global_functions_im.plot_image(phase_map_im, title='Phase image', cmap=cmr.arctic, colorbar_label='Phase') # arctic
# fig = global_functions_im.plot_image(phase_map_ref, title='Phase reference', cmap=cmr.arctic, colorbar_label='Phase') # arctic
