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


# ----------------------------------------------------------------

def save_images_to_json(phase_unwrap, electron_density, avg_el_dens, json_filename):
    """Saves two image matrices plus their average density into a JSON file."""
    # ensure numpy arrays
    pu = np.array(phase_unwrap)
    ed = np.array(electron_density)
    data = {
        "phase_unwrap": pu.tolist(),
        "electron_density": ed.tolist(),
        "avg_electron_density": float(avg_el_dens)
    }
    with open(json_filename, 'w') as f:
        json.dump(data, f, indent=4)


# -------------------------------------------------------------------

# -------------- LOAD configurations ----------------------------
# here I save images, .json etc...:
path_out_fold = r'C:\Users\User\z\Desktop\WUT\Diplomka\RESULTS\DATASET do prilohy\H0 1064 nm\el dens'

path_json = r'C:\Users\User\z\Desktop\WUT\Diplomka\RESULTS\DATASET do prilohy\H0 1064 nm\img\(4) 1064nm_H0_6-15.json'
with open(path_json) as f:
    data = json.load(f)

# Main variables
px_crop = data['px_crop']
peak_name = data['peak_name']  # "minus1", "plus1"
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
# ------------- DATA PATH --------------
# this stays constant:
path_img = r"C:\Users\User\z\Desktop\WUT\Diplomka\RESULTS\DATASET do prilohy\H0 1064 nm\img"
data_folders = global_utils.get_folder_names(path_img)  #
path_img_files = os.path.join(path_img, data_folders[3])  #
print(path_img_files)
files_img = global_utils.get_file_names(path_img_files)  # files to iterate over

ref = tiff.imread(
    r'C:\Users\User\z\Desktop\WUT\Diplomka\RESULTS\DATASET do prilohy\H0 1064 nm\img\REF2_X3_x2.tif')
# -------------------------------------
if __name__ == "__main__":
    # number of rows is total files divided by 10
    # Loop over X positions 1…10
    # Loop over n positions 1…n # number of M
    for i in range(len(files_img)):
        print(files_img[i])
        img = tiff.imread(os.path.join(path_img_files, files_img[i]))

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
        phase_shift_crop = utils.crop_lrt(phase_shift_crop,
                                          n=10)  # crops image left, right, top by 10 px to get rid of edge artefacts
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
                                                               radius=mask_radius // 2,
                                                               smooth_sigma=None)

        # mask plasma:
        masked_plasma = image_unwrapped * semicircle_mask
        # ---- PRE-PROCESS ABEL TRANSFORM (2)----
        symmetrize_plasma = functions_abel.symmetrize_plasma_img(masked_plasma, n_rows=mask_radius,
                                                                 gaussian_sigma=symmetrize_gauss,
                                                                 smooth_per_row=True)
        row_center = symmetrize_plasma.shape[1] // 2  # Half of the image size
        # right side so that half[:,0] is on the axis, row_center = index column of symmetry
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
        print(integrated_ne)

        # save to .json
        in_name = os.path.basename(os.path.join(path_img_files, files_img[i]))
        stem, _ = os.path.splitext(in_name)  # e.g. "M1_X14"
        new_name = f"{stem}_electron_density.json"  # -> "M1_X14_edit.png"
        path_out = os.path.join(path_out_fold, new_name)

        save_images_to_json(image_unwrapped, electron_density, integrated_ne, path_out)
