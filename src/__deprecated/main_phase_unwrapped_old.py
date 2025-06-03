import matplotlib.pyplot as plt
import tifffile as tiff # Loads .tiff as a NumPy array
import numpy as np # low level image manipulation (matrix)
from scipy.ndimage import gaussian_filter
import os
import cmasher as cmr
import global_functions_im

## -- import .tiff image ---
base_dir = r"C:\\Users\\User\\z\\Desktop\\WUT\\Diplomka\\ZPRACOVÁNÍ\\Data testing\\Data"
file_path_1 = os.path.join(base_dir, "1,5usGD_100nsGW_230mcp___X8.tif")
file_path_ref = os.path.join(base_dir, "reference_image_X1.tif")
img1 = tiff.imread(file_path_1)  
img_ref = tiff.imread(file_path_ref)

## --- plot ---
clip_percent = 2
gamma_value  = 1.2
sgm          = 1

# 1) Make the overall figure bigger:
fig, ax = plt.subplots(1, 2, figsize=(16, 8))  # was (12,6)

# first image
filtered   = gaussian_filter(img1, sigma=sgm)
stretched  = functions_im.stretch_contrast(filtered, clip_percent=clip_percent)
gamma_img1 = functions_im.gamma_correction(stretched, gamma=gamma_value)
im1        = ax[0].imshow(gamma_img1, cmap=getattr(cmr.cm, 'nuclear'))
ax[0].set_title("Plasma image")

# shrink the colorbar to 80% of its default size
fig.colorbar(im1, ax=ax[0], shrink=0.8, pad=0.03)

# second image
filtered   = gaussian_filter(img_ref, sigma=sgm)
stretched  = global_functions_im.stretch_contrast(filtered, clip_percent=clip_percent)
gamma_img2 = functions_im.gamma_correction(stretched, gamma=gamma_value)
im2        = ax[1].imshow(gamma_img2, cmap=getattr(cmr.cm, 'nuclear'))
ax[1].set_title("Reference")

fig.colorbar(im2, ax=ax[1], shrink=0.8, pad=0.03)

plt.tight_layout()
plt.show()

## -- for loop to get phase difference ---
for i in range(2):
    if i == 0:
        img = img1
         # Define shift from center = center of 1st diff order
        X_do, Y_do = 5, -20  # move to the right from center by 30 px, up from center by 50 px
    else:
        img = img_ref
        X_do, Y_do = 1, -15
    # crop image
    img_crop = img[116:238, 289:519] 
    # padding image
    img_padded = np.pad(img_crop, pad_width = 10, mode = 'constant') # constant, wrap, symmetric

    ## -- Fourier transform ---
    # Compute the FFT and shift the zero frequency component to the center
    fft_image = np.fft.fft2(img_padded)
    fft_shifted = np.fft.fftshift(fft_image)
    # magnitude spectrum |fft_shifted| - (log-scaled for visibility)
    magnitude_spectrum = np.log1p(np.abs(fft_shifted))
    # magnitude_spectrum = np.abs(fft_shifted)
    # phase spectrum angle(fft_shifted)
    phase_spectrum = np.angle(fft_shifted)

    ## -- Fourier transform ---
    rows, cols = magnitude_spectrum.shape
    # create elliptical gaussian filter placed in [x, y] coordinate
    # Get Fourier center
    y_center, x_center = rows // 2, cols // 2 # rounded "floor" division operator
    # Apply the shift to the Fourier center
    shifted_peak_x = x_center + X_do
    shifted_peak_y = y_center + Y_do

    ## --- create elliptical gaussian mask ---

    if i == 0:
        # Define the elliptical mask centered at the shifted location
        a, b = cols // 9, rows // 9  # Semi-axes of the ellipse
        sgm = 2
    else:
        a, b = cols // 11, rows // 11
        sgm = 0.1
    # Create a coordinate 1D array...meshgrid
    y, x = np.ogrid[:rows, :cols] # y = vektor hodnot 0 až 144 (výška obr), x = vektor hodnot 0 až 250 (šířka obr), 
    # Define the elliptical mask equation at the new location
    ellipse_mask = (((x - shifted_peak_x) / a) ** 2 + ((y - shifted_peak_y) / b) ** 2) <= 1
    # Generate a Gaussian fade-out mask at the new location
    gaussian_fade = np.zeros_like(ellipse_mask, dtype=float) # zeros matrix with the size of ellipse_mask
    gaussian_fade[ellipse_mask] = 1
    gaussian_fade = gaussian_filter(gaussian_fade, sigma=sgm)  # Smooth transition

    ## --- Extract phase from 2D FFT ---
    # Apply the shifted mask to the Fourier spectrum
    filtered_fft = fft_shifted * gaussian_fade
    # Compute the inverse FFT
    ifft_shifted = np.fft.ifftshift(filtered_fft)
    reconstructed_image = np.fft.ifft2(ifft_shifted)
    phase_im = np.angle(reconstructed_image)  # Extract phase
    if i == 0:
        phase_im1 = phase_im
    else:
        phase_im_ref = phase_im

## --- plot ---
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
im1 = ax[0].imshow(phase_im1, cmap='cividis')
ax[0].set_title("Phase image 1")
fig.colorbar(im1, ax=ax[0])
im_ref = ax[1].imshow(phase_im_ref, cmap='cividis')
ax[1].set_title("Phase image reference")
fig.colorbar(im_ref, ax=ax[1])
plt.show()

## --- Phase shift image ---
# substract phases
phase_shift = phase_im1 - phase_im_ref
phase_shift = np.mod(phase_shift, 2 * np.pi) # modulo 2pi
plt.imshow(phase_shift, cmap='winter')
plt.title("Modulo Phase shift")
plt.colorbar(label="phase")
plt.show()

## --- Phase UNWRAP with step function algorithms ---
# openCV library
# Skimage library
from skimage.restoration import unwrap_phase
image_unwrapped = unwrap_phase(phase_shift, wrap_around=(False, False))

plt.imshow(image_unwrapped, cmap='jet')
plt.colorbar()
plt.title('Unwrapped Phase Map')
plt.show()

########################################################
#### Abel equation solving algorithm implementation ####
########################################################
## IDEA
# 1) Fourier Method -> v článku Gorbushkin, Shabanov
# 2) f-Interpolation
# 3) Backus-Gilbert-Method
## Other algorithms:
# Fourier-Hankel transform
# https://pyabel.readthedocs.io/en/v0.8.2/transform_methods/fh.html
# Pyable package:
# https://pyabel.readthedocs.io/en/latest/abel.html