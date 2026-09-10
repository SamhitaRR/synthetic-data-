import numpy as np
from scipy.signal import fftconvolve
from tifffile import imread, imwrite
import os

# Paths
#input_folder = "/data7/samhitar/nnUNet/nnUNet_raw/Dataset067_PSFbgVacuoles/imagesTr"
#output_folder = "/data7/samhitar/nnUNet/nnUNet_raw/Dataset067_PSFbgVacuoles/imagesTrPSF"

input_folder = "/media/elstore/users/samhita/038_paper/synhetic_image_variants/imagesTr"
output_folder = "/media/elstore/users/samhita/038_paper/synhetic_image_variants/imagesTrPSF"

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Load PSF
psf = imread("/media/elstore/users/samhita/01_timelapse/16092025/nnunet/psf/PSFs_all/PSF_NA1-4.tif").astype(np.float32)
psf /= psf.sum()  # Normalize PSF

# Loop over all TIFF images in the folder
for filename in os.listdir(input_folder):
    if filename.endswith(".tif") or filename.endswith(".tiff"):
        # Load image
        image_path = os.path.join(input_folder, filename)
        image = imread(image_path).astype(np.float32)
        
        # Convolve with PSF
        simulated = fftconvolve(image, psf, mode='same')
        
        # Save the result
        output_path = os.path.join(output_folder, filename)
        imwrite(output_path, simulated)

print("Convolution completed for all images!")

