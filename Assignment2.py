# Importing the 'colour' library for color science calculations and visualizations
import colour
# Importing all functions from 'colour.plotting' module for plotting capabilities related to color science
from colour.plotting import *
# Importing the 'tkinter' library for creating GUI applications
import tkinter as tk
# Importing the 'cv2' for OpenCV to handle image operations
import cv2
# Importing 'numpy' for numerical operations on arrays and matrices
import numpy as np
# Importing 'matplotlib.pyplot' for creating static, interactive, and animated visualizations in Python
import matplotlib.pyplot as plt
# Importing 'seaborn' for making statistical graphics in Python
import seaborn as sea
# Setting the default seaborn style for plots
sea.set()

# Define a function to open an image from a given path
def open_image(image_path):
    # Read the image from the given path with unchanged color channels
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    # Convert the image from BGR to RGB color space and scale pixel values to 0-1 range
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 65536.0
    return image_rgb

# Define a function to convert an image from RGB to XYZ color space
def convert_to_XYZ(image, type):
    # Initialize an empty string for the color model
    model = ""
    # If the type is AdobeRGB, convert using Adobe RGB (1998) color space
    if type == "AdobeRGB":
        RGB_image = colour.RGB_to_XYZ(image, 'Adobe RGB (1998)', apply_cctf_decoding=True)
    # If the type is ProPhoto, convert using ProPhoto RGB color space
    elif type == "ProPhoto":
        RGB_image = colour.RGB_to_XYZ(image, 'ProPhoto RGB', apply_cctf_decoding=True)
    # If the type is sRGB, convert using sRGB color space
    elif type == "sRGB":
        model = 'sRGB'
        RGB_image = colour.sRGB_to_XYZ(image, apply_cctf_decoding=True)
    return RGB_image

# Define a function to convert an image from XYZ to CIE L*a*b* color space
def XYZ_to_LAB(image):
    LAB_image = colour.XYZ_to_Lab(image)
    return LAB_image

# Define the main function where the script starts execution
def main():
    # Define paths to the images that will be processed
    image1_path = 'Image1_AdobeRGB.png'  # Example path, replace with actual image path
    image2_path = 'Image1_ProPhoto.png'  # Example path, replace with actual image path
    image3_path = 'Image1_sRGB.png'      # Example path, replace with actual image path

    # Open the images using the previously defined function
    image1 = open_image(image1_path)
    image2 = open_image(image2_path)
    image3 = open_image(image3_path)

    # List of RGB color models to iterate over
    rgb_models = ['ProPhoto', 'AdobeRGB', 'sRGB']

    # Loop through the RGB color models and process the images
    for model in rgb_models:
        # Convert each image to XYZ color space
        xyz_image1 = convert_to_XYZ(image1, "AdobeRGB")
        xyz_image2 = convert_to_XYZ(image2, 'ProPhoto')
        xyz_image3 = convert_to_XYZ(image3, 'sRGB')

        # Convert XYZ images to Lab color space
        lab_image1 = XYZ_to_LAB(xyz_image1)
        lab_image2 = XYZ_to_LAB(xyz_image2)
        lab_image3 = XYZ_to_LAB(xyz_image3)
        # Example code to print pixel values, commented out

        # Create a 3x2 subplot layout for displaying images and their Lab color distributions
        fig, axs = plt.subplots(3, 2, sharex='col', sharey='col')
        # Display the RGB images on the left column
        axs[0, 0].imshow(image1)
        axs[2, 0].imshow(image2)
        axs[1, 0].imshow(image3)
        # Plot the distribution of a* and b* values from the Lab images on the right column, coloring by L* value
        axs[0, 1].scatter(lab_image1[:, :, 1], lab_image1[:, :, 2], c=lab_image1[:, :, 0], cmap='hot')
        axs[1, 1].scatter(lab_image2[:, :, 1], lab_image2[:, :, 2], c=lab_image2[:, :, 0], cmap='hot')
        axs[2, 1].scatter(lab_image3[:, :, 1], lab_image3[:, :, 2], c=lab_image3[:, :, 0], cmap='hot')
        # Set the x and y axis limits for the right column plots
        for ax in axs[:, 1]:
            ax.set_xlim(-100, 100)
            ax.set_ylim(-100, 100)

        # Display the plots
        plt.show()
        print("finished")

# Check if the script is run directly (as opposed to being imported) and call main if it is
if __name__ == "__main__":
    main()