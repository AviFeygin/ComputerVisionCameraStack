import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
from IPython.display import Image
from scipy.ndimage import gaussian_filter
import os

verbose = True
mpl.use('Qt5Agg')
paths = ["waterfall_ev0.jpg","waterfall_ev_plus1.jpg", "waterfall_ev_minus1.jpg"]
levels = 3  # Adjust based on your needs
weightParam = [0.3, 0.3, 0.3]  # Adjust based on your needs


class ExposureFusion:
    def __init__(self, paths, levels=3, weights=[0.3, 0.3, 0.3]):
        self.levels = levels
        self.paths = paths
        # Load images, check for validity and consistency in dimensions
        self.images = [self.load_image(path) for path in paths]
        self.validate_images(self.images)
        self.weight_maps = []
        self.weights = weights + [0]

        self.gaussian_pyramid_G = None
        self.gaussian_pyramid_R = None
        self.gaussian_pyramid_B = None

        self.laplacian_pyramids_R = None
        self.laplacian_pyramids_G = None
        self.laplacian_pyramids_B = None

    def load_image(self, path):
        if verbose:
            print("got to load image")
        """Load an image from the given path."""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Image at path {path} could not be loaded.")
        return img.astype(np.float32) / 255.0

    @staticmethod
    def validate_images(images):
        print("validating images:")
        """Ensure all images have the same dimensions and number of channels."""
        first_shape = images[0].shape
        if not all(img.shape == first_shape for img in images):
            raise ValueError("All images must have the same dimensions and number of channels.")

    def separate_image_channels(self, image):
        if verbose:
            print("got to separate image channels")
        # Read the image
        # OpenCV reads images in BGR format, so we need to convert it to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Separate the channels
        R, G, B = cv2.split(image_rgb)
        R = np.array(R)
        G = np.array(G)
        B = np.array(B)

        return R, G, B  # return the separated channels as numpy arrays

    def blur_image(self, R, G, B):
        if verbose:
            print("got to blur image")
        # Apply Gaussian blur to each color channel
        blurred_R = gaussian_filter(R, sigma=2)
        blurred_G = gaussian_filter(G, sigma=2)
        blurred_B = gaussian_filter(B, sigma=2)

        return blurred_R, blurred_G, blurred_B

    def downsample(self, channel):  # input 1 channels numpy arrays, output 1 channels numpy arrays
        if verbose:
            print("got to downsample")
        # Downsample the image by taking every second pixel in both the width and height dimensions from colour channels
        # This results in an image that is 1/2 the width and 1/2 the height of the original
        downsized_channel = channel[::2, ::2]

        return downsized_channel

    def merge_color_channels(self, R, G, B):
        """
        Merge three color channels (R, G, B) into a single image.

        Parameters:
        - R, G, B: Numpy arrays representing the red, green, and blue channels of an image.

        Returns:
        - A PIL Image object representing the merged image.
        """
        if verbose:
            print("got to merge color channels")
        # check if the dimensions of the three channels are the same
        if not R.shape == G.shape == B.shape:
            raise ValueError("The dimensions of the R, G, and B channels must be the same.")

        # Stack the R, G, B channels depth-wise to form a single 3D array
        merged_array = np.dstack((R, G, B))
        print("merged array shape is ", merged_array.shape)
        return merged_array

        # Apply Gaussian filtering on the original image of size M x N. This will result in smoothing of sharp edges, and results in a blur image of size M/2 x N/2, due to downsampling.
        # Upsample the blur image to size M x N.
        # Subtract the blur image from the original image, to get the Laplacian at level 0.
        # For next iteration, replace the original image with blur image obtained in step 1 of size M/2 x N/2.

    def create_gaussian_pyramid(self, channel, levels, sigma=1):
        if verbose:
            print("got to create gaussian pyramid")
        # create a gaussian pyramid for a colour channel
        pyramid = []
        pyramid.append(channel)
        temp = channel

        for i in range(1, levels + 1, 1):
            # Apply Gaussian blur to the previous level
            blurred = gaussian_filter(temp, mode='reflect', sigma=sigma)

            # Downsample the blurred image
            downsampled = self.downsample(blurred)
            pyramid.append(downsampled)
            temp = downsampled

        # output list of numpy arrays
        print("pyramid size is ")
        print(len(pyramid))
        return pyramid

    def upsample(self, channel):
        """
        Upsample an image using nearest neighbor interpolation.

        :param image: Input image (numpy array).
        :return: Upsampled image as a numpy array.
        """
        if verbose:
            print("got to upsample")

        original = channel * 4
        # Calculate the ratio of the new size compared to the original size
        upsampled = np.zeros((channel.shape[0] * 2, channel.shape[1] * 2))
        upsampled[::2, ::2] = original
        upsampled = gaussian_filter(upsampled, sigma=1)

        return upsampled

    def gaussian_to_laplacian_numpy(self, gaussian_pyramid):
        if verbose:
            print("got to gaussian to laplacian")
            print("gaussian pyramid:")
            print(len(gaussian_pyramid))
            for i in range(0, len(gaussian_pyramid)):
                print("level", i, ":", gaussian_pyramid[i].shape)
        length = len(gaussian_pyramid)
        laplacian_pyramid = []
        # check if length of gaussian pyramid is greater than 1
        if length < 2:
            raise ValueError("Gaussian pyramid must have at least 2 levels.")

        # Add the last level of the Gaussian pyramid to the Laplacian pyramid
        laplacian_pyramid.append(gaussian_pyramid[0])
        print("length at start of loop is ", len(laplacian_pyramid))
        # Iterate over the Gaussian pyramid levels starting from the last level
        length = len(gaussian_pyramid) - 1
        # TODO fix this loop

        # loop in reverse through the gaussian pyramid
        for i in range(length, 0, -1):
            # Upsample the next level of the Gaussian pyramid
            upsampled = self.upsample(gaussian_pyramid[i])

            # Calculate the difference between the upsampled image and the current level
            laplacian = gaussian_pyramid[i - 1] - upsampled
            laplacian_pyramid.append(laplacian)

        laplacian_pyramid = laplacian_pyramid[::-1]
        laplacian_pyramid[length] = gaussian_pyramid[length]
        if verbose:
            print("laplacian pyramid 1")
            for i in range(0, len(laplacian_pyramid)):
                print("level", i, ":", laplacian_pyramid[i].shape)

        return laplacian_pyramid

    def boost_pyramid(self, laplacian_pyramid, weights):
        # input weights are from bottom of pyramid to top
        if verbose:
            print("got to boost pyramid")

        # check if length of weights is equal to the length of the laplacian pyramid -1
        if len(weights) != len(laplacian_pyramid) - 1:
            raise ValueError("Weights must have length equal to the number of levels in the Laplacian pyramid minus 1.")

        boosted_pyramid = laplacian_pyramid
        for i in range(0, len(laplacian_pyramid) - 1):
            boosted_pyramid[i] = laplacian_pyramid[i] * weights[i + 1]

        return boosted_pyramid

    def fuse_pyramids_using_maps(self, lp_0, lp_1, lp_2, wm_0, wm_1, wm_2):
        if verbose:
            print("got to fuse pyramids using maps")
        # check if the length of the pyramids and wm are the same
        if len(lp_0) != len(lp_1) or len(lp_0) != len(lp_2) or len(lp_0) != len(wm_0) or len(lp_0) != len(wm_1) or len(
                lp_0) != len(wm_2):
            raise ValueError("The Laplacian pyramids and weight maps must have the same number of levels.")

        fused_pyramid = []
        for i in range(0, len(lp_0)):
            fused_level = lp_0[i] * wm_0[i] + lp_1[i] * wm_1[i] + lp_2[i] * wm_2[i]
            fused_pyramid.append(fused_level)

        return fused_pyramid

    def reconstruct_image(self, gaussian, laplacian, weights):
        # input weights must be in the form of a list
        reconstructed = np.zeros(gaussian[0].shape)
        if verbose:
            print("got to reconstruct image")

        return reconstructed

    def get_weight_map(self, image):
        if verbose:
            print("got to get weight mapping")
        # L = 0.2126 * R + 0.7152 * G + 0.0722 * B
        R_WEIGHT = 0.2126
        G_WEIGHT = 0.7152
        B_WEIGHT = 0.0722
        # Calculate the luminance of the image
        luminance = R_WEIGHT * image[:, :, 0] + G_WEIGHT * image[:, :, 1] + B_WEIGHT * image[:, :, 2]
        # Apply Gaussian blur to the luminance image
        blurred_luminance = gaussian_filter(luminance, sigma=2)
        # Normalize the blurred luminance image
        normalized_luminance = (blurred_luminance - np.min(blurred_luminance)) / (
                    np.max(blurred_luminance) - np.min(blurred_luminance))
        # Calculate the weight map
        weight_map = 1 - normalized_luminance

        return weight_map

    def process(self):
        if verbose:
            print("got to process")
        """Process the images using exposure fusion."""
        # open the images
        R_0, G_0, B_0 = self.separate_image_channels(self.images[0])
        # Step 1: Create Gaussian pyramids for each color channel
        gaussian_pyramids_R_0 = self.create_gaussian_pyramid(R_0, self.levels)
        gaussian_pyramids_G_0 = self.create_gaussian_pyramid(G_0, self.levels)
        gaussian_pyramids_B_0 = self.create_gaussian_pyramid(B_0, self.levels)

        R_1, G_1, B_1 = self.separate_image_channels(self.images[1])
        # Step 1: Create Gaussian pyramids for each color channel
        gaussian_pyramids_R_1 = self.create_gaussian_pyramid(R_1, self.levels)
        gaussian_pyramids_G_1 = self.create_gaussian_pyramid(G_1, self.levels)
        gaussian_pyramids_B_1 = self.create_gaussian_pyramid(B_1, self.levels)

        R_2, G_2, B_2 = self.separate_image_channels(self.images[2])
        # Step 1: Create Gaussian pyramids for each color channel
        gaussian_pyramids_R_2 = self.create_gaussian_pyramid(R_2, self.levels)
        gaussian_pyramids_G_2 = self.create_gaussian_pyramid(G_2, self.levels)
        gaussian_pyramids_B_2 = self.create_gaussian_pyramid(B_2, self.levels)

        # Step 2: Convert Gaussian pyramids to Laplacian pyramids
        laplacian_pyramids_R_0 = self.gaussian_to_laplacian_numpy(gaussian_pyramids_R_0)
        laplacian_pyramids_G_0 = self.gaussian_to_laplacian_numpy(gaussian_pyramids_G_0)
        laplacian_pyramids_B_0 = self.gaussian_to_laplacian_numpy(gaussian_pyramids_B_0)
        laplacian_pyramids_R_1 = self.gaussian_to_laplacian_numpy(gaussian_pyramids_R_1)
        laplacian_pyramids_G_1 = self.gaussian_to_laplacian_numpy(gaussian_pyramids_G_1)
        laplacian_pyramids_B_1 = self.gaussian_to_laplacian_numpy(gaussian_pyramids_B_1)
        laplacian_pyramids_R_2 = self.gaussian_to_laplacian_numpy(gaussian_pyramids_R_2)
        laplacian_pyramids_G_2 = self.gaussian_to_laplacian_numpy(gaussian_pyramids_G_2)
        laplacian_pyramids_B_2 = self.gaussian_to_laplacian_numpy(gaussian_pyramids_B_2)

        # create the weight maps
        weight_map_0 = self.get_weight_map(self.images[0])
        weight_map_1 = self.get_weight_map(self.images[1])
        weight_map_2 = self.get_weight_map(self.images[2])

        total_weights = weight_map_0 + weight_map_1 + weight_map_2

        # Normalize each weight map so that the sum of weights at each pixel across the maps equals 1
        weight_map_0_normalized = weight_map_0 / total_weights
        weight_map_1_normalized = weight_map_1 / total_weights
        weight_map_2_normalized = weight_map_2 / total_weights

        # create the Gaussian pyramids for the weight maps
        gaussian_weight_map_0 = self.create_gaussian_pyramid(weight_map_0, self.levels)
        gaussian_weight_map_1 = self.create_gaussian_pyramid(weight_map_1, self.levels)
        gaussian_weight_map_2 = self.create_gaussian_pyramid(weight_map_2, self.levels)
        self.weight_maps = [weight_map_0, weight_map_1, weight_map_2]
        # print shape of gaussian weight maps
        print("gaussian_weight_map_0")
        for i in range(0, len(gaussian_weight_map_0)):
            print("level", i, ":", gaussian_weight_map_0[i].shape)
        # print shape of laplacian pyramids
        print("laplacian_pyramids_R_0")
        for i in range(0, len(laplacian_pyramids_R_0)):
            print("level", i, ":", laplacian_pyramids_R_0[i].shape)
        # multiply the weight maps by the laplace pyramids

        # fuse the colour channels
        red_fused = self.fuse_pyramids_using_maps(laplacian_pyramids_R_0, laplacian_pyramids_R_1,
                                                  laplacian_pyramids_R_2, gaussian_weight_map_0, gaussian_weight_map_1,
                                                  gaussian_weight_map_2)
        green_fused = self.fuse_pyramids_using_maps(laplacian_pyramids_G_0, laplacian_pyramids_G_1,
                                                    laplacian_pyramids_G_2, gaussian_weight_map_0,
                                                    gaussian_weight_map_1, gaussian_weight_map_2)
        blue_fused = self.fuse_pyramids_using_maps(laplacian_pyramids_B_0, laplacian_pyramids_B_1,
                                                   laplacian_pyramids_B_2, gaussian_weight_map_0, gaussian_weight_map_1,
                                                   gaussian_weight_map_2)

        # collapse the pyramids
        R = self.reconstruct_image(gaussian_pyramids_R_0, red_fused, self.weights)
        G = self.reconstruct_image(gaussian_pyramids_G_0, green_fused, self.weights)
        B = self.reconstruct_image(gaussian_pyramids_B_0, blue_fused, self.weights)

        # Merge the color channels into a single image
        fused_image = self.merge_color_channels(R, G, B)

        self.display_images(fused_image, self.images[1], self.weight_maps)

        return _

    def display_images(self, orignal, fused, maps, title="Image HDR Fusion"):
        if verbose:
            print("got to display image")
        # Create a figure and a set of subplots to show 4 images, the 3 original images and the fused image
        fig, axs = plt.subplots(1, 4, figsize=(12, 12))
        fig.suptitle(title)
        axs[0].imshow(fused)
        axs[0].set_title("Fused Image")
        axs[0].axis("off")
        for i in range(1, 4):
            axs[i].imshow(cv2.cvtColor(self.images[i - 1], cv2.COLOR_BGR2RGB))
            axs[i].set_title(self.paths[i - 1])
            axs[i].axis("off")
        plt.show()



# Example usage:
fusion = ExposureFusion(paths, levels, weightParam)
fused_image = fusion.process()

# Display the result
_ = cv2.imshow("Fused Image", fused_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
