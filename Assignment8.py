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
paths = ["manor_ev0.jpg", "manor_ev_minus1.jpg", "manor_ev_plus1.jpg"]
levels = 8  # Adjust based on your needs
weightParam = [1,1,1,1,1,1,1,1]


class ExposureFusion:
    def __init__(self, paths, levels=3, weights=[1, 1, 1]):
        self.levels = levels
        self.paths = paths
        # Load images, check for validity and consistency in dimensions
        self.images = [self.load_image(path) for path in paths]
        self.validate_images(self.images)
        self.weight_maps = []
        self.weights = weights

        self.gaussian_pyramid_G = None
        self.gaussian_pyramid_R = None
        self.gaussian_pyramid_B = None

        self.laplacian_pyramids_R = None
        self.laplacian_pyramids_G = None
        self.laplacian_pyramids_B = None

    def blend(self, l1, l2, l3, w1, w2, w3):
        if verbose:
            print("got to blend")
        if not len(l1) == len(l2) == len(l3) == len(w1) == len(w2) == len(w3):
            raise ValueError("The Laplacian pyramids and weight maps must have the same number of levels.")
        blended = []
        for i in range(0, len(l1)):
            print("shape of weight_map_pyramid_L1_normalized is ", l1[i].shape)
            print("shape of weight_map_pyramid_W1_normalized is ", w1[i].shape)
        for i in range(len(l2)):
            blended.append(l1[i] * w1[i] + l2[i] * w2[i] + l3[i] * w3[i])
        return blended

    def blur_image(self, R, G, B):
        if verbose:
            print("got to blur image")
        # Apply Gaussian blur to each color channel
        blurred_R = gaussian_filter(R, sigma=2)
        blurred_G = gaussian_filter(G, sigma=2)
        blurred_B = gaussian_filter(B, sigma=2)

        return blurred_R, blurred_G, blurred_B

    def collapse_pyramid(self, pyramid):
        if verbose:
            print("Got to collapse pyramid")
        if not pyramid:
            raise ValueError("The pyramid is empty.")
        if len(pyramid) < 2:
            raise ValueError("The pyramid must have at least two levels to collapse.")

        # Start with the smallest level of the pyramid (this is typically a copy of the Gaussian pyramid's smallest level)
        current_level = pyramid[-1]
        # Iterate from the second to last level to the first one (index 0 is the first level of the Laplacian pyramid)
        for i in range(len(pyramid) - 2, -1, -1):
            if verbose:
                print(f"Collapsing level {i}")
                print("current level shape:", current_level.shape)
                print("pyramid shape:", pyramid[i].shape)

            # Upsample the current level to the size of the next level up
            upsampled = self.upsample(current_level)
            # Ensure the upsampled image is the same size as the next level up
            upsampled = cv2.resize(upsampled, (pyramid[i].shape[1], pyramid[i].shape[0]))
            # Add the upsampled image to the next level up
            current_level = pyramid[i] + upsampled

        return current_level

    def display_images(self, orignal, fused, title="Image HDR Fusion"):
        if verbose:
            print("got to display image")
        # Create a figure and a set of subplots to show 4 images, the 3 original images and the fused image
        fig, axs = plt.subplots(1, 4, figsize=(12, 12))
        print("size of fused is: ", fused.shape)
        fig.suptitle(title)
        axs[0].imshow(fused)
        axs[0].set_title("Fused Image")
        axs[0].axis("off")
        for i in range(1, 4):
            axs[i].imshow(cv2.cvtColor(self.images[i - 1], cv2.COLOR_BGR2RGB))
            axs[i].set_title(self.paths[i - 1])
            axs[i].axis("off")
        plt.show()

    def display_final_image(self, image, title="Final Image"):
        if verbose:
            print("got to display final image")
        # Display the final image
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
        plt.show()

    def display_maps(self, maps, title="Weight Maps"):
        if verbose:
            print("got to display maps")
        # Create a figure and a set of subplots to show 3 weight maps
        fig, axs = plt.subplots(1, 3, figsize=(12, 12))
        fig.suptitle(title)
        for i in range(len(maps)):
            axs[i].imshow(maps[i], cmap="gray")
            axs[i].set_title(self.paths[i])
            axs[i].axis("off")
        plt.show()

    def display_weight_pyramid(self, pyramid, title="Weight Pyramid"):
        if verbose:
            print("got to display weight pyramid")
        # Create a figure and a set of subplots to show the weight pyramid
        fig, axs = plt.subplots(1, len(pyramid), figsize=(12, 12))
        fig.suptitle(title)
        for i in range(len(pyramid)):
            axs[i].imshow(pyramid[i], cmap="gray")
            axs[i].set_title(f"Level {i}")
            axs[i].axis("off")
        plt.show()

    def display_pyramid(self, pyramid, title="Laplacian Pyramid"):
        if verbose:
            print("got to display pyramid")
        # Create a figure and a set of subplots to show the Laplacian pyramid
        fig, axs = plt.subplots(1, len(pyramid), figsize=(12, 12))
        fig.suptitle(title)
        for i in range(len(pyramid)):
            axs[i].imshow(pyramid[i], cmap="gray")
            axs[i].set_title(f"Level {i}")
            axs[i].axis("off")
        plt.show()

    def load_image(self, path):
        if verbose:
            print("got to load image")
        """Load an image from the given path."""
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Image at path {path} could not be loaded.")
        return img.astype(np.float32) / 255.0

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

    def gaussian_to_laplacian_numpy(self, gaussian_pyramid):
        if verbose:
            print("Converting Gaussian to Laplacian")
            print("Gaussian pyramid levels:", len(gaussian_pyramid))
            for i in range(len(gaussian_pyramid)):
                print("Level", i, "shape:", gaussian_pyramid[i].shape)

        if len(gaussian_pyramid) < 2:
            raise ValueError("Gaussian pyramid must have at least 2 levels.")

        laplacian_pyramid = []
        for i in range(len(gaussian_pyramid) - 1, 0, -1):
            upsampled = self.upsample(gaussian_pyramid[i])
            upsampled = cv2.resize(upsampled, (gaussian_pyramid[i - 1].shape[1], gaussian_pyramid[i - 1].shape[0]))
            laplacian = gaussian_pyramid[i - 1] - upsampled
            laplacian_pyramid.append(laplacian)

        laplacian_pyramid = laplacian_pyramid[::-1]  # Reverse the list to have the smallest level first
        # Append the smallest level as the base of the Laplacian pyramid
        laplacian_pyramid.append(gaussian_pyramid[-1])

        return laplacian_pyramid

    # mulitply weight pyramids by laplacian pyramids
    def weight_laplacian_pyramid(self, laplacian_pyramid, weight_pyramid):
        if verbose:
            print("got to weight laplacian pyramid")
        # check if the length of the laplacian pyramid and weight pyramid are the same
        if len(laplacian_pyramid) != len(weight_pyramid):
            raise ValueError("The Laplacian pyramid and weight pyramid must have the same number of levels.")

        weighted_pyramid = []
        for i in range(0, len(laplacian_pyramid)):
            weighted_pyramid.append(laplacian_pyramid[i] * weight_pyramid[i])

        return weighted_pyramid

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

        # check if length of weights is equal to the length of the laplacian pyramid -1
        if len(weights) != len(laplacian_pyramid) - 1:
            raise ValueError("Weights must have length equal to the number of levels in the Laplacian pyramid minus 1.")

        boosted_pyramid = laplacian_pyramid
        for i in range(0, len(laplacian_pyramid) - 1):
            boosted_pyramid[i] = laplacian_pyramid[i] * weights[i + 1]

        return boosted_pyramid

    def get_weight_map(self, image):
        # takes an image channel and computes the weight maps from the channels luminance
        # returns a list of numpy arrays representing the weight maps generated from the luminance of the image channel
        if verbose:
            print("got to get weight map")
            # Compute the luminance of the image channel

            R_luminance = self.luminosity(image[:, :, 0])

            G_luminance = self.luminosity(image[:, :, 1])

            B_luminance = self.luminosity(image[:, :, 2])

        # Compute the weight map using the luminance
        weight_map_R = R_luminance
        weight_map_G = G_luminance
        weight_map_B = B_luminance

        return weight_map_R, weight_map_G, weight_map_B

    def luminosity(self, value, sigma=.2):
        if verbose:
            print("got to luminosity")
        # calculate the gauss curve of the value
        output = lambda t: np.exp(-((t - .5) ** 2) / (2 * sigma ** 2))
        return output(value)

    def normalize_map(self, wm1, wm2, wm3, levels=3):
        if verbose:
            print("got to normalize map")
        # check if weight maps are the same size
        if not wm1.shape == wm2.shape == wm3.shape:
            raise ValueError("The weight maps must have the same dimensions.")

        # combine the weight maps
        wm1_n = self.create_gaussian_pyramid(wm1, levels)
        wm2_n = self.create_gaussian_pyramid(wm2, levels)
        wm3_n = self.create_gaussian_pyramid(wm3, levels)

        print("shape of wm1_n is ", wm1_n[0].shape)

        for i in range(0, levels):
            weight_map_sum = wm1_n[i] + wm2_n[i] + wm3_n[i]
            print("shape of weight_map_sum is ", weight_map_sum.shape)
            # normalize the weight map
            wm1_n[i] = wm1_n[i] / (weight_map_sum + 1e-6)
            wm2_n[i] = wm2_n[i] / (weight_map_sum + 1e-6)
            wm3_n[i] = wm3_n[i] / (weight_map_sum + 1e-6)
        # normalize the weight map
        print("shape of wm1_n is ", len(wm1_n))
        return wm1_n, wm2_n, wm3_n

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

    @staticmethod
    def validate_images(images):
        print("validating images:")
        """Ensure all images have the same dimensions and number of channels."""
        first_shape = images[0].shape
        if not all(img.shape == first_shape for img in images):
            raise ValueError("All images must have the same dimensions and number of channels.")

    def return_channel_pyramid(self, image):
        if verbose:
            print("got to create gaussian pyramid")
        # create a gaussian pyramid for a colour channel
        R, G, B = self.separate_image_channels(image)
        # Step 1: Create Gaussian pyramids for each color channel
        gaussian_pyramids_R = self.create_gaussian_pyramid(R, self.levels)
        gaussian_pyramids_G = self.create_gaussian_pyramid(G, self.levels)
        gaussian_pyramids_B = self.create_gaussian_pyramid(B, self.levels)
        # output list of numpy arrays
        return gaussian_pyramids_R, gaussian_pyramids_G, gaussian_pyramids_B

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

    def process(self):
        if verbose:
            print("got to process")
        """Process the images using exposure fusion."""

        # open the images
        gaussian_pyramids_R_0, gaussian_pyramids_G_0, gaussian_pyramids_B_0 = self.return_channel_pyramid(
            self.images[0])

        gaussian_pyramids_R_1, gaussian_pyramids_G_1, gaussian_pyramids_B_1 = self.return_channel_pyramid(
            self.images[1])

        gaussian_pyramids_R_2, gaussian_pyramids_G_2, gaussian_pyramids_B_2 = self.return_channel_pyramid(
            self.images[2])

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
        # From computed weight map, compute Gaussian Pyramid. Multiple this against Laplace Pyramid.
        weight_map_R0, weight_map_G0, weight_map_B0 = self.get_weight_map(self.images[0])
        weight_map_R1, weight_map_G1, weight_map_B1 = self.get_weight_map(self.images[1])
        weight_map_R2, weight_map_G2, weight_map_B2 = self.get_weight_map(self.images[2])

        weight_map_pyramid_R0_normalized, weight_map_pyramid_R1_normalized, weight_map_pyramid_R2_normalized = self.normalize_map(
            weight_map_R0, weight_map_R1, weight_map_R2, levels)
        weight_map_pyramid_G0_normalized, weight_map_pyramid_G1_normalized, weight_map_pyramid_G2_normalized = self.normalize_map(
            weight_map_G0, weight_map_G1, weight_map_G2, levels)
        weight_map_pyramid_B0_normalized, weight_map_pyramid_B1_normalized, weight_map_pyramid_B2_normalized = self.normalize_map(
            weight_map_B0, weight_map_B1, weight_map_B2, levels)

        for i in range(0, levels):
            print("shape of weight_map_pyramid_R0_normalized is ", weight_map_pyramid_R0_normalized[i].shape)

        self.display_maps([weight_map_pyramid_R0_normalized[0], weight_map_pyramid_R1_normalized[0],
                            weight_map_pyramid_R2_normalized[0]], "Weight Maps Normalized")

        # TODO here
        # Step 4: Blend the Laplacian pyramids using he weight maps
        R_blended_laplacian = self.blend(laplacian_pyramids_R_0, laplacian_pyramids_R_1, laplacian_pyramids_R_2,
                                weight_map_pyramid_R0_normalized, weight_map_pyramid_R1_normalized,
                                weight_map_pyramid_R2_normalized)
        G_blended_laplacian = self.blend(laplacian_pyramids_G_0, laplacian_pyramids_G_1, laplacian_pyramids_G_2,
                                weight_map_pyramid_G0_normalized, weight_map_pyramid_G1_normalized,
                                weight_map_pyramid_G2_normalized)
        B_blended_laplacian = self.blend(laplacian_pyramids_B_0, laplacian_pyramids_B_1, laplacian_pyramids_B_2,
                                weight_map_pyramid_B0_normalized, weight_map_pyramid_B1_normalized,
                                weight_map_pyramid_B2_normalized)
        self.display_weight_pyramid(R_blended_laplacian, "Blended Laplacian Pyramid")
        self.display_weight_pyramid(G_blended_laplacian, "Blended Laplacian Pyramid")
        self.display_weight_pyramid(B_blended_laplacian, "Blended Laplacian Pyramid")

        # Step 5: Reconstruct the blended Laplacian pyramids to obtain the fused image
        R = self.collapse_pyramid(R_blended_laplacian)
        G = self.collapse_pyramid(G_blended_laplacian)
        B = self.collapse_pyramid(B_blended_laplacian)

        self.display_maps([R, G, B], "Color Channels")
        print("shape of R is ", R.shape)
        print("shape of G is ", G.shape)
        print("shape of B is ", B.shape)

        # Merge the color channels into a single image
        fused_image = self.merge_color_channels(R, G, B)

        print("shape of fused image is ", fused_image.shape)

        # Display the results
        self.display_pyramid(R_blended_laplacian, "R Blended Laplacian")
        self.display_maps([weight_map_pyramid_R0_normalized[0], weight_map_pyramid_R1_normalized[0],
                        weight_map_pyramid_R2_normalized[0]], "Weight Maps")
        self.display_weight_pyramid(weight_map_pyramid_R2_normalized, "Gaussian Weight Pyramid")
        self.display_images(self.images[1], fused_image, "Image HDR Fusion")
        self.display_final_image(fused_image, "Final Image")
        return fused_image


# Example usage:
fusion = ExposureFusion(paths, levels, weightParam)
fused_image = fusion.process()
# Display the results