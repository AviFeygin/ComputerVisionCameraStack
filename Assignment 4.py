import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rawpy
mpl.use('Qt5Agg')
raw_image_path = 'Image4_A4_RGGB.DNG'
# Path to the rawf image fil

def demosaic_RGGB(img):
    # Get the dimensions of the input image
    height, width = img.shape

    # Create an empty array to store the demosaiced image with three channels (RGB)
    demosaiced_img = np.zeros((height, width, 3))

    # Iterate over each pixel in the image
    for i in range(1,height-1):
        for j in range(1,width-1):
            # Check if the current row is even
            if i % 2 == 0:
                # Check if the current column is even
                if j % 2 == 0:  # Red pixel

                    # print('got to here red pixel')
                    # red channel set
                    demosaiced_img[i, j, 0] = img[i, j]
                    # Green channel set
                    # print((img[i-1, j]+img[i+1, j]+img[i, j-1]+img[i, j+1])/4)
                    demosaiced_img[i,j,1] = (img[i-1, j]+img[i+1, j]+img[i, j-1]+img[i, j+1])/5
                    # print(demosaiced_img[i, j, :])
                    # blue channel set
                    # print((img[i-1, j+1]+img[i-1, j-1]+img[i+1, j-1]+img[i+1, j+1])/4)
                    demosaiced_img[i,j,2] =(img[i-1, j+1]+img[i-1, j-1]+img[i+1, j-1]+img[i+1, j+1])/4.2
                    # print(demosaiced_img[i,j,:])
                else:  # Green pixel
                    # Interpolate red channel values using neighboring pixels
                    demosaiced_img[i, j, 0] = (img[max(0, i - 1), j] + img[min(height - 1, i + 1), j]) / 2
                    # For green pixels, assign the green channel value directly from the single-channel image
                    demosaiced_img[i, j, 1] = img[i, j]
                    # Interpolate blue channel values using neighboring pixels
                    demosaiced_img[i, j, 2] = (img[i, max(0, j - 1)] + img[i, min(width - 1, j + 1)]) / 2
                    # print(demosaiced_img[i, j, :])
            else:
                # Check if the current COLUMN is even
                if j % 2 == 0:  # Green pixel
                    # Interpolate red channel values using neighboring pixels
                    demosaiced_img[i, j, 0] = (img[i, max(0, j - 1)] + img[i, min(width - 1, j + 1)]) / 2
                    # For green pixels, assign the green channel value directly from the single-channel image
                    demosaiced_img[i, j, 1] = img[i, j]
                    # Interpolate blue channel values using neighboring pixels
                    demosaiced_img[i, j, 2] = (img[max(0, i - 1), j] + img[min(height - 1, i + 1), j]) / 2
                    # print(demosaiced_img[i, j, :])

                else:
                    # COLUMN is ODD
                    # Blue pixel
                    # Interpolate red channel values using neighboring pixels
                    demosaiced_img[i, j, 0] = (img[max(0, i - 1), max(0, j - 1)] + img[
                        min(height - 1, i + 1), min(width - 1, j + 1)] + img[min(height - 1, i + 1), max(0, j - 1)] +
                                               img[max(0, i - 1), min(width - 1, j + 1)]) / 4
                    # Interpolate green channel values using neighboring pixels
                    demosaiced_img[i, j, 1] = (img[max(0, i - 1), j] + img[min(height - 1, i + 1), j] + img[
                        i, max(0, j - 1)] + img[i, min(width - 1, j + 1)]) / 5
                    # For blue pixels, assign the blue channel value directly from the single-channel image
                    demosaiced_img[i, j, 2] = img[i, j]
                    # print(demosaiced_img[i, j, :])

    # Return the demosaiced image
    demosaiced_img[::, ::, 1] = demosaiced_img[::, ::, 1] / 1.2
    return demosaiced_img


def demosaic_GRGB(img):
        # Get the dimensions of the input image
 height, width = img.shape

        # Create an empty array to store the demosaiced image with three channels (RGB)
 demosaiced_img = np.zeros((height, width, 3))

 for i in range(1, height - 1):
     for j in range(1, width - 1):

        if i % 2 == 0: #  if the current row is even
            #  if the current column is even
            if j % 2 == 0:  # Green Pixel

                # print('got to here red pixel')
                # RED channel set
                # print((img[i-1, j]+img[i+1, j]+img[i, j-1]+img[i, j+1])/4)
                demosaiced_img[i, j, 0] = (img[i, j+1] + img[i, j-1])  / 2
                # Green channel set
                demosaiced_img[i, j, 1] = img[i, j]
                # print(demosaiced_img[i, j, :])
                # blue channel set
                # print((img[i-1, j+1]+img[i-1, j-1]+img[i+1, j-1]+img[i+1, j+1])/4)
                demosaiced_img[i, j, 2] = ((img[i - 1, j] + img[i+1, j]) / 2.5)
                # print(demosaiced_img[i,j,:])

            else:  # RED pixel COL is ODD
                # Interpolate red channel values using neighboring pixels
                demosaiced_img[i, j, 0] = img[i, j]
                # For green pixels, assign the green channel value directly from the single-channel image
                demosaiced_img[i, j, 1] = (img[i,j+1]+img[i,j-1]+img[i+1,j]+img[i-1,j]) / 5
                # Interpolate blue channel values using neighboring pixels
                demosaiced_img[i, j, 2] = (img[i+1, j+1]+img[i+1,j-1]+img[i-1,j+1]+img[i-1,j-1]) / 4.5
                # print(demosaiced_img[i, j, :])

        else: #ROW is ODD
            # Check if the current COLUMN is even
            if j % 2 == 0:  # BLUE
                # Interpolate red channel values using neighboring pixels
                demosaiced_img[i, j, 0] = (img[i+1, j+1]+img[i+1,j-1]+img[i-1,j+1]+img[i-1,j-1]) / 4
                # For green pixels, assign the green channel value directly from the single-channel image
                demosaiced_img[i, j, 1] = (img[i,j+1]+img[i,j-1]+img[i+1,j]+img[i-1,j]) / 5
                # Interpolate blue channel values using neighboring pixels
                demosaiced_img[i, j, 2] = img[i, j]
                # print(demosaiced_img[i, j, :])

            else:
                # COLUMN is ODD
                # GREEN
                # Interpolate red channel values using neighboring pixels
                demosaiced_img[i, j, 0] = ((img[i - 1, j] + img[i+1, j]) / 2)
                # assign green
                demosaiced_img[i, j, 1] = img[i, j]
                # interpolate BLUE
                demosaiced_img[i, j, 2] =(img[i, j+1] + img[i, j-1])  / 2.5
  # Return the demosaiced image

 demosaiced_img[::, ::, 1] = demosaiced_img[::, ::, 1] / 1.2
 demosaiced_img[:]
 return demosaiced_img

# Open the raw image file
with rawpy.imread(raw_image_path) as raw:
    # Extract the Bayer pattern image
    bayer_image = raw.raw_image_visible.astype('uint16')
    print(bayer_image.shape)
    bayer_pattern = "".join([chr(raw.color_desc[i]) for i in raw.raw_pattern.flatten()])
    print(bayer_pattern)
    height, width = bayer_image.shape
    new_bayer_image = np.zeros((height, width, 3))
    print(bayer_image.shape)
    rgb = raw.postprocess()
    # Scale the array by its maximum intensity
    max_intensity = np.max(bayer_image)
    bayer_image = bayer_image / max_intensity


    # Apply gamma correction (gamma = 2.2)
    gamma = 2.2
    bayer_image = np.power(bayer_image, 1 / gamma)


    if bayer_pattern == 'RGGB':
     # slicing              [row, col, channel]
     red_channel = bayer_image[::2, ::2]
     green_channel = bayer_image[1::2, ::2]
     green_channel2 = bayer_image[::2, 1::2]
     blue_channel = bayer_image[1::2, 1::2]
     new_bayer_image[::2, ::2, 0] = red_channel
     new_bayer_image[1::2, ::2,1] = green_channel
     new_bayer_image[::2, 1::2,1] = green_channel2
     new_bayer_image[1::2, 1::2,2] = blue_channel

    elif bayer_pattern == 'GRBG':
        # slicing              [row, col, channel]
     red_channel = bayer_image[::2, 1::2]
     green_channel = bayer_image[::2, ::2]
     green_channel2 = bayer_image[1::2, 1::2]
     blue_channel = bayer_image[1::2, ::2]
     new_bayer_image[::2, 1::2, 0] = red_channel
     new_bayer_image[::2, ::2, 1] = green_channel
     new_bayer_image[1::2, 1::2,1] = green_channel2
     new_bayer_image[1::2, ::2,2] = blue_channel

    if bayer_pattern == 'GRBG':
        demosaiced_image = demosaic_GRGB(bayer_image)
    elif bayer_pattern == 'RGGB':
        demosaiced_image = demosaic_RGGB(bayer_image)
    print(demosaiced_image.shape)
    print(demosaiced_image[100:200,100:200,:])


    # imshow(bayer_image, cmap='gray', interpolation='none')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle('Image processing, demosaicing')
    plt.axis('off')
    plt.title('Bayer grey Image')
    ax1.imshow(bayer_image, cmap='gray', aspect  = 'equal', interpolation = 'none')

    plt.axis('off')
    plt.title('Bayer Image coloured')
    ax2.imshow(new_bayer_image, aspect = 'equal', interpolation = 'none')

    plt.axis('off')
    plt.title('Demosaiced image')
    ax3.imshow(demosaiced_image, interpolation = 'none')
    # Display the Bayer image
    ax4.imshow(rgb)
    plt.title('original image')
    plt.axis('off')  # Turn off axis
    fig.tight_layout()
    plt.show()