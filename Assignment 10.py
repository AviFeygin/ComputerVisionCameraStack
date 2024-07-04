import numpy as np
import cv2
import matplotlib.pyplot as plt


def histogram_equalization(img, nbr_bins=256):
    """ Apply histogram equalization to the given image. """
    # Get the image histogram
    hist, bins = np.histogram(img.flatten(), nbr_bins, [0, 256])
    # Calculate the cumulative distribution function
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / (cdf.max() + 1e9)  # avoid division by zero

    # Mask to avoid division by zero
    cdf_m = np.ma.masked_equal(cdf, 0)
    # Histogram equalization formula
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min() + 1e9)
    # Fill the masked values with 0
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    # Apply the equalization
    img2 = cdf[img.astype('uint8')]
    return img2


def clahe(img, clip_limit=2.0, grid_size=(8, 8)):
    if len(img.shape) == 3:
        # Convert image to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = clahe_channel(l, clip_limit, grid_size)
        l = l.astype('uint8')  # Ensure the L channel is in the correct type for merging
        merged = cv2.merge((l, a, b))
        final_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        return final_img
    else:
        return clahe_channel(img, clip_limit, grid_size)


def clahe_channel(channel, clip_limit=2.0, grid_size=(8, 8)):
    original_h, original_w = channel.shape
    channel = channel.astype('float32')
    grid_h, grid_w = grid_size

    # Calculate number of tiles
    tiles_x = int(np.ceil(original_w / grid_w))
    tiles_y = int(np.ceil(original_h / grid_h))

    # Pad the image to fit the tiles exactly
    img_padded = cv2.copyMakeBorder(channel, 0, tiles_y * grid_h - original_h, 0, tiles_x * grid_w - original_w,
                                    cv2.BORDER_REFLECT)

    # Create an output image
    output = np.zeros_like(img_padded)

    # Process each tile
    for i in range(tiles_y):
        for j in range(tiles_x):
            # Extract the tile
            row_start = i * grid_h
            col_start = j * grid_w
            tile = img_padded[row_start:row_start + grid_h, col_start:col_start + grid_w]

            # Apply histogram equalization
            eq_tile = histogram_equalization(tile, 256)

            # Place tile in the output image
            output[row_start:row_start + grid_h, col_start:col_start + grid_w] = eq_tile

    # Remove padding and ensure output matches original channel size
    output = output[:original_h, :original_w]
    return output


# Example usage:
img = cv2.imread('Image2_sRGB.png')
result = clahe(img, clip_limit=2.0, grid_size=(8, 8))

# Display the results
plt.figure(figsize=(12, 6))
plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(122), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title('CLAHE Image')
plt.show()
