import numpy as np
import matplotlib.pyplot as plt


def combine_images(image1, image2):
    """
    Combine two images stored in NumPy arrays so that each image occupies half of the width.

    Args:
    - image1: NumPy array representing the first image
    - image2: NumPy array representing the second image

    Returns:
    - combined_image: Combined image
    """
    # Check if both images have the same height
    assert image1.shape[0] == image2.shape[0], "Images must have the same height"

    # Calculate the width for each half
    half_width = min(image1.shape[1], image2.shape[1]) // 2

    # Create an array to store the combined image
    combined_image = np.zeros((image1.shape[0], half_width * 2, image1.shape[2]), dtype=np.uint8)

    # Copy pixels from the first image
    combined_image[:, :half_width, :] = image1[:, :half_width, :]

    # Copy pixels from the second image
    combined_image[:, half_width:, :] = image2[:, :half_width, :]

    return combined_image


# Example images (replace these with your actual images)
image1 = np.ones((100, 200, 3))
image2 = np.zeros((100, 200, 3))

# Combine the images
combined_image = combine_images(image1, image2)

# Display the combined image
plt.imshow(combined_image)
plt.axis('off')
plt.title('Combined Image')
plt.show()