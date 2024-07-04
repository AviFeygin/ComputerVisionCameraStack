import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifft2

# Load the image
image_path = 'Image1_ProPhoto.png'
image = plt.imread(image_path)
gray_image = np.mean(image, axis=-1)  # Convert to grayscale

# Perform 2D Fourier Transform
fourier_transform = fftshift(fft2(gray_image))

# Get the frequency coordinates
freq_x = np.fft.fftshift(np.fft.fftfreq(gray_image.shape[1]))
freq_y = np.fft.fftshift(np.fft.fftfreq(gray_image.shape[0]))

# Create a meshgrid for 3D plotting
x, y = np.meshgrid(freq_x, freq_y)

# Create a Gaussian map
sigma = 0.1  # Adjust the standard deviation as needed
gaussian_map = np.exp(-(x**2 + y**2) / (2 * sigma**2))

# Apply the Gaussian filter to the Fourier Transform
fourier_transform_filtered = fourier_transform * gaussian_map

# Create a meshgrid for 3D plotting of filtered transform
x_filtered, y_filtered = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(gray_image.shape[1])),
                                       np.fft.fftshift(np.fft.fftfreq(gray_image.shape[0])))

# Plot in 3D
fig = plt.figure(figsize=(15, 5))

# Plot Fourier Transform
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(x, y, np.abs(fourier_transform), cmap='viridis')
ax1.set_title('Fourier Transform')

# Plot Gaussian Map
ax2 = fig.add_subplot(132, projection='3d')
ax2.plot_surface(x, y, gaussian_map, cmap='viridis')
ax2.set_title('Gaussian Map')

# Plot Filtered Fourier Transform
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(x_filtered, y_filtered, np.abs(fourier_transform_filtered), cmap='viridis')
ax3.set_title('Filtered Fourier Transform')

# Show the plots
plt.show()