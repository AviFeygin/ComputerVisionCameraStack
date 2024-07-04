import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.ndimage import gaussian_filter

# Load an image (replace 'your_image_path.jpg' with the path to your image)
image = plt.imread('Image1_sRGB.png')
image = np.mean(image, axis=-1)  # Convert to grayscale if it's a color image

# Apply FFT to the image
fourier_transform = fftshift(fft2(image))
# Get the frequency coordinates
freq_x = np.fft.fftshift(np.fft.fftfreq(image.shape[1]))
freq_y = np.fft.fftshift(np.fft.fftfreq(image.shape[0]))

x_filtered, y_filtered = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(image.shape[1])),
                                       np.fft.fftshift(np.fft.fftfreq(image.shape[0])))
# create a meshgrid
x, y = np.meshgrid(freq_x, freq_y)


# Create figure and 3D subplot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(151)
plt.imshow(image, cmap='gray')
plt.title('Original Image')

# plot the frequency domain
ax = fig.add_subplot(152, projection='3d')
ax.plot_surface(x, y, np.abs(fourier_transform), cmap='viridis')

# plot the gaussian filter
ax = fig.add_subplot(153, projection='3d')


# Create a grid of x, y values
x = np.linspace(-image.shape[1], image.shape[1], 100)
y = np.linspace(-image.shape[0], image.shape[0], 100)
x, y = np.meshgrid(x, y)

# Initial sigma value
initial_sigma = 20.0

# Create 2D Gaussian
def create_gaussian(x, y, sigma):

    return np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))


gaussian = create_gaussian(x, y, initial_sigma)

# Plot 3D Gaussian
surface = ax.plot_surface(x, y, gaussian, cmap='viridis', linewidth=0, alpha=0.8)
ax.set_title(f'2D Gaussian (Sigma: {initial_sigma:.2f})')

# Add a slider for Gaussian sigma
ax_sigma = plt.axes([0.2, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
slider = Slider(ax_sigma, 'Sigma', 1.0, 50.0, valinit=initial_sigma, valstep=1.0)


# Function to update the plot when the slider value changes
def update(val):
    sigma = slider.val

    # Clear the plot
    for collection in ax.collections:
        collection.remove()

    # Update the Gaussian plot
    updated_gaussian = create_gaussian(x, y, sigma)
    surface = ax.plot_surface(x, y, updated_gaussian, cmap='viridis', linewidth=0, alpha=0.8)

    # Update the title
    ax.set_title(f'2D Gaussian (Sigma: {sigma:.2f})')

    # Redraw the figure
    fig.canvas.draw_idle()

# plot not working to create the filtered frequencies
ax = fig.add_subplot(154, projection='3d')
# fourier_transform_filtered = fourier_transform * gaussian
# ax3.plot_surface(x_filtered, y_filtered, np.abs(fourier_transform_filtered), cmap='viridis')
#
# ax3.set_title('Filtered Fourier Transform')

ax = fig.add_subplot(155)
reconstructed_image = np.abs(ifft2(ifftshift(fourier_transform)))
plt.imshow(reconstructed_image, cmap='gray')
# Connect slider to update function
slider.on_changed(update)
plt.show()