import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sea
from matplotlib.widgets import Button
import rawpy

mpl.use('Qt5Agg')
global raw_image_path
raw_image_path  = 'Image4_A5_GRBG.dng'
global x, y, r,g,b
global demosaiced_image
global balanced_image
global original_image


def colour_box(r=.3,g=.4,b=.5):
    box = np.ones((100,100,3),dtype=float)
    box[:,:,0] = r
    box[:,:,1] = g
    box[:,: 2] = b
    return box
def remove_outliers(data, threshold):
    # Calculate the median and median absolute deviation (MAD)
    median = np.median(data)
    mad = np.median(np.abs(data - median))

    # Define the threshold for outlier detection
    lower_bound = median - threshold * mad
    upper_bound = median + threshold * mad

    # Remove outliers
    cleaned_data = np.clip(data, lower_bound, upper_bound)

    return cleaned_data

# white balance estimation for raw image
def estimate_white_balance(demosaiced):

    # Remove outliers from each color channel
    # cleaned_r = remove_outliers(demosaiced[::, ::, 0])
    # cleaned_g = remove_outliers(demosaiced[::, ::, 1])
    # cleaned_b = remove_outliers(demosaiced[::, ::, 2])

    # Calculate the average pixel values for each color channel
    avg_r = np.mean(demosaiced[:, :, 0])  # Red channel
    avg_g = np.mean(demosaiced[:, :, 1])  # Green channel
    avg_b = np.mean(demosaiced[:, :, 2])  # Blue channel

    print(avg_r, avg_g, avg_b)
    # Calculate white balance coefficients
    wb_coeffs = [avg_g/avg_r , 1.0, avg_g/avg_b]  # Adjusted to make green channel 1

    return wb_coeffs


def demosaic(raw_image_path):
    # Open the raw file
    with rawpy.imread(raw_image_path) as raw:
        # Extract the Bayer pattern image
        bayer_image = raw.raw_image_visible.astype('uint16')
        print(bayer_image.shape)
        bayer_pattern = "".join([chr(raw.color_desc[i]) for i in raw.raw_pattern.flatten()])
        print(bayer_pattern)
        height, width = bayer_image.shape
        demosaiced_image = np.zeros((height // 2, width // 2, 3))
        print(bayer_image.shape)
        # rgb = raw.postprocess()
        # Scale the array by its maximum intensity
        max_intensity = np.max(bayer_image)
        bayer_image = bayer_image / max_intensity

        # Apply gamma correction (gamma = 2.2)
        gamma = 2.2
        bayer_image = np.power(bayer_image, 1 / gamma)


    if bayer_pattern == 'RGGB':
        # GRGB
        temp_red = bayer_image[::2, ::2]
        # print(temp_red)
        temp_blue = bayer_image[1::2, 1::2]
        # print(temp_blue[1:10])
        temp_green = ((bayer_image[1::2, ::2] + raw_image[::2, 1::2])) / 2
    elif bayer_pattern == 'GRBG':
        temp_red = bayer_image[::2, 1::2]
        temp_green = (bayer_image[::2, ::2] + bayer_image[1::2, 1::2]) / 2
        temp_blue = bayer_image[1::2, ::2]


    demosaiced_image[:, :, 0] = temp_red
    demosaiced_image[:, :, 1] = temp_green
    demosaiced_image[:, :, 2] = temp_blue

    # plt.imshow(demosaiced_image)
    print(demosaiced_image.dtype)
    return height, width, demosaiced_image
def onclick(event):
    if event.xdata != None and event.ydata != None:
         x  = int(event.xdata +.5)
         y = int(event.ydata +.5)
         r = demosaiced_image[x,y,0]
         g = demosaiced_image[x,y,1]
         b = demosaiced_image[x,y,2]
         print(r,g,b)

         temp_coeffs = [r / g, 1, b / g]
         balanced_image_temp = np.zeros(demosaiced_image.shape)
         balanced_image_temp[:, :, 0] = demosaiced_image[:, :, 0] * temp_coeffs[2]
         balanced_image_temp[:, :, 1] = demosaiced_image[:, :, 1] * temp_coeffs[1]
         balanced_image_temp[:, :, 2] = demosaiced_image[:, :, 2] * temp_coeffs[1]
         temp_button = colour_box(r,g,b)

         # print(temp_button)
         plt.imshow(temp_button, interpolation='none')
         ax.imshow(balanced_image_temp)
         plt.draw()

def global_white_balance(event):
    ax.imshow(balanced_image)
    plt.draw()

def show_orignal(event):
    ax.imshow(original_image)
    plt.draw()

# initialize global variables
x ,y = 0,0
r,g,b = .5,.5,.5
selected = np.ones((100,100,3)).astype('uint16')
selected = selected * [r,g,b]
# get the demosiaced image
height, width, demosaiced_image = demosaic(raw_image_path)
dpi = 80
figsize = width / float(dpi), height / float(dpi)
original_image = demosaiced_image.copy()
# plt.imshow(demosaiced_image)
wb_coefficients = estimate_white_balance(demosaiced_image)
balanced_image = np.zeros(demosaiced_image.shape)
print(wb_coefficients)
balanced_image[:,:,0] = demosaiced_image[:,:,0] * wb_coefficients[0]
balanced_image[:,:,1] = demosaiced_image[:,:,1] * wb_coefficients[1]
balanced_image[:,:,2] = demosaiced_image[:,:,2] * wb_coefficients[2]

fig, ax = plt.subplots(figsize= (10,5))

half_balanced = demosaiced_image
half_balanced[:,0:(width//2),:] = balanced_image[:,0:(width//2),:]
# Plot the first image
ax.imshow(original_image)
ax.set_title('Image')
ax.axis('off')

# connect the mouse click
fig.canvas.mpl_connect('button_press_event', onclick)

# give the button axes locations
ax_original = plt.axes([0.01, 0.2, 0.15, 0.075])
ax_global = plt.axes([0.01, 0.4, 0.15, 0.075])
ax_select = plt.axes([0.01, 0.6, 0.15, 0.075])

# name and create buttons
button_global_white_balance = Button(ax_global, 'Global White Balance')
# button_selected_white_balance = Button(ax_select, 'Selected Point White Balance', color=(r,g,b), hovercolor=(r,g,b))
button_orginal = Button(ax_original, 'Original')
# Function to handle button clicks

# Connect buttons to their respective functions
button_global_white_balance.on_clicked(global_white_balance)
# button_selected_white_balance.on_clicked(white_balance_selected)
button_orginal.on_clicked(show_orignal)

# Adjust layout to prevent overlap
plt.tight_layout()

plt.show()