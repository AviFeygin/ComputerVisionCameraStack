import sys

import scipy.ndimage
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QSlider, QLabel, QMessageBox
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.image as mpimg
from skimage.transform import resize
import cv2
from scipy.ndimage import zoom

verbose = True
import numpy as np
from scipy.fftpack import dct
from scipy.fftpack import idct
import matplotlib.pyplot as plt
import random
from typing import List
import numpy as np
from scipy.fft import dct, idct
import seaborn as sea
import traceback


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.image = None
        self.compressed_image = None
        self.setWindowTitle("Image Viewer")


        self.setGeometry(100, 100, 800, 600)
        self.layout = QVBoxLayout()
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.layout.addWidget(self.canvas)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(63)
        self.slider.setValue(32)
        self.original_bucket = 1
        self.slider.setTickPosition(QSlider.TicksBelow)  # Add ticks below the slider
        self.slider.setTickInterval(1)  # Set the interval for tick marks
        self.slider.valueChanged.connect(self.updateImage)
        self.layout.addWidget(self.slider)
        self.button = QPushButton("Open Image")
        self.button.clicked.connect(self.openImage)
        self.layout.addWidget(self.button)
        self.RMSE = None
        self.Y = None
        self.U = None
        self.V = None
        self.Y_modified = None
        self.U_modified = None
        self.V_modified = None

        self.Y_dct_original = []
        self.U_dct_original = []
        self.V_dct_original = []

        self.Y_dct_modified = []
        self.U_dct_modified = []
        self.V_dct_modified = []

        self.Y_decompressed = []
        self.U_decompressed = []
        self.V_decompressed = []

        self.setLayout(self.layout)
        self.heatmap = np.ones((8, 8), dtype=int)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.bin_chosen = 0
        self.dct_display = None
        self.displayed_cofficients = []
        self.dct_coefficients_original = []
        self.dct_coefficients_modified = []
        self.zigzag = np.array(
            [[0, 1, 5, 6, 14, 15, 27, 28],
             [2, 4, 7, 13, 16, 26, 29, 42],
             [3, 8, 12, 17, 25, 30, 41, 43],
             [9, 11, 18, 24, 31, 40, 44, 53],
             [10, 19, 23, 32, 39, 45, 52, 54],
             [20, 22, 33, 38, 46, 51, 55, 60],
             [21, 34, 37, 47, 50, 56, 59, 61],
             [35, 36, 48, 49, 57, 58, 62, 63], ])

        self.zigzag_order = [
            0, 1, 8, 16, 9, 2, 3, 10,
            17, 24, 32, 25, 18, 11, 4, 5,
            12, 19, 26, 33, 40, 48, 41, 34,
            27, 20, 13, 6, 7, 14, 21, 28,
            35, 42, 49, 56, 57, 50, 43, 36,
            29, 22, 15, 23, 30, 37, 44, 51,
            58, 59, 52, 45, 38, 31, 39, 46,
            53, 60, 61, 54, 47, 55, 62, 63
        ]

    def openImage(self):
        if verbose:
            print("got to openFileNameDialog")
        try:
            filename, _ = QFileDialog.getOpenFileName(self, "Open Image File", "", "Image files (*.jpg *.jpeg *.png *.gif)")

            if filename:
                self.image = mpimg.imread(filename)
                if self.image is None:
                    raise ValueError("Failed to load image")
                self.RMSE = np.zeros((self.image.shape))
                self.Y, self.U, self.V = self.downsize_yuv_image(cv2.cvtColor(self.image, cv2.COLOR_RGB2YUV))
                self.Y_dct_original, self.U_dct_original, self.V_dct_original = self.dct_transform_8x8_blocks_scipy(
                    self.Y, self.U, self.V)
                print("length of YUV BLOCKS")
                print(len(self.Y_dct_original))
                print(len(self.U_dct_original))
                print(len(self.V_dct_original))

                self.dct_display = np.array(self.Y_dct_original[self.bin_chosen])
                self.updateImage()
        except Exception as e:
            self.showErrorMessage("Error Opening Image", str(e))

    def updateImage(self):
        if verbose:
            print("got to updateImage")
        try:
            slider_value = self.slider.value()
            print(slider_value)
            temp_dct_coefficients = self.heatmap.reshape(-1)
            self.figure.clear()

            # Plot original image and its YUV components in the top row
            ax1 = self.figure.add_subplot(2, 4, 1)
            ax1.imshow(self.image)
            ax1.axis('off')
            ax1.set_title('Original')

            ax2 = self.figure.add_subplot(2, 4, 2)
            ax2.imshow(self.Y, cmap='gray')
            ax2.axis('off')
            ax2.set_title('Luma')

            ax3 = self.figure.add_subplot(2, 4, 3)
            ax3.imshow(self.U, cmap='gray')
            ax3.axis('off')
            ax3.set_title('U/4')

            ax4 = self.figure.add_subplot(2, 4, 4)
            ax4.imshow(self.V, cmap='gray')
            ax4.axis('off')
            ax4.set_title('V/4 ')

            # Plot compressed images in the bottom row
            # TODO: compress the image
            self.zero_dct_coefficients_above_threshold()
            # For testing purposes, we will only display the first 8x8 block of the Y channel
            # print("Y_dct_modified for slider value 32")
            # # Plot the Y_dct_modified as a heatmap using seaborn
            # plt.figure(figsize=(10, 8))
            # sea.heatmap(self.Y_dct_modified[0], annot=True, cmap='coolwarm', cbar=True, linecolor='black',
            #             linewidths=0.5)
            # plt.title('Y_dct_modified[0]')
            # plt.show()
            self.idct_transform_8x8_blocks_scipy()
            self.compressed_image = self.reconstruct_image(self.Y_modified, self.U_modified, self.V_modified)
            ax5 = self.figure.add_subplot(2, 4, 5)
            self.compressed_image =  cv2.cvtColor(self.compressed_image, cv2.COLOR_BGR2RGB)
            ax5.imshow(self.compressed_image)
            ax5.axis('off')
            ax5.set_title('Compressed')
            # print("sample of YUV BLOCKS1")
            # print(self.Y_dct_modified[0])

            self.RMSE = self.compute_RMSE(self.image, self.compressed_image)

            ax6 = self.figure.add_subplot(2, 4, 6)
            ax6.imshow(self.RMSE)
            ax6.axis('off')
            ax6.set_title('RMSE')
            # Heatmap of the DCT coefficients in ax7
            dct_active = self.zero_dct_heatmap(slider_value).astype(int)
            # print(dct_active)
            ax7 = self.figure.add_subplot(2, 4, 7)
            sea.heatmap(dct_active, annot=True, cmap='coolwarm', cbar=False, linecolor='black', linewidths=0.5, ax=ax7)
            ax7.axis('on')
            ax7.set_title('DCT Coefficients active')
            # Heatmap of the DCT coefficients in ax8
            ax8 = self.figure.add_subplot(2, 4, 8)
            self.dct_display = np.array(self.Y_dct_modified[self.bin_chosen])
            sea.heatmap(self.dct_display, annot=True, cmap='coolwarm', cbar=False, linecolor='black', linewidths=0.5,
                        ax=ax8)
            ax8.axis('on')
            ax8.set_title('Selected DCT Coefficients')

            self.canvas.draw()
        except Exception as e:
            self.showErrorMessage("Error Updating Image", str(e))

    def downsize_yuv_image(self, yuv_image):
        if verbose:
            print("got to downsize_yuv_image")
        try:
            # Split YUV channels
            y, u, v = cv2.split(yuv_image)

            # Downsize U channel
            downsampled_u = zoom(u, 0.5)

            # Downsize V channel
            downsampled_v = zoom(v, 0.5)

            return y, downsampled_u, downsampled_v
        except Exception as e:
            self.showErrorMessage("Error Downsizing YUV Image", str(e))
            return None, None, None

    def compute_RMSE(self, image1, image2):
        """
        Calculate the Root Mean Square Error (RMSE) per pixel between two RGB images.

        Args:
        image1 (numpy.ndarray): The first image as a numpy array.
        image2 (numpy.ndarray): The second image as a numpy array.

        Returns:
        numpy.ndarray: An array of RMSE values per pixel.
        """
        # Ensure the images have the same dimensions
        if image1.shape != image2.shape:
            raise ValueError("Images must have the same dimensions")

        # Convert images to float32 for precise calculation
        img1 = image1.astype(np.float32)
        img2 = image2.astype(np.float32)
        img2 = img2[:, :, ::-1]
        # Calculate the squared differences per pixel
        squared_diff = (img1 - img2) ** 2

        # Calculate the mean squared error per pixel
        rmse_per_pixel = np.sqrt(np.mean(squared_diff, axis=-1))  # Mean across the color channels
        mse_per_image = np.mean(squared_diff)  # Mean across the entire image



        self.RMSE = rmse_per_pixel
        print("RMSE per image")
        print(mse_per_image)
        return rmse_per_pixel

    # change the heatmap displayed
    def zero_dct_heatmap(self, slider):
        if verbose:
            print("got to zero_dct_heatmap")

        #     take in the slider value and set any values in the heatmap that are greater than the slider value to 0
        temp = self.zigzag.copy()
        temp[temp > slider] = 0
        result = np.where(temp > 0, 1, 0)
        result[0, 0] = 1
        return result

    def zero_dct_coefficients_above_threshold(self):
        """
        Zero out DCT coefficients at positions specified by the pattern where the pattern's values exceed a given threshold.
        Returns:
            None
        """
        if verbose:
            print("got to zero_dct_coefficients_above_threshold")

        zigzag = np.array(self.zigzag + 1)
        zigzag[zigzag > (self.slider.value() + 1)] = 0

        max_length = len(self.Y_dct_original)

        # Ensure the modified lists have the correct length
        self.Y_dct_modified = self.Y_dct_original.copy()
        self.U_dct_modified = self.U_dct_original.copy()
        self.V_dct_modified = self.V_dct_original.copy()

        # Process Y coefficients
        for i in range(len(self.Y_dct_original)):
            temp_y_np = np.array(self.Y_dct_original[i])
            temp_y_np = np.where(zigzag > 0, temp_y_np, 0)
            self.Y_dct_modified[i] = temp_y_np

        # Process U coefficients
        for i in range(len(self.U_dct_original)):
            temp_u_np = np.array(self.U_dct_original[i])
            temp_u_np = np.where(zigzag > 0, temp_u_np, 0)
            self.U_dct_modified[i] = temp_u_np

        # Process V coefficients
        for i in range(len(self.V_dct_original)):
            temp_v_np = np.array(self.V_dct_original[i])
            temp_v_np = np.where(zigzag > 0, temp_v_np, 0)
            self.V_dct_modified[i] = temp_v_np
        # it's ok up to here

    def on_click(self, event):
        if verbose:
            print("got to on_click")

        if event.inaxes:
            x, y = event.xdata, event.ydata  # Get the coordinates of the click
            print(f"Clicked at x={x}, y={y}")

            # Calculate the bucket coordinates
            bucket_x = int(x) // 8
            bucket_y = int(y) // 8
            # convert the x and y to one integer in the 1 to 64 range
            bucket = bucket_x * 8 + bucket_y

            print(f"Bucket coordinates, x,y,bin: ({bucket_x}, {bucket_y},{bucket})")

            # Return the bucket coordinates
            self.bin_chosen = bucket
            self.updateImage()

    def dct_transform_8x8_blocks_scipy(self, y_channel, u_channel, v_channel):
        if verbose:
            print("got to dct_transform_8x8_blocks_scipy")

        """
        Breaks each of the Y, U, and V channels into 8x8 blocks, applies the Discrete Cosine Transform (DCT)
        to each block using scipy's dct function, and returns the blocks.

        Args:
        y_channel (numpy.ndarray): The Y channel of the image.
        u_channel (numpy.ndarray): The U channel of the image.
        v_channel (numpy.ndarray): The V channel of the image.

        Returns:
        tuple: A tuple containing lists of the DCT blocks for the Y, U, and V channels.
        """

        def dct_blocks(channel):
            height, width = channel.shape
            dct_blocks = []
            for i in range(0, height, 8):
                for j in range(0, width, 8):
                    block = channel[i:i + 8, j:j + 8]
                    if block.shape[0] == 8 and block.shape[1] == 8:
                        dct_block = dct(dct(block.T, type=2, norm='ortho').T, type=2, norm='ortho')
                    else:
                        padded_block = np.zeros((8, 8))
                        padded_block[:block.shape[0], :block.shape[1]] = block
                        dct_block = dct(dct(padded_block.T, type=2, norm='ortho').T, type=2, norm='ortho')
                    dct_blocks.append(dct_block)
            return dct_blocks

        # Apply DCT to each block in each channel

        dct_y_blocks = dct_blocks(y_channel)
        dct_u_blocks = dct_blocks(u_channel)
        dct_v_blocks = dct_blocks(v_channel)
        # print("length of YUV BLOCKS")
        # print(len(dct_y_blocks))
        # print(len(dct_u_blocks))
        # print(len(dct_v_blocks))
        # print("last YUV" )
        # print(dct_y_blocks[-1])
        # print(dct_u_blocks[-1])
        # print(dct_v_blocks[-1])
        return dct_y_blocks, dct_u_blocks, dct_v_blocks

    def upsample(self, channel):
        if verbose:
            print("got to upsample")
        """
        Upsamples channels to match the dimensions of the Y channel by upsampling in both dimensions.

        Args:
        channel (numpy.ndarray): The U or V channel of the image.

        Returns:
        numpy.ndarray: The upsampled channel.
        """

        return zoom(channel, 2, order=0)

    import numpy as np
    from scipy.fftpack import idct
    import matplotlib.pyplot as plt

    def idct_transform_8x8_blocks_scipy(self, verbose=False):
        if verbose:
            print("got to idct_transform_8x8_blocks_scipy")

        Y_modified = []
        U_modified = []
        V_modified = []
        # print("length of YUV BLOCKS")
        # print(len(self.Y_dct_modified))
        # print(len(self.U_dct_modified))
        # print(len(self.V_dct_modified))
        # print("shape of YUV BLOCKS")
        # print(self.Y_dct_modified[0].shape)
        # print(self.U_dct_modified[0].shape)
        # print(self.V_dct_modified[0].shape)

        # IDCT for Y channel
        for i in range(len(self.Y_dct_modified)):
            Y_block = np.array(self.Y_dct_modified[i])
            # Apply 2D IDCT to each 8x8 block
            Y_modified.append(idct(idct(Y_block.T, type=2, norm='ortho').T, type=2, norm='ortho'))

        # IDCT for U and V channels
        for i in range(len(self.U_dct_modified)):
            U_block = np.array(self.U_dct_modified[i])
            V_block = np.array(self.V_dct_modified[i])
            U_modified.append(idct(idct(U_block.T, type=2, norm='ortho').T, type=2, norm='ortho'))
            V_modified.append(idct(idct(V_block.T, type=2, norm='ortho').T, type=2, norm='ortho'))
        # TODO:Problem is here zooming too much


        self.Y_decompressed = Y_modified
        self.U_decompressed = U_modified
        self.V_decompressed = V_modified


    def reconstruct_image(self, y_channel, u_channel, v_channel):
        """
        Reconstructs the RGB image from the Y, U, and V channels.
        """
        if verbose:
            print("got to reconstruct_image")

        temp_Y = self.Y_decompressed.copy()
        temp_U = self.U_decompressed.copy()
        temp_V = self.V_decompressed.copy()

        height, width = self.image.shape[:2]
        # print("height and width")
        # print(height)
        # print(width)
        blocks_width = width // 8
        blocks_height = height // 8
        final_Y = np.zeros((height, width))
        final_U = np.zeros((height//2, width//2))
        final_V = np.zeros((height//2, width//2))
        for i in range(blocks_width):
            for j in range(blocks_height):
                final_Y[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = temp_Y[i * blocks_height + j]

        for x in range(blocks_width//2):
            for y in range(blocks_height//2):
                final_U[x * 8:(x + 1) * 8, y * 8:(y + 1) * 8] = temp_U[x * blocks_height//2 + y]
                final_V[x * 8:(x + 1) * 8, y * 8:(y + 1) * 8] = temp_V[x * blocks_height//2 + y]

        final_U = self.upsample(final_U)
        final_V = self.upsample(final_V)

        # Merge the channels
        yuv_image = cv2.merge([final_Y,final_V,final_U])
        plt.imshow(yuv_image)
        # plt.show()
        # Convert YUV back to RGB
        rgb_image = cv2.cvtColor(yuv_image.astype(np.float32), cv2.COLOR_YUV2RGB)
        return rgb_image

    def showErrorMessage(self, title, message):
        # Capture the traceback information
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback_details = traceback.format_tb(exc_traceback)

        # Format the traceback to include function name and line number
        formatted_traceback = "".join(traceback_details)
        full_message = f"{message}\n\nTraceback:\n{formatted_traceback}"

        # Display the error message in a popup
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle(title)
        msg.setText(full_message)
        msg.setDetailedText(full_message)
        msg.exec_()

        # Print the error message to the terminal as well
        print(f"{title}: {full_message}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
