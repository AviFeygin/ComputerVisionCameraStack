import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QFileDialog, QLabel, QSlider
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import cv2
import numpy as np
from scipy.fftpack import dct, idct
verbose = True # Set to True to enable print statements
import random
from scipy.ndimage import zoom

class App(QMainWindow):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 Matplotlib Example - YUV Image Viewer'
        self.left = 10
        self.top = 10
        self.width = 1600
        self.height = 1200
        self.initUI()
        self.image = None
        # holds the DCT coefficients for each 8x8 block in the image
        self.bins = []
        self.image_width = None
        self.image_height = None
        self.width_bins = None
        self.height_bins = None
        self.zigzag = np.array([ [ 0,  1,  5,  6, 14, 15, 27, 28], [ 2,  4,  7, 13, 16, 26, 29, 42], [ 3,  8, 12, 17, 25, 30, 41, 43], [ 9, 11, 18, 24, 31, 40, 44, 53], [10, 19, 23, 32, 39, 45, 52, 54], [20, 22, 33, 38, 46, 51, 55, 60], [21, 34, 37, 47, 50, 56, 59, 61], [35, 36, 48, 49, 57, 58, 62, 63], ])
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
        self.selected_number = 63


    def initUI(self):
        if verbose:
            print("got to initUI")
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.widget = QWidget(self)
        self.layout = QVBoxLayout(self.widget)

        self.button = QPushButton('Open Image')
        self.button.clicked.connect(self.openFileNameDialog)

        self.label = QLabel("Select an image to view its YUV components.")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(1)  # Set minimum value to 1
        self.slider.setMaximum(63)  # Set maximum value to 65
        self.slider.setValue(62)  # Set a default value within the new range
        self.slider.valueChanged.connect(self.update_plots)

        self.layout.addWidget(self.button)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.slider)

        self.setCentralWidget(self.widget)

    def openFileNameDialog(self):
        if verbose:
            print("got to openFileNameDialog")

        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                            "All Files (*);;JPEG (*.jpg;*.jpeg);;PNG (*.png)", options=options)
        if fileName:
            self.load_image(fileName)
            self.displayImage(fileName)

    def load_image(self, filePath):
        if verbose:
            print("got to load_image")
        # Read the image
        img = cv2.imread(filePath)
        self.image = img
        self.image_width = img.shape[1]
        self.image_height = img.shape[0]
        self.width_bins = self.image_width // 8
        self.height_bins = self.image_height // 8
        self.selected_bin = None
        self.bins = self.get_bins(img)
        # print(len(self.bins))
        # print(self.bins[0])

        # Create a larger figure to hold the plots. Adjust figsize to increase the size.

    def displayImage(self, filePath, selected):
        if verbose:
            print("got to displayImage")

        yuv_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(yuv_img)

        # Create a larger figure to hold the plots. Adjust figsize to increase the size.
        self.fig, self.axs = plt.subplots(2, 4, figsize=(20, 10))  # Increase figsize here

        # Display the original and YUV component images
        self.axs[0, 0].imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        self.axs[0, 1].imshow(y, cmap='gray')
        self.axs[0, 2].imshow(u, cmap='gray')
        self.axs[0, 3].imshow(v, cmap='gray')
        print("got to here")
        # selected bin to display the DCT coefficients as a heatmap
        print(self.bins[0])
        # Display the DCT coefficients as a heatmap




        # Remove axis labels for cleanliness
        for ax in self.axs.flat:
            ax.label_outer()

        # Adjust the layout to make room for all subplots
        plt.tight_layout()

        # Display the figure in the PyQt5 window
        self.canvas = FigureCanvas(self.fig)
        self.layout.addWidget(self.canvas)

        # Connect the click event
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        if verbose:
            print("got to onclick")
        # Check if the click was on one of the axes
        if event.inaxes is not None:
            # Convert click coordinates to pixel values
            x, y = int(event.xdata), int(event.ydata)
            print(f"selected location - x: {x}, y: {y}")
            # self.label.setText(f"Last click location - x: {x}, y: {y}")
            self.label.setText(f"Selected bin Row, Col: {x // 8}, {y // 8}")
            self.selected_bin = self.bins[(y // 8) * self.width_bins + (x // 8)]
            self.selected_bin_dct = self.dct2(self.selected_bin)

    def get_bins(self, img):
        if verbose:
            print("got to get_bins")
        bins = []
        for i in range(0, img.shape[0], 8):
            for j in range(0, img.shape[1], 8):
                bins.append(self.dct2(img[i:i+8, j:j+8]))
        self.bins = bins
        pass

    def update_plots(self):
        if verbose:
            print("got to update_plots")

        # Placeholder for the logic to update the second row of plots
        # based on the slider's value. You may want to clear and redraw
        # the second row of plots here.

        self.label.setText(f"Slider Value: {self.slider.value()}")
        # update the heatmap plot based on the slider value
        self.plot_masked_heatmap(self.slider.value())

    def plot_rmse(self):
        if verbose:
            print("got to plot_rmse")
        # Placeholder for the logic to plot the RMSE values
        # based on the selected bins. You may want to clear and redraw
        # the RMSE plot here.
        pass

    def plot_compressed(self):
        if verbose:
            print("got to plot_compressed")
        # Placeholder for the logic to plot the compressed image
        # based on the selected bins. You may want to clear and redraw
        # the compressed image here.
        pass

    def plot_DCT_coefficients(self):
        if verbose:
            print("got to plot_DCT_coefficients")

        # Placeholder for the logic to plot the DCT coefficients
        # based on the selected bins. You may want to clear and redraw
        # the DCT coefficients here.
        pass

    def plot_DCT_selected(self):
        if verbose:
            print("got to plot_DCT_selected")

        # Placeholder for the logic to plot the selected DCT coefficients
        # based on the selected bins. You may want to clear and redraw
        # the selected DCT coefficients here.
        pass

    def calculate_rmse(self):
        if verbose:
            print("got to calculate_rmse")

        # Placeholder for the logic to calculate the RMSE values
        # based on the selected bins. You may want to calculate the RMSE
        # values here.
        pass

    def calculate_compressed_image(self):
        if verbose:
            print("got to calculate_compressed_image")

        # Placeholder for the logic to calculate the compressed image
        # based on the selected bins. You may want to calculate the compressed
        # image here.
        pass

    def calculate_DCT_coefficients(self):
        if verbose:
            print("got to calculate_DCT_coefficients")

        # Placeholder for the logic to calculate the DCT coefficients
        # based on the selected bins. You may want to calculate the DCT
        # coefficients here.
        pass

    def calculate_all_bins_dct(self):
        if verbose:
            print("got to calculate_all_bins_dct")
        # Placeholder for the logic to calculate the DCT coefficients
        # for all bins. You may want to calculate the DCT coefficients
        # for all bins here.
        for i in range(64):
            self.bins[i] = self.dct2(img[i])
        pass

    def dct2(self, array):
        if verbose:
            print("got to dct2")
        """
        Compute the 2-D Discrete Cosine Transform.

        Parameters:
        - array: A 2-D numpy array.

        Returns:
        - A 2-D numpy array containing the DCT coefficients.
        """
        # Apply the DCT along the rows
        dct_temp = dct(array, axis=0, norm='ortho')
        # Apply the DCT along the columns
        dct_result = dct(dct_temp, axis=1, norm='ortho')
        print(dct_result)
        print(dct_result.shape)
        return dct_result

    def idct2(self, array):
        if verbose:
            print("got to idct2")
        """
        Compute the 2-D Inverse Discrete Cosine Transform.

        Parameters:
        - array: A 2-D numpy array of DCT coefficients.

        Returns:
        - A 2-D numpy array reconstructed from the DCT coefficients.
        """
        # Apply the IDCT (Inverse DCT) along the rows
        idct_temp = idct(array, axis=0, norm='ortho')
        # Apply the IDCT along the columns
        idct_result = idct(idct_temp, axis=1, norm='ortho')
        return idct_result

    def plot_masked_heatmap(self, number):
        if verbose:
            print("got to plot_masked_heatmap")
        temp = self.zigzag.copy()
        temp = np.where(temp > i, 0)
        print(temp)
        # Plot the array as a heatmap
        plt.figure(figsize=(8, 8))
        plt.imshow(temp, cmap='hot', interpolation='nearest')
        plt.title(f"Heatmap up to number {number}")
        plt.colorbar(label='Included in Mask')
        plt.show()



if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
