import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog, QComboBox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting toolkit
import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from PIL import Image
import pillow_lut


class LUTApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LUT Application")
        self.setGeometry(100, 100, 2400, 1500)  # Adjusted for taller display

        self.imagePath = None
        self.lutPath = None
        self.currentIntensity = 'None'
        self.intensity_curve = None
        self.intensity_curve_array = None
        self.lut = None
        self.image = None
        self.image_lut = None
        self.final_image = None

        # Main layout
        layout = QVBoxLayout()

        # Image display
        self.figure = Figure(figsize=(32, 16))  # Adjusted for a larger figure
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Button for selecting the image file
        self.btnSelectImage = QPushButton('Select Image')
        self.btnSelectImage.clicked.connect(self.selectImage)
        layout.addWidget(self.btnSelectImage)

        # Button for selecting the LUT file
        self.btnSelectLUT = QPushButton('Select LUT File')
        self.btnSelectLUT.clicked.connect(self.selectLUT)
        layout.addWidget(self.btnSelectLUT)

        # Dropdown for selecting intensity
        self.intensityDropdown = QComboBox()
        self.intensityDropdown.addItems(['None', 'Light', 'Medium', 'Heavy'])
        self.intensityDropdown.currentIndexChanged.connect(self.selectIntensity)
        layout.addWidget(self.intensityDropdown)

        # Button to apply LUT
        self.btnApplyLUT = QPushButton('Apply Transformation')
        self.btnApplyLUT.clicked.connect(self.applyLUT)
        layout.addWidget(self.btnApplyLUT)

        # Set the layout to the central widget
        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

        #precalculated intensity curves
        self.no_intensity = np.array([[0, 0], [255, 255]])
        self.no_intensity_spline = CubicSpline(self.no_intensity[:, 0], self.no_intensity[:, 1])
        self.light = np.array([[0, 0], [78, 73], [177, 182], [255, 255]])
        self.light_intensity_spline = CubicSpline(self.light[:, 0], self.light[:, 1])
        self.medium = np.array([[0, 0], [74, 56], [164, 164], [255, 255]])
        self.medium_intensity_spline = CubicSpline(self.medium[:, 0], self.medium[:, 1])
        self.strong = np.array([[0, 0], [75, 59], [150, 150], [190, 205], [255, 255]])
        self.strong_intensity_spline = CubicSpline(self.strong[:, 0], self.strong[:, 1])
        self.intensity_curve_array = self.no_intensity
        self.intensity_curve = self.no_intensity_spline
        self.plot_cubic_spline(self.currentIntensity)

    # this is bad coding should be able to plot any spline redo if time TODO
    def plot_cubic_spline(self, intensity):
        print("Plot Cubic Spline called")
        self.currentIntensity = intensity

        if intensity == 'None':
            cs = self.no_intensity_spline
            self.intensity_curve_array = self.no_intensity
            x_values = self.no_intensity[:, 0]
            y_values = self.no_intensity[:, 1]
        elif intensity == 'Light':
            cs = self.light_intensity_spline
            self.intensity_curve_array = self.light
            x_values = self.light[:, 0]
            y_values = self.light[:, 1]
        elif intensity == 'Medium':
            cs = self.medium_intensity_spline
            self.intensity_curve_array = self.medium
            x_values = self.medium[:, 0]
            y_values = self.medium[:, 1]
        elif intensity == 'Heavy':
            cs = self.strong_intensity_spline
            self.intensity_curve_array = self.strong
            x_values = self.strong[:, 0]
            y_values = self.strong[:, 1]
        else:
            print("Invalid intensity")
            return

        # Generate a dense set of x values for a smooth curve
        x_dense = np.linspace(0, 255, 255)
        y_dense = cs(x_dense)
        # Plotting
        ax = self.figure.add_subplot(1, 5, 4)
        ax.plot(x_values, y_values, 'o', label='Data points')
        ax.plot(x_dense, y_dense, label='Cubic spline')
        ax.set_title('Tone Curve')
        ax.grid(True)
        self.canvas.draw()

    def ComputeSplines(self, points):
        print("Compute Splines called")
        x = self.intensity_curve[:,1]
        y = self.intensity_curve[:,2]
        return np.polyfit(x, y, len(points) - 1)

    def selectImage(self):
        print("Select Image called")
        self.imagePath, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        self.image = Image.open(self.imagePath).copy()
        self.updatePlot()

    def selectLUT(self):
        print("Select LUT called")
        self.lutPath, _ = QFileDialog.getOpenFileName(self, "Select LUT File", "", "LUT Files (*.cube)")
        self.lut = pillow_lut.load_cube_file(self.lutPath)
        # print("got to here")
        # print(self.lut)
        self.update_lut_plot()

    def selectIntensity(self, index):
        print("Select Intensity called")
        self.currentIntensity = self.intensityDropdown.itemText(index)
        if self.currentIntensity == 'None':
            self.intensity_curve = self.no_intensity_spline
        elif self.currentIntensity == 'Light':
            self.intensity_curve = self.light_intensity_spline
        elif self.currentIntensity == 'Medium':
            self.intensity_curve = self.medium_intensity_spline
        elif self.currentIntensity == 'Heavy':
            self.intensity_curve = self.strong_intensity_spline
        try:
            self.plot_cubic_spline(self.currentIntensity)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")

    def applyLUT(self):
        print("Apply Lut called")
        if self.lut == None:
            print("LUT not selected")
            return
        if self.image == None:
            print("Image not selected")
            return
        else:
            print('3rd case')
        #     lut = self.lut
        #     im = self.image
            try:
                self.image_lut = Image.open(self.imagePath).filter(self.lut)
            except Exception as e:
                print("Error applying LUT:", e)
                return
            ax = self.figure.add_subplot(1, 5, 3)
            print("got to here3")
            try:
                ax.imshow(self.image_lut)
            except Exception as e:
                print("Error displaying image:", e)
                return
            ax.set_title("Lut Applied")
            ax.axis('off')
            print('got to plotting')
            self.applyToneCurve()
            self.plot_final_image()
            self.canvas.draw()

    def update_lut_plot(self):
        print("Update Lut Plot called")

        if self.lut == None:
            # Placeholder for 3D LUT visualization
            ax_3d = self.figure.add_subplot(1, 5, 2, projection='3d')
            # Here you would add your 3D LUT visualization logic
            # This is just a placeholder for demonstration
            ax_3d.scatter(np.zeros(10), np.zeros(10), np.zeros(10))
            ax_3d.set_title('3D LUT Visualization')
            # Assuming `arr` is your 32x32x32x3 numpy array

        elif self.lut != None:
            # lut = self.lut
            print(32 * 32 * 32)
            # print(lut.table)
            lut_table = self.lut.table
            arr = np.array(lut_table)
            # print("got to here")
            # Reshape the array to flatten the first three dimensions
            points = arr.reshape(-1, 3)  # Now it's (32768, 3)

            # The color is the last dimension
            colors = points  # In this example, the colors are the points themselves

            # Separate the points for plotting
            # x, y, z = points[:, 0], points[:, 1], points[:, 2]
            grid = np.mgrid[-0:32:1, 0:32:1, 0:32:1]

            # print(x)
            # Create a 3D scatter plot
            ax_3d = self.figure.add_subplot(1, 5, 2, projection='3d')
            ax_3d.scatter(grid[0], grid[1], grid[2], c=colors)
            ax_3d.set_title('3D LUT Visualization')
            plt.show()
            self.canvas.draw()

    def applyToneCurve(self):
        print("Apply Tone Curve called")
        # Generate the cubic spline
        try:
            cs = self.intensity_curve
            print(self.image_lut)
            img = self.image_lut
            img = img.convert('RGB')  # Ensure the image is in RGB mode

            # Convert the image to a numpy array for vectorized operations
            img_array = np.array(img)

            for channel in range(3):  # Loop through the R, G, B channels
                img_array[..., channel] = cs(img_array[..., channel])

            # Clip values to the valid range (0-255) and convert back to uint8
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)

            # Convert the numpy array back to a PIL image
            img_modified = Image.fromarray(img_array)
            self.final_image = img_modified.copy()

            plot_final_image(self)
        except Exception as err:
            print(f"Unexpected {err=}, {type(err)=}")

    def plot_final_image(self):
        if self.final_image is None:
            return
        else:
            print("Plot Final Image called")
            ax = self.figure.add_subplot(1, 5, 5)
            ax.imshow(self.final_image)
            ax.set_title('Final Image')
            ax.axis('off')
            self.canvas.draw()

    def updatePlot(self):
        print("Update Plot called")
        try:
            self.figure.clear()

            # Original Image
            if self.imagePath:
                img = cv2.imread(self.imagePath)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                ax_img = self.figure.add_subplot(1, 5, 1)
                ax_img.imshow(img)
                ax_img.set_title('Original')
                ax_img.axis('off')

            self.update_lut_plot()
            self.applyLUT()
            self.plot_cubic_spline(self.currentIntensity)
            self.plot_final_image(self)
            self.canvas.draw()
        except Exception as e:
            print("Error updating plot:", e)
def main():
    app = QApplication(sys.argv)
    main = LUTApplication()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
