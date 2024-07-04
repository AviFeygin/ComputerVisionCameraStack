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
        self.setGeometry(100, 100, 1600, 800)  # Adjusted for taller display

        self.imagePath = None
        self.lutPath = None
        self.currentIntensity = 'None'
        self.intensity_curve = None
        self.lut = None
        self.image = None
        self.image_lut = None
        self.final_image = None

        # Main layout
        layout = QVBoxLayout()

        # Image display
        self.figure = Figure(figsize=(16, 8))  # Adjusted for a larger figure
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
            self.intensity_curve = np.array([[1, 0, 0], [2, 255, 255]])

            cs = CubicSpline(x, y)

        elif seslf.currentIntensity == 'Light':
            self.intensity_curve = np.array([[1, 0, 0], [2, 78, 73], [3, 177, 182], [4, 255, 255]])
        elif self.currentIntensity == 'Medium':
            self.intensity_curve = np.array([[1, 0, 0], [2, 74, 56], [3, 164, 164], [4, 255, 255]])
        elif self.currentIntensity == 'Heavy':
            Strong = np.array([[1, 0, 0], [2, 75, 59], [3, 150, 150], [4, 190, 205], [5, 255, 255]])
        self.updatePlot()


    # TODO Implement this method
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

    def updatePlot(self):
        print("Update Plot called")
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

        # Other placeholders
        for i in range(4, 6):
            ax = self.figure.add_subplot(1, 5, i)
            ax.imshow(np.zeros(img.shape, dtype=np.uint8))  # Placeholder image
            ax.set_title(f'Effect {i-2}')
            ax.axis('off')

            self.canvas.draw()

def main():
    app = QApplication(sys.argv)
    main = LUTApplication()
    main.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
