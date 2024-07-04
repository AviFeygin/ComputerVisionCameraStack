import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Import the 3D plotting toolkit

class LUTApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LUT Application")
        self.setGeometry(100, 100, 1600, 600)  # Adjusted for a proportionate display

        self.imagePath = None
        self.lutPath = None
        self.currentIntensity = 'None'

        # Main layout
        layout = QVBoxLayout()

        # Image display
        self.figure = Figure(figsize=(16, 6))  # Adjusted for the new layout
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
        self.btnApplyLUT = QPushButton('Apply LUT')
        self.btnApplyLUT.clicked.connect(self.applyLUT)
        layout.addWidget(self.btnApplyLUT)

        # Set the layout to the central widget
        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)

    def selectImage(self):
        self.imagePath, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        self.updatePlot()

    def selectLUT(self):
        self.lutPath, _ = QFileDialog.getOpenFileName(self, "Select LUT File", "", "LUT Files (*.cube)")
        self.updatePlot()

    def selectIntensity(self, index):
        self.currentIntensity = self.intensityDropdown.itemText(index)
        self.updatePlot()

    def applyLUT(self):
        self.updatePlot(process=True)

    def updatePlot(self, process=False):
        self.figure.clear()

        if self.imagePath:
            img = cv2.imread(self.imagePath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Original Image
            ax_img = self.figure.add_subplot(1, 5, 1)
            ax_img.imshow(img)
            ax_img.set_title('Original')
            ax_img.axis('off')

            # Placeholder for 3D LUT visualization
            ax_3d = self.figure.add_subplot(1, 5, 2, projection='3d')
            # For demonstration purposes, using random data
            ax_3d.scatter(np.random.rand(10), np.random.rand(10), np.random.rand(10))
            ax_3d.set_title('3D LUT Visualization')

            # Adjusted loop for remaining placeholders
            for i in range(3, 6):
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