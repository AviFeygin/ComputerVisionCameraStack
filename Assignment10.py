import sys  # Import the sys module for system-specific parameters and functions
import cv2  # Import the OpenCV library for image processing
import numpy as np  # Import NumPy for numerical operations
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout,
                             QPushButton, QFileDialog, QSlider, QWidget, QHBoxLayout,
                             QMessageBox)  # Import necessary PyQt5 widgets
from PyQt5.QtCore import Qt  # Import Qt core functionality
from PyQt5.QtGui import QImage, QPixmap  # Import QImage and QPixmap for image display
from matplotlib.figure import Figure  # Import Figure for creating Matplotlib figures
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas  # Import FigureCanvas for embedding Matplotlib in PyQt


class CLAHEApp(QMainWindow):  # Define the main application class inheriting from QMainWindow
    def __init__(self, verbose=False):  # Initialization method with an optional verbose parameter
        super().__init__()  # Call the superclass's initializer

        self.setWindowTitle('CLAHE Interactive Interface')  # Set the window title
        self.setGeometry(100, 100, 1200, 800)  # Set the window size and position

        self.image = None  # Initialize image attribute
        self.clip_limit = 2.0  # Set default clip limit for CLAHE
        self.grid_size = 8  # Set default grid size for CLAHE
        self.verbose = verbose  # Set verbose attribute

        self.initUI()  # Call method to initialize the UI

    def initUI(self):  # Method to initialize the UI components
        if self.verbose:  # Print method name if verbose is True
            print("got to initUI")

        layout = QVBoxLayout()  # Create a vertical box layout

        # Create and configure the open image button
        self.openButton = QPushButton('Open Image')
        self.openButton.clicked.connect(self.open_image)  # Connect button click to open_image method

        # Create and configure the clip limit slider
        self.clipLimitSlider = QSlider(Qt.Horizontal)
        self.clipLimitSlider.setMinimum(1)  # Set minimum slider value
        self.clipLimitSlider.setMaximum(40)  # Set maximum slider value
        self.clipLimitSlider.setValue(int(self.clip_limit * 10))  # Set initial slider value
        self.clipLimitSlider.valueChanged.connect(
            self.update_clahe)  # Connect slider value change to update_clahe method

        # Create and configure the grid size slider
        self.gridSizeSlider = QSlider(Qt.Horizontal)
        self.gridSizeSlider.setMinimum(1)  # Set minimum slider value
        self.gridSizeSlider.setMaximum(20)  # Set maximum slider value
        self.gridSizeSlider.setValue(self.grid_size)  # Set initial slider value
        self.gridSizeSlider.valueChanged.connect(
            self.update_clahe)  # Connect slider value change to update_clahe method

        # Add widgets to the layout
        layout.addWidget(self.openButton)
        layout.addWidget(QLabel('Clip Limit'))
        layout.addWidget(self.clipLimitSlider)
        layout.addWidget(QLabel('Grid Size'))
        layout.addWidget(self.gridSizeSlider)

        # Image display area
        imageLayout = QHBoxLayout()  # Create a horizontal box layout for images
        self.imageLabel = QLabel()  # Label for the original image
        self.processedLabel = QLabel()  # Label for the processed image
        imageLayout.addWidget(self.imageLabel)
        imageLayout.addWidget(self.processedLabel)
        layout.addLayout(imageLayout)

        # Histogram display area
        histLayout = QHBoxLayout()  # Create a horizontal box layout for histograms
        self.histFigureOriginal = Figure()  # Figure for the original histogram
        self.histCanvasOriginal = FigureCanvas(self.histFigureOriginal)  # Canvas for the original histogram
        self.histFigureProcessed = Figure()  # Figure for the processed histogram
        self.histCanvasProcessed = FigureCanvas(self.histFigureProcessed)  # Canvas for the processed histogram
        histLayout.addWidget(self.histCanvasOriginal)
        histLayout.addWidget(self.histCanvasProcessed)
        layout.addLayout(histLayout)

        centralWidget = QWidget()  # Create a central widget
        centralWidget.setLayout(layout)  # Set the central widget's layout
        self.setCentralWidget(centralWidget)  # Set the central widget

    def open_image(self):  # Method to open an image file
        if self.verbose:  # Print method name if verbose is True
            print("got to open_image")
        try:
            options = QFileDialog.Options()  # Create file dialog options
            filePath, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                      "Images (*.png *.xpm *.jpg *.jpeg *.bmp)",
                                                      options=options)  # Open file dialog
            if filePath:  # If a file is selected
                self.image = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
                if self.image is None:  # Check if image loading failed
                    raise ValueError("Failed to load the image.")  # Raise an error if image loading failed
                self.display_image()  # Display the original image
                self.display_histogram_original()  # Display the histogram of the original image
                self.apply_clahe()  # Apply CLAHE to the image
        except Exception as e:  # Catch any exceptions
            self.show_error_message(str(e))  # Show error message

    def display_image(self):  # Method to display the original image
        if self.verbose:  # Print method name if verbose is True
            print("got to display_image")
        try:
            if self.image is not None:  # Check if image is loaded
                height, width = self.image.shape  # Get image dimensions
                bytesPerLine = width  # Set bytes per line
                qImg = QImage(self.image.data, width, height, bytesPerLine,
                              QImage.Format_Grayscale8)  # Create QImage from the image data
                self.imageLabel.setPixmap(
                    QPixmap.fromImage(qImg).scaled(400, 400, Qt.KeepAspectRatio))  # Display the image in the label
        except Exception as e:  # Catch any exceptions
            self.show_error_message(str(e))  # Show error message

    def display_histogram_original(self):  # Method to display the histogram of the original image
        if self.verbose:  # Print method name if verbose is True
            print("got to display_histogram_original")
        try:
            if self.image is not None:  # Check if image is loaded
                self.histFigureOriginal.clear()  # Clear the previous histogram
                ax = self.histFigureOriginal.add_subplot(111)  # Add a subplot for the histogram
                ax.hist(self.image.ravel(), 256, [0, 256])  # Plot the histogram
                ax.set_title('Original Histogram')  # Set histogram title
                self.histCanvasOriginal.draw()  # Draw the histogram
        except Exception as e:  # Catch any exceptions
            self.show_error_message(str(e))  # Show error message

    def apply_clahe(self):  # Method to apply CLAHE to the image
        if self.verbose:  # Print method name if verbose is True
            print("got to apply_clahe")
        try:
            if self.image is not None:  # Check if image is loaded
                clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                        tileGridSize=(self.grid_size, self.grid_size))  # Create CLAHE object
                self.clahe_image = clahe.apply(self.image)  # Apply CLAHE to the image
                self.display_processed_image()  # Display the CLAHE-processed image
                self.display_histogram_processed()  # Display the histogram of the CLAHE-processed image
        except Exception as e:  # Catch any exceptions
            self.show_error_message(str(e))  # Show error message

    def display_processed_image(self):  # Method to display the CLAHE-processed image
        if self.verbose:  # Print method name if verbose is True
            print("got to display_processed_image")
        try:
            if self.clahe_image is not None:  # Check if CLAHE-processed image is available
                height, width = self.clahe_image.shape  # Get image dimensions
                bytesPerLine = width  # Set bytes per line
                qImg = QImage(self.clahe_image.data, width, height, bytesPerLine,
                              QImage.Format_Grayscale8)  # Create QImage from the image data
                self.processedLabel.setPixmap(QPixmap.fromImage(qImg).scaled(400, 400,
                                                                             Qt.KeepAspectRatio))  # Display the processed image in the label
        except Exception as e:  # Catch any exceptions
            self.show_error_message(str(e))  # Show error message

    def display_histogram_processed(self):  # Method to display the histogram of the CLAHE-processed image
        if self.verbose:  # Print method name if verbose is True
            print("got to display_histogram_processed")
        try:
            if self.clahe_image is not None:  # Check if CLAHE-processed image is available
                self.histFigureProcessed.clear()  # Clear the previous histogram
                ax = self.histFigureProcessed.add_subplot(111)  # Add a subplot for the histogram
                ax.hist(self.clahe_image.ravel(), 256, [0, 256])  # Plot the histogram
                ax.set_title('Processed Histogram')  # Set histogram title
                self.histCanvasProcessed.draw()  # Draw the histogram
        except Exception as e:  # Catch any exceptions
            self.show_error_message(str(e))  # Show error message

    def update_clahe(self):  # Method to update CLAHE parameters and reapply it
        if self.verbose:  # Print method name if verbose is True
            print("got to update_clahe")
        try:
            self.clip_limit = self.clipLimitSlider.value() / 10.0  # Update clip limit based on slider value
            self.grid_size = self.gridSizeSlider.value()  # Update grid size based on slider value
            self.apply_clahe()  # Reapply CLAHE with the updated parameters
        except Exception as e:  # Catch any exceptions
            self.show_error_message(str(e))  # Show error message

    def show_error_message(self, message):  # Method to show error messages
        msg = QMessageBox()  # Create a QMessageBox
        msg.setIcon(QMessageBox.Critical)  # Set the icon to critical
        msg.setText("An error occurred")  # Set the main text
        msg.setInformativeText(message)  # Set the informative text
        msg.setWindowTitle("Error")  # Set the window title
        msg.exec_()  # Display the message box


if __name__ == '__main__':  # Main entry point of the application
    app = QApplication(sys.argv)  # Create an application instance
    window = CLAHEApp(verbose=True)  # Create the main window with verbose mode enabled
    window.show()  # Show the main window
    sys.exit(app.exec_())  # Start the application event loop
