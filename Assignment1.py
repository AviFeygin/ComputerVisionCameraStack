import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import rawpy
import numpy as np

# create the image viewer class
class RawImageViewer:
    # initialize the class
    def __init__(self, master):
        #define window "master"
        self.master = master
        # set title
        self.master.title("Raw Image Viewer")
        # define an image to be loaded
        self.raw_image = None
        # numpy image
        self.raw_image_numpy = None
        # define the exposure slider
        self.exposure_slider = None
        # set the axis and figure of the slider
        self.fig, self.ax = Figure(), None
        #embedding the matplolib figure into the canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack()
        # load the image loading button
        self.load_button = tk.Button(self.master, text="Load Raw Image", command=self.load_raw_image)
        self.load_button.pack(pady=10)

    def load_raw_image(self):
        # get the file path
        file_path = filedialog.askopenfilename()
        # if valid imread it
        if file_path:
            self.raw_image = rawpy.imread(file_path)
            self.raw_image_numpy = self.raw_image.raw_image
            # call create slider
            self.create_slider()
            # display the image
            self.display_image(self.raw_image_numpy)

    def create_slider(self):
        if self.exposure_slider is None:
            self.exposure_slider = tk.Scale(self.master, label="Exposure", from_=0, to=10, resolution=0.5, orient=tk.HORIZONTAL, command=self.update_exposure)
            self.exposure_slider.set(1.0)
            self.exposure_slider.pack(pady=10)

    def update_exposure(self, value):
        if self.raw_image:
            exposure = float(value)
            temp = np.multiply(self.raw_image_numpy,exposure)
            print(temp)
            self.display_image(self,temp)

    def display_image(self, image):
        self.ax.imshow(image)
        self.canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    app = RawImageViewer(root)
    root.mainloop()
