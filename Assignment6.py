'''Author:Ariel Feygin, 214527790'''

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sea
from matplotlib.widgets import Button
import cv2
import rawpy
from matplotlib.animation import FuncAnimation
import time

mpl.use('Qt5Agg')
image_stack_path = 'Stack2.npy'

class SobelEnergyCalculator:
    def __init__(self, image_stack_path):
        self.x = None
        self.y = None

        # load the image stack path
        self.image_stack = np.load(image_stack_path)
        print(self.image_stack.shape)
        self.num_images, self.height, self.width, _ = self.image_stack.shape

        # compute the global energies
        self.global_energies = self.compute_sobel_energy(self.image_stack, interpretation=1)
        # find the index of the highest energy/sharpness
        self.maximum_sharpness_global_index = np.argmax(self.global_energies)

        #create arrays to hold bounding box energies
        self.selected_energies = np.ones(len(self.global_energies))
        self.maximum_sharpness_selected_index = 0
        self.bounding_box_images_stack = None


        # Create a figure and axes for displaying the image and the Sobel energy plot
        self.fig, self.ax = plt.subplots(1, 2, figsize=(10, 5))
        self.ax[0].set_title('Image')  # Set title for the image display
        self.ax[1].set_title('Sobel Energy')  # Set title for the Sobel energy plot
        plt.subplots_adjust(bottom=0.2)  # Adjust the position of the subplots to make room for buttons

        # Create "Next" and "Previous" buttons
        self.next_button = Button(plt.axes([0.3, 0.05, 0.1, 0.075]), 'Next')
        self.prev_button = Button(plt.axes([0.2, 0.05, 0.1, 0.075]), 'Previous')
        self.global_button = Button(plt.axes([0.5, 0.05, 0.1, 0.075]), 'Global')

        # Attach event handlers to the buttons
        self.next_button.on_clicked(self.next_image)
        self.prev_button.on_clicked(self.prev_image)
        self.global_button.on_clicked(self.global_reset)
        self.current_index = 0  # Initialize the index of the currently displayed image

        # Connect the event handler for mouse clicks on the image
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.bbox = None  # Initialize bounding box to None
        #flag for selected mode or global
        self.mode = "global"

    def get_index_values(self):
        if self.mode == "global":
            index_before = self.global_energies[(self.current_index - 1) % len(self.image_stack)]
            index_after = self.global_energies[(self.current_index + 1) % len(self.image_stack)]
            index_before_before = self.global_energies[(self.current_index - 2) % len(self.image_stack)]
            index_after_after = self.global_energies[(self.current_index + 2) % len(self.image_stack)]
            index_value = (self.global_energies[self.current_index])
            values = np.array([index_before_before, index_before, index_value, index_after, index_after_after])
        elif self.mode == "selected":
            index_before = self.selected_energies[(self.current_index - 1) % len(self.image_stack)]
            index_after = self.selected_energies[(self.current_index + 1) % len(self.image_stack)]
            index_before_before = self.selected_energies[(self.current_index - 2) % len(self.image_stack)]
            index_after_after = self.selected_energies[(self.current_index + 2) % len(self.image_stack)]
            index_value = (self.selected_energies[self.current_index])
            values = np.array([index_before_before, index_before, index_value, index_after, index_after_after])
        print(values)
        return values

    def driving_loop(self):
        print("Driving loop")
        at_min = False
        min_index = np.argmax(self.global_energies)
        min_value = np.min(self.global_energies)
        values = self.get_index_values()
        # while loop to find minima, perceptive field two steps away to avoid local minima
        while(np.argmax(values) != 2):
            print("got here")
            if np.argmax(values) == 2:
                at_min = True
            else:
                if np.argmax(values) >2:
                    print("got here 2")
                    self.next_image_()
                elif np.argmax(values) <2:
                    print("got here 3")
                    self.prev_image_()
                    print(self.current_index)
            values = self.get_index_values()
            mpl.pyplot.pause(.5)

    def _sobel_operator(self, image):
        # Define Sobel kernels for horizontal and vertical gradients
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        # Convolve image with Sobel kernels for each channel
        gradient_x = np.zeros_like(image)
        gradient_y = np.zeros_like(image)
        for i in range(image.shape[-1]):
            gradient_x[:, :, i] = np.abs(
                np.convolve(image[:, :, i].flatten(), kernel_x.flatten(), mode='same').reshape(image[:, :, i].shape))
            gradient_y[:, :, i] = np.abs(
                np.convolve(image[:, :, i].flatten(), kernel_y.flatten(), mode='same').reshape(image[:, :, i].shape))
        return gradient_x, gradient_y

    def compute_sobel_energy(self, image_stack, interpretation=1):
        energies = []
        for img in image_stack:
            # Calculate gradients using Sobel operator
            gradient_x, gradient_y = self._sobel_operator(img)
            # Compute gradient magnitude
            gradient_magnitude = np.sqrt(np.sum(gradient_x ** 2, axis=-1) + np.sum(gradient_y ** 2, axis=-1))
            # Compute Sobel energy based on interpretation
            if interpretation == 1:
                energy = np.sum(gradient_magnitude)  # Sum of gradient magnitudes
            elif interpretation == 2:
                energy = np.sum(gradient_magnitude ** 2)  # Sum of squared gradient magnitudes
            else:
                raise ValueError("Invalid interpretation value. Choose either 1 or 2.")
            energies.append(energy)
        return energies

    def plot_sobel_energies(self, mode):
        if mode == "global":
            # Display the current image
            self.image_display = self.ax[0].imshow(self.image_stack[self.current_index])
            # Plot Sobel energies
            self.line_plot, = self.ax[1].plot(range(len(self.global_energies)), self.global_energies, marker='o')
            # Draw a red vertical line indicating the current image index
            self.red_line = self.ax[1].axvline(x=self.current_index, color='red', linestyle='--')
            # Set labels and title for the Sobel energy plot
            self.ax[1].set_xlabel('Image Index')
            self.ax[1].set_ylabel('Sobel Energy')
            self.ax[1].set_title('Sobel Energy of Image Stack')
            self.ax[1].grid(True)  # Show grid in the plot
            self.update_plot(mode=mode)  # Update the plot with current data
            plt.show()  # Display the plot
        elif mode == "selected":
            # Display the current image
            self.image_display = self.ax[0].imshow(self.image_stack[self.current_index])
            # Plot Sobel energies
            self.line_plot, = self.ax[1].plot(range(len(self.selected_energies)), self.selected_energies, marker='o')
            # Draw a red vertical line indicating the current image index
            self.red_line = self.ax[1].axvline(x=self.current_index, color='red', linestyle='--')
            # Set labels and title for the Sobel energy plot
            self.ax[1].set_xlabel('Image Index')
            self.ax[1].set_ylabel('Sobel Energy')
            self.ax[1].set_title('Sobel Energy of Selected Location Image Stack')
            self.ax[1].grid(True)  # Show grid in the plot
            # Draw a yellow bounding box centered at the clicked coordinates

            self.update_plot(mode=mode)  # Update the plot with current data
            plt.show()  # Display the plot


    def update_plot(self, mode):
        # Update image display with the current image
        self.image_display.set_data(self.image_stack[self.current_index])

        # Update the position of the red vertical line to indicate the current image index
        self.red_line.set_xdata(self.current_index)
        if mode == "global":
            # Update Sobel energy plot with the current energies
            self.line_plot.set_ydata(self.global_energies)

        elif mode == "selected":
            # Update Sobel energy plot with the current energies
            self.line_plot.set_ydata(self.selected_energies)
            if self.bbox:
                self.bbox.remove()
            bbox_x = max(0, self.x - 50)  # Calculate top-left x-coordinate of the bounding box
            bbox_y = max(0, self.y - 50)  # Calculate top-left y-coordinate of the bounding box

            self.bbox = plt.Rectangle((bbox_x, bbox_y), 100, 100, edgecolor='yellow', linewidth=2, fill=False)
            self.ax[0].add_patch(self.bbox)  # Add the bounding box to the image display
        # Update image display with the current image
        self.image_display.set_data(self.image_stack[self.current_index])
        # Update the position of the red vertical line to indicate the current image index
        self.red_line.set_xdata(self.current_index)

        # Draw the updated plot
        self.fig.canvas.draw()

    def next_image(self, event):
        # Move to the next image in the stack
        self.current_index = (self.current_index + 1) % len(self.image_stack)
        # Update the plot with the new image
        self.update_plot(mode=self.mode)

    def next_image_(self):
        # Move to the next image in the stack
        self.current_index = (self.current_index + 1) % len(self.image_stack)
        # Update the plot with the new image
        self.update_plot(mode=self.mode)

    def prev_image(self, event):
        # Move to the previous image in the stack
        self.current_index = (self.current_index - 1) % len(self.image_stack)
        # Update the plot with the new image
        self.update_plot(mode=self.mode)

    def prev_image_(self):
        # Move to the previous image in the stack
        self.current_index = (self.current_index - 1) % len(self.image_stack)
        # Update the plot with the new image
        self.update_plot(mode=self.mode)

    def global_reset(self, event):
        self.mode = "global"
        self.ax[1].clear()
        self.ax[0].clear()
        self.ax[0].set_title('Image')  # Set title for the image display
        self.image_display = self.ax[0].imshow(self.image_stack[self.current_index])
        self.bbox = None
        sobel_calculator.plot_sobel_energies(mode=self.mode)
        self.driving_loop()

    def on_click(self, event):
        # Event handler for mouse clicks on the image
        if event.inaxes == self.ax[0]:  # Check if the click occurred on the image axis
            self.ax[1].clear()
            self.current_index = 0
            self.x, self.y = int(event.xdata), int(event.ydata)  # Get the coordinates of the click
            self.mode = "selected"
            # Remove the existing bounding box, if any
            if self.bbox:
                self.bbox.remove()
            # Draw a yellow bounding box centered at the clicked coordinates
            bbox_x = max(0, self.x - 50)  # Calculate top-left x-coordinate of the bounding box
            bbox_y = max(0, self.y - 50)  # Calculate top-left y-coordinate of the bounding box


            self.bbox = plt.Rectangle((bbox_x, bbox_y), 100, 100, edgecolor='yellow', linewidth=2, fill=False)
            self.ax[0].add_patch(self.bbox)  # Add the bounding box to the image display

            # TODO:  # Compute the Sobel energies for the bounding box stack fix it
            #get the bounding box stack taking into account the edges. x:min 0 max width, y min 0 max height
            # Define the boundaries of the bounding box, ensuring they stay within image dimensions
            y_start = max(self.y - 49, 0)
            y_end = min(self.y + 50, self.height)
            x_start = max(self.x - 49, 0)
            x_end = min(self.x + 50, self.width)

            # Extract the bounding box from the image stack
            self.bounding_box_stack = self.image_stack[:, y_start:y_end, x_start:x_end, :].copy()

            self.selected_energies = self.compute_sobel_energy(self.bounding_box_stack,2)
            self.plot_sobel_energies(mode="selected")
            self.fig.canvas.draw()  # Redraw the figure to show the bounding box
            self.driving_loop()
sobel_calculator = SobelEnergyCalculator(image_stack_path)
# Plot Sobel energies
sobel_calculator.plot_sobel_energies(mode="global")





