__copyright__ = """

    Copyright 2024 Jason Hoford

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""
__license__ = "Apache 2.0"

import os
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from utils import load_images_and_coordinates, load_config

config = load_config()
eval_data_dir = config['directories']['eval_data_dir']

def plot_perpendicular_lines(image_path, coords, ax, show_extra):
    """
    Plots points and perpendicular lines on the image.

    Parameters:
    - image_path (str): Path to the image file.
    - coords (list): List of coordinates for the points.
    - ax (matplotlib.axes.Axes): Matplotlib Axes object for plotting.
    - show_extra (bool): Whether to show extra details.
    """
    coords = np.array(coords).reshape(-1, 2)

    if coords[0][0] < 1:
        coords *= 512

    A, B, C, D = coords

    # Clear previous plots
    ax.clear()

    # Load the selected image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Plot the image
    ax.imshow(image)

    # Plot and label the points A, B, C, D
    labels = ['A', 'B', 'C', 'D']
    for i, (x, y) in enumerate(coords):
        ax.scatter(x, y, color='red')
        ax.text(x, y, labels[i], color='blue', fontsize=12, ha='right')

    if not show_extra:
        ax.set_title(f'Points on Image: {os.path.basename(image_path)}')
        ax.axis('off')
        return

    # Connect points A and B to form line a
    ax.plot([A[0], B[0]], [A[1], B[1]], 'yellow', label='Line a')

    # Extend line a for better visualization
    line_a_slope = (B[1] - A[1]) / (B[0] - A[0]) if A[0] != B[0] else None
    if line_a_slope is not None:
        x_vals = np.array(ax.get_xlim())
        y_vals = line_a_slope * (x_vals - A[0]) + A[1]
        ax.plot(x_vals, y_vals, 'yellow', linestyle='dotted')

    # Calculate the slope and intercept of line a
    if A[0] != B[0]:  # To avoid division by zero
        slope_a = (B[1] - A[1]) / (B[0] - A[0])
        intercept_a = A[1] - slope_a * A[0]

        # Calculate slope of the perpendicular line
        slope_b = -1 / slope_a

        # Equation of line a: y = slope_a * x + intercept_a
        # Equation of line b: y = slope_b * x + intercept_b
        # For the point A* on line a, we solve the system of equations:
        # slope_a * x_A* + intercept_a = slope_b * x_A* + intercept_b
        # intercept_b = D[1] - slope_b * D[0]

        intercept_b = D[1] - slope_b * D[0]

        # Calculate coordinates of A*
        x_A_star = (intercept_b - intercept_a) / (slope_a - slope_b)
        y_A_star = slope_a * x_A_star + intercept_a
    else:
        # If line a is vertical, then the perpendicular line will be horizontal through point D
        x_A_star = A[0]
        y_A_star = D[1]

    A_star = np.array([x_A_star, y_A_star])

    # Plot point A*
    ax.scatter(A_star[0], A_star[1], color='purple', label='A*')
    ax.text(A_star[0], A_star[1], 'A*', color='purple', fontsize=12, ha='right')

    # Connect A* and D to form line b (perpendicular to line a)
    ax.plot([A_star[0], D[0]], [A_star[1], D[1]], 'green', label='Line b (perpendicular to line a)')

    # Connect A* and C and show the distance
    ax.plot([A_star[0], C[0]], [A_star[1], C[1]], 'blue', label='Line A* - C')
    distance_A_star_C = np.linalg.norm(A_star - C)
    ax.text((A_star[0] + C[0]) / 2, (A_star[1] + C[1]) / 2, f'{distance_A_star_C:.2f}', color='blue', fontsize=12)

    # Calculate and show the distance between A* and D
    distance_A_star_D = np.linalg.norm(A_star - D)
    ax.text((A_star[0] + D[0]) / 2, (A_star[1] + D[1]) / 2 + 30, f'{distance_A_star_D:.2f}', color='green', fontsize=12)

    ax.legend()
    ax.set_title(f'Points and Perpendicular Lines on Image: {os.path.basename(image_path)}')
    ax.axis('off')

class ImageNavigator:
    def __init__(self, master, directory, text_file_path):
        self.master = master
        self.master.title("Image Navigator")
        self.directory = directory
        self.text_file_path = text_file_path
        self.image_paths, self.coordinates = load_images_and_coordinates(directory, text_file_path)
        self.image_index = 0
        self.show_extra = tk.BooleanVar(value=True)  # Initialize the show_extra flag

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.plot_current_image()

        self.prev_button = tk.Button(master, text="Previous", command=self.prev_image, font=("Helvetica", 16), height=2)
        self.prev_button.pack(side=tk.LEFT)

        self.next_button = tk.Button(master, text="Next", command=self.next_image, font=("Helvetica", 16), height=2)
        self.next_button.pack(side=tk.RIGHT)

        self.extra_checkbox = tk.Checkbutton(master, text="Show Extra Details", variable=self.show_extra, command=self.plot_current_image, font=("Helvetica", 16))
        self.extra_checkbox.pack(side=tk.BOTTOM)

    def plot_current_image(self):
        plot_perpendicular_lines(self.image_paths[self.image_index], self.coordinates[self.image_index], self.ax, self.show_extra.get())
        self.canvas.draw()

    def prev_image(self):
        if self.image_index > 0:
            self.image_index -= 1
            self.plot_current_image()

    def next_image(self):
        if self.image_index < len(self.image_paths) - 1:
            self.image_index += 1
            self.plot_current_image()


if __name__ == "__main__":
    directory = eval_data_dir
    text_file_path = os.path.join(directory, "cords.txt")

    root = tk.Tk()
    app = ImageNavigator(root, directory, text_file_path)
    root.mainloop()
