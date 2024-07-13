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
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from utils import load_config, load_predictions

config = load_config()
image_folder = config['directories']['eval_data_dir']
results_dir = config['directories']['results_text_dir']
original_data_file = os.path.join(config['directories']['eval_data_dir'], 'cords.txt')

class ImageNavigator:
    def __init__(self, root, image_folder, predictions_dict):
        self.root = root
        self.image_folder = image_folder
        self.predictions_dict = predictions_dict
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        self.current_index = 0

        self.figure, self.ax = plt.subplots(figsize=(10, 10))  # Set initial figure size
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=2, sticky="nsew")

        self.button_prev = tk.Button(root, text="Previous", command=self.show_previous_image)
        self.button_prev.grid(row=1, column=0, sticky="ew")

        self.button_next = tk.Button(root, text="Next", command=self.show_next_image)
        self.button_next.grid(row=1, column=1, sticky="ew")

        root.grid_rowconfigure(0, weight=1)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)

        self.show_image(self.image_files[self.current_index])

    def show_image(self, img_name):
        img_path = os.path.join(self.image_folder, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.ax.clear()
        self.ax.imshow(image)

        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

        for i, (model_name, predictions) in enumerate(self.predictions_dict.items()):
            if img_name in predictions:
                coords = predictions[img_name]
                x_coords = coords[0::2]
                y_coords = coords[1::2]
                marker = 'x' if model_name == 'human' else 'o'
                self.ax.scatter(x_coords, y_coords, c=colors[i], label=model_name, alpha=1 if model_name == 'human' else 0.5, marker=marker)

        self.ax.legend()
        self.ax.set_title(img_name)
        self.canvas.draw()

    def show_next_image(self):
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.show_image(self.image_files[self.current_index])

    def show_previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.show_image(self.image_files[self.current_index])

if __name__ == "__main__":
    # Load predictions from different models
    predictions_resnet18 = load_predictions(os.path.join(results_dir, 'predictions_resnet18.txt'))
    predictions_resnet50 = load_predictions(os.path.join(results_dir, 'predictions_resnet50.txt'))
    predictions_mobilenet_v2 = load_predictions(os.path.join(results_dir, 'predictions_mobilenet_v2.txt'))
    predictions_efficientnet_b0 = load_predictions(os.path.join(results_dir, 'predictions_efficientnet_b0.txt'))
    original = load_predictions(original_data_file)

    # Combine predictions into a dictionary
    predictions_dict = {
        'resnet18': predictions_resnet18,
        'resnet50': predictions_resnet50,
        'mobilenet_v2': predictions_mobilenet_v2,
        'efficientnet_b0': predictions_efficientnet_b0,
        'human': original
    }

    # Create the main window
    root = tk.Tk()
    root.title("Image Navigator")
    root.geometry("800x800")

    # Initialize the ImageNavigator
    navigator = ImageNavigator(root, image_folder, predictions_dict)

    # Run the Tkinter event loop
    root.mainloop()
