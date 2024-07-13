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
import torch
import numpy as np
from point_detection_model import PointDetectionModel  # Ensure the main model script is named point_detection_model.py
from utils import load_images_and_coordinates, write_text_file, load_config

config = load_config()
model_dir = config['directories']['model_dir']
eval_data_dir = config['directories']['eval_data_dir']
results_dir = config['directories']['evaluation_text_dir']
eval_text_file_path = os.path.join(eval_data_dir, "cords.txt")

def evaluate_model(directory, text_file_path, model_name, model_weights_path, output_file_path):
    """
    Evaluates the model on the dataset and saves the results.

    Parameters:
    - directory (str): Path to the directory containing the images.
    - text_file_path (str): Path to the evaluation file containing the ground truth coordinates.
    - model_name (str): Name of the model to evaluate.
    - model_weights_path (str): Path to the model weights file.
    - output_file_path (str): Path to the output evaluation file for saving the evaluation results.
    """
    model = PointDetectionModel(directory, text_file_path, model_name=model_name)
    model.model.load_state_dict(torch.load(model_weights_path))
    model.model.eval()

    image_paths, ground_truth_coords = load_images_and_coordinates(directory, text_file_path)

    pixel_differences = []
    for image_path, true_coords in zip(image_paths, ground_truth_coords):
        pred_coords = model.predict_points(image_path) * 512
        pred_coords = pred_coords.astype(int)
        true_coords = np.array(true_coords) * 512
        diff = np.abs(pred_coords - true_coords)
        pixel_differences.append(diff)

    mean_pixel_diff = np.mean(pixel_differences)
    result_data = [",".join(map(str, diff)) for diff in pixel_differences]
    result_data.append(f"Mean Pixel Difference: {mean_pixel_diff}")
    write_text_file(output_file_path, result_data)

    print(f"Mean Pixel Difference for {model_name}: {mean_pixel_diff}")

if __name__ == "__main__":
    model_names = ['resnet18', 'resnet50', 'mobilenet_v2', 'efficientnet_b0']
    for model_name in model_names:
        model_weights_path = os.path.join(model_dir, f'point_detection_model_best_{model_name}.pth')
        output_file_path = os.path.join(results_dir, f'evaluation_results_{model_name}.txt')
        evaluate_model(eval_data_dir, eval_text_file_path, model_name, model_weights_path, output_file_path)