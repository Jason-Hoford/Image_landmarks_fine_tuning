import os
import yaml

def read_text_file(file_path):
    """
    Reads a evaluation file and returns its contents as a list of strings, split by commas.

    Parameters:
    - file_path (str): Path to the evaluation file.

    Returns:
    - list: List of strings containing the evaluation file data.
    """
    with open(file_path, 'r') as file:
        data = file.read().strip().split(',')
    return data

def write_text_file(file_path, data):
    """
    Writes a list of strings to a evaluation file, joining them with commas.

    Parameters:
    - file_path (str): Path to the evaluation file.
    - data (list): List of strings to write to the file.
    """
    with open(file_path, 'w') as file:
        file.write(",".join(data))

def create_directory(directory_path):
    """
    Creates a directory if it does not exist.

    Parameters:
    - directory_path (str): Path to the directory.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def load_images_and_coordinates(directory, text_file_path):
    """
    Loads image paths and coordinates from a evaluation file.

    Parameters:
    - directory (str): Path to the directory containing the images.
    - text_file_path (str): Path to the evaluation file containing the coordinates.

    Returns:
    - list: List of image paths.
    - list: List of corresponding coordinates.
    """
    image_paths = []
    coordinates = []
    data = read_text_file(text_file_path)

    for i in range(0, len(data), 9):
        img_name = data[i]
        coords = list(map(int, data[i + 1:i + 9]))
        coords = [i / 512 for i in coords]
        image_path = os.path.join(directory, img_name)
        if os.path.exists(image_path):
            image_paths.append(image_path)
            coordinates.append(coords)

    return list(image_paths), list(coordinates)

def load_config(config_path="config.yaml"):
    """
    Loads the configuration file.

    Parameters:
    - config_path (str): Path to the configuration file.

    Returns:
    - dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_predictions(prediction_file):
    """
    Loads prediction data from a text file.

    Parameters:
    - prediction_file (str): Path to the prediction file.

    Returns:
    - dict: Dictionary with image names as keys and coordinates as values.
    """
    with open(prediction_file, 'r') as file:
        data = file.read().strip().split(',')
    predictions = {}
    for i in range(0, len(data), 9):
        img_name = data[i]
        coords = list(map(int, data[i + 1:i + 9]))
        predictions[img_name] = coords
    return predictions

