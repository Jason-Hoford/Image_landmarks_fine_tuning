import os
import numpy as np
from utils import load_config, read_text_file

def calculate_distances(coords):
    """
    Calculate distances between A* to D and A* to C.

    Parameters:
    - coords (list): List of coordinates for the points.

    Returns:
    - tuple: Distances A* to D and A* to C.
    """
    coords = np.array(coords).reshape(-1, 2)
    A, B, C, D = coords

    # Calculate the slope and intercept of line a
    if A[0] != B[0]:  # To avoid division by zero
        slope_a = (B[1] - A[1]) / (B[0] - A[0])
        intercept_a = A[1] - slope_a * A[0]

        # Calculate slope of the perpendicular line
        slope_b = -1 / slope_a
        intercept_b = D[1] - slope_b * D[0]

        # Calculate coordinates of A*
        x_A_star = (intercept_b - intercept_a) / (slope_a - slope_b)
        y_A_star = slope_a * x_A_star + intercept_a
    else:
        # If line a is vertical, then the perpendicular line will be horizontal through point D
        x_A_star = A[0]
        y_A_star = D[1]

    A_star = np.array([x_A_star, y_A_star])

    # Calculate distances
    distance_A_star_C = np.linalg.norm(A_star - C)
    distance_A_star_D = np.linalg.norm(A_star - D)

    return distance_A_star_C, distance_A_star_D

def get_distances_from_file(file_path):
    """
    Get distances A* to D and A* to C from a text file.

    Parameters:
    - file_path (str): Path to the text file containing coordinates.

    Returns:
    - list: List of distances tuples (A* to D, A* to C) for each image.
    """
    data = read_text_file(file_path)
    distances = []

    for i in range(0, len(data), 9):
        img_name = data[i]
        coords = list(map(int, data[i + 1:i + 9]))
        distance_A_star_C, distance_A_star_D = calculate_distances(coords)
        distances.append((img_name, distance_A_star_C, distance_A_star_D))

    return distances

if __name__ == "__main__":

    text_file_path = "data/evaluation/cords.txt"

    distances = get_distances_from_file(text_file_path)

    for img_name, distance_A_star_C, distance_A_star_D in distances:
        print(f"Image: {img_name}, Distance A* to C: {distance_A_star_C:.2f}, Distance A* to D: {distance_A_star_D:.2f}")
