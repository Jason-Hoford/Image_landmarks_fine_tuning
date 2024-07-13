import os
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from utils import read_text_file, load_config

config = load_config()
results_dir = config['directories']['evaluation_text_dir']

def read_evaluation_data(file_path):
    """
    Reads the evaluation data from a evaluation file.

    Parameters:
    - file_path (str): Path to the evaluation results evaluation file.

    Returns:
    - list: List of pixel differences.
    - float: Mean pixel difference.
    """
    data = read_text_file(file_path)
    pixel_differences = [list(map(float, line.split(','))) for line in data[:-1]]
    mean_pixel_diff = float(data[-1].split(':')[-1])

    # Flatten the list of lists
    pixel_differences = [item for sublist in pixel_differences for item in sublist]

    return pixel_differences, mean_pixel_diff

def plot_pixel_difference_boxplot(results_dir, model_names):
    """
    Plots a boxplot and bar plot for pixel differences and mean pixel differences.

    Parameters:
    - results_dir (str): Directory containing the evaluation results files.
    - model_names (list): List of model names for which the evaluation results are available.
    """
    pixel_differences = {}
    mean_diffs = []

    for model_name in model_names:
        file_path = os.path.join(results_dir, f'evaluation_results_{model_name}.txt')
        differences, mean_diff = read_evaluation_data(file_path)
        pixel_differences[model_name] = differences
        mean_diffs.append(mean_diff)

    # Boxplot for pixel differences
    plt.figure(figsize=(12, 6))
    data = [pixel_differences[model_name] for model_name in model_names]
    plt.boxplot(data, labels=model_names, patch_artist=True)
    plt.xlabel('Models')
    plt.ylabel('Pixel Difference')
    plt.title('Pixel Difference Distribution for Different Pretrained Models')
    plt.show()

    # Bar plot for mean pixel differences
    plt.figure(figsize=(12, 6))
    plt.bar(model_names, mean_diffs, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('Mean Pixel Difference')
    plt.title('Mean Pixel Difference for Different Pretrained Models')
    plt.show()

def plot_pixel_differences_diagram(results_dir, model_names):
    """
    Plots a diagram with dots representing pixel differences and connects them with lines.

    Parameters:
    - results_dir (str): Directory containing the evaluation results files.
    - model_names (list): List of model names for which the evaluation results are available.
    """
    plt.figure(figsize=(12, 6))

    for model_name in model_names:
        file_path = os.path.join(results_dir, f'evaluation_results_{model_name}.txt')
        differences, _ = read_evaluation_data(file_path)
        x_values = np.arange(len(differences))
        plt.plot(x_values, differences, marker='o', label=model_name)

    plt.xlabel('Data Points')
    plt.ylabel('Pixel Difference')
    plt.title('Pixel Differences for Different Pretrained Models')
    plt.legend()
    plt.show()

def plot_mean_of_closest_n(results_dir, model_names, n):
    """
    Plots a diagram with dots representing the mean of the closest n pixel differences and connects them with lines.

    Parameters:
    - results_dir (str): Directory containing the evaluation results files.
    - model_names (list): List of model names for which the evaluation results are available.
    - n (int): Number of closest values to consider for mean calculation.
    """
    plt.figure(figsize=(12, 6))

    for model_name in model_names:
        file_path = os.path.join(results_dir, f'evaluation_results_{model_name}.txt')
        differences, _ = read_evaluation_data(file_path)
        means = [np.mean(differences[max(0, i-n):i+1]) for i in range(len(differences))]
        x_values = np.arange(len(differences))
        plt.plot(x_values, means, marker='o', label=model_name)

    plt.xlabel('Data Points')
    plt.ylabel(f'Mean of Closest {n} Pixel Differences')
    plt.title(f'Mean of Closest {n} Pixel Differences for Different Pretrained Models')
    plt.legend()
    plt.show()

def plot_with_slider(results_dir, model_names):
    """
    Plots a diagram with a slider to adjust the number of closest values for mean calculation in real-time.

    Parameters:
    - results_dir (str): Directory containing the evaluation results files.
    - model_names (list): List of model names for which the evaluation results are available.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.subplots_adjust(bottom=0.25)

    sliders = {}

    for model_name in model_names:
        file_path = os.path.join(results_dir, f'evaluation_results_{model_name}.txt')
        differences, _ = read_evaluation_data(file_path)
        sliders[model_name] = {
            'differences': differences,
            'line': ax.plot([], [], marker='o', label=model_name)[0]
        }

    ax.set_xlabel('Data Points')
    ax.set_ylabel('Pixel Difference')
    ax.set_title('Pixel Differences with Adjustable Mean Calculation')
    ax.legend()

    slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(slider_ax, 'Closest N', 1, 10, valinit=2, valstep=1)

    def update(val):
        n = int(slider.val)
        for model_name, data in sliders.items():
            differences = data['differences']
            means = [np.mean(differences[max(0, i-n):i+1]) for i in range(len(differences))]
            x_values = np.arange(len(differences))
            data['line'].set_data(x_values, means)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    slider.on_changed(update)
    update(2)  # Initial update with default value

    plt.xlim(0,20)
    plt.show()

if __name__ == "__main__":
    model_names = ['resnet18', 'resnet50', 'mobilenet_v2', 'efficientnet_b0']
    plot_pixel_difference_boxplot(results_dir, model_names)
    # plot_pixel_differences_diagram(results_dir, model_names)
    # plot_mean_of_closest_n(results_dir, model_names, 2)
    # plot_mean_of_closest_n(results_dir, model_names, 5)
    plot_with_slider(results_dir, model_names)
