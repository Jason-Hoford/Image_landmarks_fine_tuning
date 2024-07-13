import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import models
import torch.nn as nn
from matplotlib.patches import ConnectionPatch
from utils import create_directory, write_text_file, load_config

config = load_config()
model_dir = config['directories']['model_dir']

results_dir = config['directories']['results_text_dir']
image_dir = config['directories']['images_dir']
MORE_DETAIN = False

class PointPredictionAndPlot:
    def __init__(self, model_path, model_name='resnet18'):
        self.model_path = model_path
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = self._load_model()

    def _load_model(self):
        if self.model_name == 'resnet18':
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 8)
        elif self.model_name == 'resnet50':
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 8)
        elif self.model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
        elif self.model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")

        model.load_state_dict(torch.load(self.model_path))
        return model.to(self.device)

    def predict_points(self, image_path):
        self.model.eval()
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            predictions = self.model(image)
        return predictions.cpu().numpy()[0].reshape(-1, 2) * 512

    def plot_points(self, image_path, points, output_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image)

        # Points A, B, C, D
        labels = ['A', 'B', 'C', 'D']
        for i, (x, y) in enumerate(points):
            ax.scatter(x, y, color='red')
            ax.text(x, y, labels[i], color='blue', fontsize=12, ha='right')

        if MORE_DETAIN:
            # Line a: A to B
            A, B, C, D = points
            ax.plot([A[0], B[0]], [A[1], B[1]], 'yellow', label='Line a')

            # Line a slope and intercept
            if A[0] != B[0]:  # To avoid division by zero
                slope_a = (B[1] - A[1]) / (B[0] - A[0])
                intercept_a = A[1] - slope_a * A[0]

                # Perpendicular line slope
                slope_b = -1 / slope_a
                intercept_b = D[1] - slope_b * D[0]

                # Point A* coordinates
                x_A_star = (intercept_b - intercept_a) / (slope_a - slope_b)
                y_A_star = slope_a * x_A_star + intercept_a
            else:
                x_A_star = A[0]
                y_A_star = D[1]

            A_star = np.array([x_A_star, y_A_star])

            # Plot point A*
            ax.scatter(A_star[0], A_star[1], color='purple', label='A*')
            ax.text(A_star[0], A_star[1], 'A*', color='purple', fontsize=12, ha='right')

            # Connect A* and D
            ax.plot([A_star[0], D[0]], [A_star[1], D[1]], 'green', label='Line b (perpendicular to line a)')
            ax.plot([A_star[0], C[0]], [A_star[1], C[1]], 'blue', label='Line A* - C')

            # Distances
            distance_A_star_C = np.linalg.norm(A_star - C)
            ax.text((A_star[0] + C[0]) / 2, (A_star[1] + C[1]) / 2 + 10, f'{distance_A_star_C:.2f}', color='blue',
                    fontsize=12)

            distance_A_star_D = np.linalg.norm(A_star - D)
            ax.text((A_star[0] + D[0]) / 2, (A_star[1] + D[1]) / 2 + 10, f'{distance_A_star_D:.2f}', color='green',
                    fontsize=12)

        # Save the plot
        plt.title(f'Predicted Points and Lines on Image: {os.path.basename(image_path)}')
        plt.axis('off')
        plt.legend()
        plt.savefig(output_path, dpi=300)
        plt.close()

    def predict_and_plot_folder(self, folder_path, coord_output_path, plot_output_dir):
        # create_directory(plot_output_dir)
        coord_data = []

        for img_name in os.listdir(folder_path):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, img_name)
                points = self.predict_points(image_path)

                # output_path = os.path.join(plot_output_dir, f'predicted_{img_name}')
                # self.plot_points(image_path, points, output_path)
                # print(f'Saved plotted image to {output_path}')

                # Prepare coordinate data in the specified format
                flat_points = points.flatten().astype(int)
                coord_str = ",".join(map(str, flat_points))
                coord_data.append(f"{img_name},{coord_str}")

        # Save coordinates to evaluation file
        write_text_file(coord_output_path, coord_data)
        print(f'Saved coordinates to {coord_output_path}')


# Example usage
if __name__ == "__main__":
    for model_name in ['resnet18', 'resnet50', 'mobilenet_v2', 'efficientnet_b0']:
        model_path = os.path.join(model_dir, f'point_detection_model_final_{model_name}.pth')

        folder_path = "data/evaluation" # The directory that contains the images that need predicting
        base_name = os.path.basename(os.path.normpath(folder_path))

        coord_output_path = os.path.join(results_dir, f'predictions_{model_name}.txt')
        plot_output_dir = os.path.join(image_dir, 'plots')

        predictor = PointPredictionAndPlot(model_path, model_name)
        predictor.predict_and_plot_folder(folder_path, coord_output_path, plot_output_dir)
