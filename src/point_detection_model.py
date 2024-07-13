import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import matplotlib.pyplot as plt
from utils import read_text_file, create_directory, write_text_file, load_images_and_coordinates, load_config

config = load_config()

data_dir = config['directories']['data_dir']
model_dir = config['directories']['model_dir']
results_dir = config['directories']['results_dir']

class PointDetectionModel:
    def __init__(self, directory, text_file_path, model_name='resnet18', batch_size=64, num_epochs=500,
                 learning_rate=0.001, patience=300):
        self.directory = directory
        self.text_file_path = text_file_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.patience = patience

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.model = self._load_pretrained_model()
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=10, factor=0.5)
        self.writer = SummaryWriter(log_dir=f'runs/{self.model_name}')

        self.image_paths, self.coordinates = load_images_and_coordinates(self.directory, self.text_file_path)
        self.train_loader, self.val_loader = self._create_data_loaders()

        self.model_dir = model_dir
        self.image_dir = os.path.join(results_dir, 'images')
        create_directory(self.model_dir)
        create_directory(self.image_dir)

    def _create_data_loaders(self):
        """
        Creates training and validation data loaders.

        Returns:
        - DataLoader: Training data loader.
        - DataLoader: Validation data loader.
        """
        train_paths, val_paths, train_coords, val_coords = train_test_split(
            self.image_paths, self.coordinates, test_size=0.2, random_state=42
        )

        train_dataset = ImageDataset(train_paths, train_coords, transform=self.transform, augment=False)
        val_dataset = ImageDataset(val_paths, val_coords, transform=self.transform, augment=False)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader

    def _load_pretrained_model(self):
        """
        Loads a pretrained model based on the specified model name and modifies it for point detection.

        Returns:
        - nn.Module: The modified pretrained model.
        """
        if self.model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, 8)
        elif self.model_name == 'resnet50':
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, 8)
        elif self.model_name == 'mobilenet_v2':
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
        elif self.model_name == 'efficientnet_b0':
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, 8)
        else:
            raise ValueError(f"Model {self.model_name} is not supported.")
        return model.to(self.device)

    def train(self):
        """
        Trains the model on the training data and evaluates it on the validation data.
        Implements early stopping and logs the training process using TensorBoard.
        """
        best_val_loss = np.inf
        trigger_times = 0

        visualizer = PointVisualizer(self.directory, self.image_paths[0], self.coordinates[0], self.transform, self.device)
        visualizer.load_visualization()

        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            for images, coords in self.train_loader:
                images, coords = images.to(self.device), coords.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, coords)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * images.size(0)

            train_loss /= len(self.train_loader.dataset)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, coords in self.val_loader:
                    images, coords = images.to(self.device), coords.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, coords)
                    val_loss += loss.item() * images.size(0)

            val_loss /= len(self.val_loader.dataset)

            # Logging to TensorBoard
            self.writer.add_scalar('Loss/train', np.sqrt(train_loss) * 512, epoch)
            self.writer.add_scalar('Loss/val', np.sqrt(val_loss) * 512, epoch)

            print(f'Epoch {epoch + 1}/{self.num_epochs}, Train Loss: {np.sqrt(train_loss) * 512:.2f} px, Val Loss: {np.sqrt(val_loss) * 512:.2f} px')

            # Save predicted points for visualization
            pred_points = self.predict_points(self.image_paths[0])
            visualizer.add_points(epoch, pred_points)

            # Check for early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.model_dir, f'point_detection_model_best_{self.model_name}.pth'))
                trigger_times = 0
            else:
                trigger_times += 1

            if trigger_times >= self.patience:
                print('Early stopping triggered.')
                break

            # Step the scheduler
            self.scheduler.step(val_loss)

        # Save the final model
        torch.save(self.model.state_dict(), os.path.join(self.model_dir, f'point_detection_model_final_{self.model_name}.pth'))

        # Save the visualization
        visualizer.save_visualization(os.path.join(self.image_dir, f'point_movement_{self.model_name}.png'))

    def predict_points(self, image_path):
        """
        Predicts points for a single image using the trained model.

        Parameters:
        - image_path (str): Path to the image file.

        Returns:
        - numpy.ndarray: Predicted points.
        """
        self.model.eval()
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            predictions = self.model(image)
        return predictions.cpu().numpy()[0]

    def predict_on_folder(self, folder_path, output_file_path):
        """
        Predicts points for all images in a folder and saves the results to a evaluation file.

        Parameters:
        - folder_path (str): Path to the folder containing images.
        - output_file_path (str): Path to the output evaluation file.
        """
        predictions = []
        for img_name in os.listdir(folder_path):
            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, img_name)
                points = self.predict_points(image_path) * 512
                predictions.append(f"{img_name}," + ",".join(map(str, points.astype(int))))

        write_text_file(output_file_path, predictions)

    def evaluate(self, eval_text_file_path, output_file_path):
        """
        Evaluates the model on a dataset and saves the results.

        Parameters:
        - eval_text_file_path (str): Path to the evaluation file containing evaluation data.
        - output_file_path (str): Path to the output evaluation file.
        """
        image_paths, ground_truth_coords = load_images_and_coordinates(self.directory, eval_text_file_path)

        pixel_differences = []
        for image_path, true_coords in zip(image_paths, ground_truth_coords):
            pred_coords = self.predict_points(image_path) * 512
            pred_coords = pred_coords.astype(int)
            true_coords = np.array(true_coords)
            diff = np.abs(pred_coords - true_coords)
            pixel_differences.append(diff)

        mean_pixel_diff = np.mean(pixel_differences)
        with open(output_file_path, 'w') as file:
            for diff in pixel_differences:
                file.write(",".join(map(str, diff)) + '\n')
            file.write(f"Mean Pixel Difference: {mean_pixel_diff}\n")

        print(f"Mean Pixel Difference: {mean_pixel_diff}")

class ImageDataset(Dataset):
    def __init__(self, image_paths, coordinates, transform=None, augment=False):
        self.image_paths = image_paths
        self.coordinates = coordinates
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        coords = self.coordinates[idx]

        if self.transform:
            image = self.transform(image)

        coords = torch.tensor(coords, dtype=torch.float32)
        return image, coords

    def _augment_image(self, image, coords):
        # Random rotation
        angle = random.uniform(-30, 30)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, matrix, (w, h))

        # Rotate coordinates
        coords = np.array(coords).reshape(-1, 2)
        ones = np.ones(shape=(len(coords), 1))
        coords = np.hstack([coords, ones])
        rotated_coords = matrix.dot(coords.T).T

        # Random scaling
        scale = random.uniform(0.8, 1.2)
        matrix = cv2.getRotationMatrix2D(center, 0, scale)
        image = cv2.warpAffine(image, matrix, (w, h))

        # Scale coordinates
        scaled_coords = matrix.dot(np.hstack([rotated_coords, ones]).T).T

        return image, scaled_coords.flatten()


class PointVisualizer:
    def __init__(self, directory, eval_image_path, eval_coords, transform, device):
        self.directory = directory
        self.eval_image_path = eval_image_path
        self.eval_coords = np.array(eval_coords).reshape(-1, 2)
        self.transform = transform
        self.device = device
        self.epochs = []
        self.pred_points = []
        self.image = None

    def add_points(self, epoch, points):
        self.epochs.append(epoch)
        b = np.array(points) * 512
        self.pred_points.append(b.reshape(-1, 2))

    def load_visualization(self):
        self.image = cv2.imread(self.eval_image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

    def save_visualization(self, output_path):
        if self.image is None:
            raise ValueError("Image not loaded. Please load the image using load_visualization() method.")

        plt.figure(figsize=(20, 20))  # Increase figure size for higher resolution
        ax = plt.gca()  # Get current axes
        ax.imshow(self.image)
        ax.scatter(self.eval_coords[:, 0] * 512, self.eval_coords[:, 1] * 512, color='green', label='Ground Truth')

        colors = plt.cm.jet(np.linspace(0, 1, len(self.epochs)))

        for epoch, points, color in zip(self.epochs, self.pred_points, colors):
            ax.scatter(points[:, 0], points[:, 1], color=color, marker='.', label=f'Epoch {epoch}')

        ax.set_xlim((-20,600))

        ax.set_ylim((600,-20))

        # Create color bar representing epochs
        cmap = plt.cm.jet
        norm = plt.Normalize(vmin=min(self.epochs), vmax=max(self.epochs))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # dummy variable to create the colorbar

        # Add colorbar to the plot
        cbar = plt.colorbar(sm, ticks=self.epochs, orientation='horizontal', ax=ax)
        cbar.set_label('Epoch', rotation=0, labelpad=15)

        plt.title('Point Movement Visualization')


        # plt.legend()
        plt.savefig(output_path, dpi=300)  # Higher resolution output
        plt.close()






if __name__ == "__main__":
    directory = config['directories']['train_data_dir']
    text_file_path = os.path.join(directory,config['directories']['text_file'])


    for model_name in ['resnet18', 'resnet50', 'mobilenet_v2', 'efficientnet_b0']:
        print(f'Training model: {model_name}')
        model = PointDetectionModel(directory, text_file_path, model_name=model_name, num_epochs=400, patience=300, batch_size=64)
        model.train()

        folder_path = config['directories']['eval_data_dir']
        result_folder_path = config['directories']['results_text_dir']

        output_file_path = os.path.join(result_folder_path, f"predictions_{model_name}.txt")
        model.predict_on_folder(folder_path, output_file_path)
        print(f'Predictions saved to {output_file_path}')

