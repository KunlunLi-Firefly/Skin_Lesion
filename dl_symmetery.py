import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split

# Define the path to the dataset and the mapping from textual labels to integers.
root_dir = '/Users/shipeiqi/Desktop/·/master/24spring/MIA/hw/hw6/skin_lesion_dataset'
label_mapping = {"Fully Symmetric": 0, "Symmetric in 1 axes": 1, "Fully Asymmetric": 2}

# Initialize lists to store image paths and their corresponding labels.
image_paths = []
labels = []

# Iterate over the directory to read image paths and labels.
for subdir in os.listdir(root_dir):
    if not subdir.startswith('.'):  # Skip hidden files and directories
        lesion_subdir = os.path.join(root_dir, subdir, f"{subdir}_lesion")
        label_file_path = os.path.join(root_dir, subdir, f"{subdir}_label.json")
        if os.path.isfile(label_file_path):
            with open(label_file_path, 'r') as f:
                label_data = json.load(f)
            label = label_mapping[label_data['Asymmetry Label']]
            for filename in os.listdir(lesion_subdir):
                if filename.endswith('.bmp'):
                    image_paths.append(os.path.join(lesion_subdir, filename))
                    labels.append(label)

# Define the dataset class for skin lesion images.
class SkinLesionDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Load the image and label.
        image = Image.open(self.image_paths[index]).convert('RGB')
        label = torch.tensor(self.labels[index], dtype=torch.long)

        # Apply transformations to the image.
        if self.transform:
            image = self.transform(image)
        return image, label

# Split the dataset into training and testing sets (60% train, 40% test).
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.4, random_state=42
)

# Define data transformations for training and testing sets.
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets and data loaders for training and testing.
train_dataset = SkinLesionDataset(train_paths, train_labels, transform=train_transforms)
test_dataset = SkinLesionDataset(test_paths, test_labels, transform=test_transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load the pretrained ResNet18 model and modify the final layer to match the number of classes.
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, len(label_mapping))

# Define the loss function and the optimizer with weight decay for regularization.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Define Early Stopping class.
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.Inf
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Initialize Early Stopping object.
early_stopping = EarlyStopping(patience=10, min_delta=0.01)

# Train the model with early stopping.
num_epochs = 50
best_model_state = model.state_dict()
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        # Zero the parameter gradients.
        optimizer.zero_grad()

        # Forward pass.
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize.
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Calculate average training loss.
    train_loss = running_loss / len(train_loader)

    # Validation phase.
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # Calculate average validation loss.
    val_loss /= len(test_loader)

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    # Check for early stopping.
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print("Early stopping")
        break
    else:
        best_model_state = model.state_dict()  # Update the best model if early stopping did not trigger.

# Load the best model state.
model.load_state_dict(best_model_state)

# Evaluate the model on the test set.
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        test_correct += (predicted == labels).sum().item()
        test_total += labels.size(0)

test_accuracy = test_correct / test_total
print(f'Test Accuracy: {test_accuracy:.2f}')
