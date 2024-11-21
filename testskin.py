# import os
# import json
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from PIL import Image
# from torchvision.models import resnet18
#
# # Load the model
# model_path = '/Users/shipeiqi/Desktop/best_model.pth'
# model = resnet18(weights=None)  # Assume initialization to match training
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, 3)  # Adjust for the number of classes (fully symmetric, etc.)
# model.load_state_dict(torch.load(model_path))
# model.eval()
#
# # Transformation setup
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])
#
#
# def process_images(root_dir):
#     results = []
#     # Iterate over subdirectories as per your training setup
#     for subdir in os.listdir(root_dir):
#         if not subdir.startswith('.'):  # Ignore hidden files/directories
#             image_dir = os.path.join(root_dir, subdir, f"{subdir}_lesion")
#             if os.path.isdir(image_dir):
#                 for filename in os.listdir(image_dir):
#                     if filename.endswith('.bmp'):
#                         image_path = os.path.join(image_dir, filename)
#                         image = Image.open(image_path).convert('RGB')
#                         image = transform(image)
#                         image = image.unsqueeze(0)  # Add batch dimension
#
#                         # Predict the class
#                         with torch.no_grad():
#                             output = model(image)
#                             _, predicted = torch.max(output, 1)
#                             predicted_class = predicted.item()
#
#                         # Map class index to label
#                         label_mapping = {0: "Fully Symmetric", 1: "Symmetric in 1 axes", 2: "Fully Asymmetric"}
#                         predicted_label = label_mapping[predicted_class]
#
#                         # Output JSON file
#                         output_filename = os.path.splitext(filename)[0] + '_label.json'
#                         output_path = os.path.join(image_dir, output_filename)
#                         with open(output_path, 'w') as f:
#                             json.dump({"Lesion Class": "Atypical Nevus", "Asymmetry Label": predicted_label}, f)
#
#                         results.append((image_path, output_path))
#     return results
#
#
# # Specify the root directory containing image subdirectories
# root_dir = '/Users/shipeiqi/Desktop/segmentation_mask'  # Update this path as needed
# processed_files = process_images(root_dir)
# for image_path, json_path in processed_files:
#     print(f'Processed {image_path}: Output JSON -> {json_path}')


import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from torchvision.models import resnet18
import logging




# Setup logging
logging.basicConfig(filename='image_processing.log', level=logging.INFO)

# Load the model
model_path = '/Users/shipeiqi/Desktop/best_model.pth'
model = resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)
try:
    model.load_state_dict(torch.load(model_path))
    model.eval()
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}")
    raise RuntimeError(f"Failed to load model: {e}")

# Transformation setup
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def process_images(input_dir, output_dir):
    results = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for subdir in os.listdir(input_dir):
        if not subdir.startswith('.') and os.path.isdir(os.path.join(input_dir, subdir)):
            input_image_path = os.path.join(input_dir, subdir, f'{subdir}_lesion.bmp')
            output_subdir = os.path.join(output_dir, subdir)  # Output directory in original case
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            if os.path.isfile(input_image_path):  # Check if the file exists
                try:
                    image = Image.open(input_image_path).convert('RGB')
                    image = transform(image)
                    image = image.unsqueeze(0)

                    with torch.no_grad():
                        output = model(image)
                        _, predicted = torch.max(output, 1)
                        predicted_class = predicted.item()

                    label_mapping = {0: "Fully Symmetric", 1: "Symmetric in 1 axes", 2: "Fully Asymmetric"}
                    predicted_label = label_mapping[predicted_class]

                    output_filename = f'{subdir.lower()}_label.json'  # Only filename in lowercase
                    output_path = os.path.join(output_subdir, output_filename)
                    with open(output_path, 'w') as f:
                        json.dump({"Lesion Class": "Atypical Nevus", "Asymmetry Label": predicted_label}, f)

                    results.append((input_image_path, output_path))
                    logging.info(f'Processed {input_image_path}: Output JSON -> {output_path}')
                except Exception as e:
                    logging.error(f"Error processing {input_image_path}: {e}")
    return results

# Example usage
input_dir = '/Users/shipeiqi/Desktop/segmentation_mask'
output_dir = '/Users/shipeiqi/Desktop/label'
processed_files = process_images(input_dir, output_dir)
