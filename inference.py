import torch
import cv2
import os
import random
import matplotlib.pyplot as plt

from model import UNet
from dataset import ISICDataset
from metrics import dice_score, iou_score


# ---------------------- Device Configuration ----------------------
# Use GPU if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------- Model Loading ----------------------
# Initialize U-Net model and load trained weights
model = UNet().to(device)
model.load_state_dict(torch.load("results/unet_model.pth", map_location=device))

# Set model to evaluation mode
model.eval()


# ---------------------- Dataset Loading ----------------------
# Load ISIC dataset for evaluation
dataset = ISICDataset("data/images", "data/masks")


# ---------------------- Multi-image Evaluation ----------------------
# Lists to store Dice and IoU scores
dice_scores = []
iou_scores = []

# Number of samples to evaluate
num_samples = 20

# Randomly select sample indices
indices = random.sample(range(len(dataset)), num_samples)

# Loop through selected samples
for idx in indices:
    image, mask = dataset[idx]

    # Add batch dimension and move to device
    image = image.unsqueeze(0).to(device)
    mask = mask.to(device)

    # Disable gradient calculation for inference
    with torch.no_grad():
        pred = model(image)

        # Apply sigmoid to convert logits to probabilities
        pred = torch.sigmoid(pred)

        # Threshold probabilities to get binary mask
        pred = (pred > 0.5).float()

    # Move tensors back to CPU and remove batch/channel dimensions
    pred = pred.cpu().squeeze()
    mask = mask.cpu().squeeze()

    # Calculate and store metrics
    dice_scores.append(dice_score(pred, mask))
    iou_scores.append(iou_score(pred, mask))


# Print average evaluation scores
print(f"\nAverage Dice Score ({num_samples} images): {sum(dice_scores)/len(dice_scores):.4f}")
print(f"Average IoU Score ({num_samples} images): {sum(iou_scores)/len(iou_scores):.4f}")


# ---------------------- Multi-image Visualization ----------------------
# Number of samples to visualize
num_visualize = 5

# Randomly select visualization samples
vis_indices = random.sample(range(len(dataset)), num_visualize)

# Create figure for visualization
plt.figure(figsize=(12, 4 * num_visualize))

# Loop through visualization samples
for i, idx in enumerate(vis_indices):
    image, mask = dataset[idx]
    image_tensor = image.unsqueeze(0).to(device)

    # Model inference
    with torch.no_grad():
        pred = model(image_tensor)
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()

    # Convert tensors to NumPy-friendly format for plotting
    image_np = image.cpu().permute(1, 2, 0)
    mask_np = mask.squeeze()
    pred_np = pred.cpu().squeeze()

    # -------- Input Image --------
    plt.subplot(num_visualize, 3, i * 3 + 1)
    plt.imshow(image_np)
    plt.title(f"Input Image {i+1}")
    plt.axis("off")

    # -------- Ground Truth Mask --------
    plt.subplot(num_visualize, 3, i * 3 + 2)
    plt.imshow(mask_np, cmap="gray")
    plt.title("Ground Truth")
    plt.axis("off")

    # -------- Predicted Mask --------
    plt.subplot(num_visualize, 3, i * 3 + 3)
    plt.imshow(pred_np, cmap="gray")
    plt.title("Prediction")
    plt.axis("off")

# Adjust layout and display results
plt.tight_layout()
plt.show()
