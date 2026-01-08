import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from dataset import ISICDataset
from model import UNet


# Training function
def train():

    # Select GPU if available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load ISIC dataset (images and masks)
    dataset = ISICDataset("data/images", "data/masks")

    # Create DataLoader for batching and shuffling data
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    # Initialize U-Net model and move it to selected device
    model = UNet().to(device)

    # Binary Cross-Entropy loss with logits for segmentation
    criterion = nn.BCEWithLogitsLoss()

    # Adam optimizer for model training
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Set model to training mode
    model.train()

    # Training configuration
    num_epochs = 20
    best_loss = float("inf")

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        # Iterate over batches
        for images, masks in loader:

            # Move data to device (CPU/GPU)
            images = images.to(device)
            masks = masks.to(device)

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, masks)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate batch loss
            epoch_loss += loss.item()

        # Calculate average loss for the epoch
        avg_loss = epoch_loss / len(loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Save model if current loss is the best so far
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "results/unet_model.pth")
            print(f"Model saved (best loss: {best_loss:.4f})")


# Entry point of the script
if __name__ == "__main__":
    train()
