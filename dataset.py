import os
import cv2
import torch
from torch.utils.data import Dataset

# Custom Dataset class for ISIC skin lesion images and segmentation masks
class ISICDataset(Dataset):

    # Constructor: initializes image and mask directories
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir      # Directory containing images
        self.mask_dir = mask_dir        # Directory containing masks

        # List all image files with .jpg extension
        self.images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith(".jpg")
        ]

    # Returns total number of image-mask pairs
    def __len__(self):
        return len(self.images)

    # Returns one image and its corresponding mask
    def __getitem__(self, idx):

        # Get image file name using index
        img_name = self.images[idx]

        # Create full path to the image
        img_path = os.path.join(self.image_dir, img_name)

        # Generate corresponding mask file name
        base_name = img_name.replace(".jpg", "")
        mask_name = base_name + "_segmentation.png"
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Read image using OpenCV
        image = cv2.imread(img_path)

        # Convert image from BGR to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image to fixed size (256x256)
        image = cv2.resize(image, (256, 256))

        # Normalize pixel values to range [0, 1]
        image = image / 255.0

        # Convert image to PyTorch tensor and change shape to (C, H, W)
        image = torch.tensor(image).permute(2, 0, 1).float()

        # Read mask in grayscale mode
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Resize mask to same size as image
        mask = cv2.resize(mask, (256, 256))

        # Normalize mask values to range [0, 1]
        mask = mask / 255.0

        # Convert mask to tensor and add channel dimension
        mask = torch.tensor(mask).unsqueeze(0).float()

        # Return image and corresponding mask
        return image, mask
