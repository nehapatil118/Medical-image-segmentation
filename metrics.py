import torch

# Function to compute Dice Score (Dice Coefficient)
def dice_score(pred, target, smooth=1e-6):

    # Ensure tensors are stored contiguously in memory
    # This avoids unexpected issues during tensor operations
    pred = pred.contiguous()
    target = target.contiguous()

    # Calculate the intersection between prediction and target
    intersection = (pred * target).sum()

    # Dice coefficient formula
    dice = (2. * intersection + smooth) / (
        pred.sum() + target.sum() + smooth
    )

    # Return Dice score as a Python float
    return dice.item()


# Function to compute Intersection over Union (IoU) score
def iou_score(pred, target, smooth=1e-6):

    # Calculate the intersection between prediction and target
    intersection = (pred * target).sum()

    # Calculate the union
    union = pred.sum() + target.sum() - intersection

    # IoU formula
    iou = (intersection + smooth) / (union + smooth)

    # Return IoU score as a Python float
    return iou.item()
