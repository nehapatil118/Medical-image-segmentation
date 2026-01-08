## Skin Lesion Segmentation using U-Net

# Description
This project implements a U-Net–based deep learning model for medical image segmentation, specifically for segmenting
skin lesions from dermoscopic images using the ISIC dataset.
Medical image segmentation is a critical step in computer-aided diagnosis systems, enabling precise localization of disease
regions. In this project, a U-Net architecture is trained end-to-end to generate pixel-wise segmentation masks for skin lesion
images.
Key Highlights :-
- Implemented U-Net from scratch using PyTorch.
- Trained on real-world medical images (ISIC).
- Evaluated using Dice Score and IoU.
- Visualized predictions vs ground truth.
Dataset reference :-
https://challenge.isic-archive.com/data/#2018

# Dependencies
The list all libraries, packages and other dependencies that need to be installed to run your project.
- Install python of version 3.10.x.
- Create the virtual environment.
- Install the torch, torchvision, torchaudio, opencv-python, matplotlib and numpy.

# Usage 
- Activate the virtual environment.
- Install all the dependencies.
- Download the dataset.
- Create the model.
- Test dataset and model.
- Train the model.
- Evaluate model performance.
- Visualize predictions.
- Modify training parameters, if needed.

# Roadmap
- Add data augmentation for robustness.
- Experiment with Dice Loss.
- Extend to multi-class medical segmentation.
- Explore U-Net architecture.

# Contributing
- Fork the repository.
- Create a new branch.
- Make changes with proper documentation.
- Submit a pull request.

# License
This project is intended for academic and educational use.
Dataset usage follows ISIC Archive licensing terms.

# Author
Neha Patil

Thank you
