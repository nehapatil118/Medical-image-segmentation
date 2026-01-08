import torch
import torch.nn as nn

# Double convolution block used throughout the U-Net
# Consists of two consecutive Conv2D + ReLU layers
class DoubleConv(nn.Module):

    # Initialize convolution layers
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            # First convolution layer
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Second convolution layer
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    # Forward pass through double convolution block
    def forward(self, x):
        return self.net(x)


# U-Net architecture for image segmentation
class UNet(nn.Module):

    # Initialize all layers of the U-Net
    def __init__(self):
        super().__init__()

        # ---------- Encoder (Downsampling Path) ----------
        self.d1 = DoubleConv(3, 64)     # Input: RGB image
        self.d2 = DoubleConv(64, 128)
        self.d3 = DoubleConv(128, 256)
        self.d4 = DoubleConv(256, 512)

        # Max pooling layer for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2)

        # ---------- Bottleneck ----------
        self.bottleneck = DoubleConv(512, 1024)

        # ---------- Decoder (Upsampling Path) ----------
        self.u4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.c4 = DoubleConv(1024, 512)   # Concatenation of encoder + decoder features

        self.u3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.c3 = DoubleConv(512, 256)

        self.u2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c2 = DoubleConv(256, 128)

        self.u1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c1 = DoubleConv(128, 64)

        # ---------- Output Layer ----------
        # Produces a single-channel segmentation mask
        self.out = nn.Conv2d(64, 1, kernel_size=1)

    # Forward propagation
    def forward(self, x):

        # ---------- Encoder ----------
        d1 = self.d1(x)
        d2 = self.d2(self.pool(d1))
        d3 = self.d3(self.pool(d2))
        d4 = self.d4(self.pool(d3))

        # ---------- Bottleneck ----------
        b = self.bottleneck(self.pool(d4))

        # ---------- Decoder ----------
        u4 = self.u4(b)
        u4 = torch.cat([u4, d4], dim=1)   # Skip connection
        u4 = self.c4(u4)

        u3 = self.u3(u4)
        u3 = torch.cat([u3, d3], dim=1)
        u3 = self.c3(u3)

        u2 = self.u2(u3)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.c2(u2)

        u1 = self.u1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.c1(u1)

        # ---------- Output ----------
        return self.out(u1)
