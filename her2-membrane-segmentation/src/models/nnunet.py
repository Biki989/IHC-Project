import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.InstanceNorm2d(out_c, affine=True)
        self.lrelu1 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(out_c, affine=True)
        self.lrelu2 = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.lrelu1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.lrelu2(x)
        return x

class nnUNet(nn.Module):
    """
    A simplified 2D nnU-Net-like architecture adapted for our pipeline.
    Key features: InstanceNorm instead of BatchNorm, LeakyReLU, no pre-training.
    (Omits deep supervision for compatibility with our standard training loop).
    """
    def __init__(self, in_channels=3, out_channels=1, base_c=32):
        super().__init__()
        
        # Encoder
        self.enc1 = ConvBlock(in_channels, base_c)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = ConvBlock(base_c, base_c * 2)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = ConvBlock(base_c * 2, base_c * 4)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = ConvBlock(base_c * 4, base_c * 8)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(base_c * 8, base_c * 16)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_c * 16, base_c * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_c * 16, base_c * 8)
        
        self.up3 = nn.ConvTranspose2d(base_c * 8, base_c * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_c * 8, base_c * 4)
        
        self.up2 = nn.ConvTranspose2d(base_c * 4, base_c * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_c * 4, base_c * 2)
        
        self.up1 = nn.ConvTranspose2d(base_c * 2, base_c, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_c * 2, base_c)
        
        self.final_conv = nn.Conv2d(base_c, out_channels, kernel_size=1)
        
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        b = self.bottleneck(self.pool4(e4))
        
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return self.final_conv(d1)
