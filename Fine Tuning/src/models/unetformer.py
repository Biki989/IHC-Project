import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x shape: (B, L, C)
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class UNetFormer(nn.Module):
    """
    CNN Encoder (ResNet34) + Transformer-based bottleneck and decoder hybrid.
    Designed to be lightweight enough for 4GB VRAM.
    """
    def __init__(self, encoder_name="resnet34", pretrained=True, in_channels=3, out_channels=1):
        super().__init__()
        
        # Encoder (CNN)
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet34(weights=weights)
        
        # Modify first conv if in_channels != 3
        if in_channels != 3:
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        self.enc1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu) # 64, H/2, W/2
        self.enc2 = nn.Sequential(resnet.maxpool, resnet.layer1)         # 64, H/4, W/4
        self.enc3 = resnet.layer2                                        # 128, H/8, W/8
        self.enc4 = resnet.layer3                                        # 256, H/16, W/16
        self.enc5 = resnet.layer4                                        # 512, H/32, W/32
        
        # Bottleneck (Transformer)
        self.bottleneck_dim = 512
        self.bottleneck_tf = TransformerBlock(dim=self.bottleneck_dim, num_heads=8)
        
        # Decoder (Hybrid CNN + simplistic transformer projection if needed, but we keep it light)
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        
        # Bottleneck
        B, C, H, W = e5.shape
        e5_flat = e5.flatten(2).transpose(1, 2) # (B, H*W, C)
        b_flat = self.bottleneck_tf(e5_flat)
        b = b_flat.transpose(1, 2).reshape(B, C, H, W)
        
        # Decoder
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
        
        out = self.final_up(d1)
        out = self.final_conv(out)
        
        return out
