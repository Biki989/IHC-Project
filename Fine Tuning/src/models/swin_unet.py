import torch
import torch.nn as nn
import timm

class SwinUNet(nn.Module):
    """
    Swin-UNet variant utilizing a pretrained Swin Transformer from `timm` 
    as the encoder, paired with a CNN-based decoder.
    Adapted for 4GB VRAM constraint (uses tiny backbone).
    """
    def __init__(self, encoder_name="swin_tiny_patch4_window7_224", pretrained=True, in_channels=3, out_channels=1):
        super().__init__()
        
        # Swin Transformer Encoder
        self.encoder = timm.create_model(
            encoder_name, 
            pretrained=pretrained, 
            features_only=True,
            in_chans=in_channels,
            out_indices=(0, 1, 2, 3) # Extracts multi-scale features
        )
        
        # Assuming swin_tiny dimensions: 96, 192, 384, 768
        # (might differ slightly by model, but tiny uses these)
        
        self.up4 = nn.ConvTranspose2d(768, 384, kernel_size=2, stride=2)
        self.dec4 = nn.Sequential(
            nn.Conv2d(768, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True)
        )
        
        self.up3 = nn.ConvTranspose2d(384, 192, kernel_size=2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose2d(192, 96, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(192, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True)
        )
        
        # Additional upsampling since patch4 downsamples by 4x initially
        self.up1 = nn.ConvTranspose2d(96, 48, kernel_size=4, stride=4)
        self.dec1 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(48, out_channels, kernel_size=1)
        
    def forward(self, x):
        original_size = x.shape[2:]
        # Resize input to match Swin Transformer expected resolution
        if x.shape[2] != 224 or x.shape[3] != 224:
            x_224 = nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        else:
            x_224 = x
            
        features = self.encoder(x_224)
        # timm's Swin returns features in [B, H, W, C], we need [B, C, H, W]
        e0 = features[0].permute(0, 3, 1, 2).contiguous()
        e1 = features[1].permute(0, 3, 1, 2).contiguous()
        e2 = features[2].permute(0, 3, 1, 2).contiguous()
        e3 = features[3].permute(0, 3, 1, 2).contiguous()
        
        # Decoder
        d4 = self.up4(e3)
        # resize if shapes don't perfectly match due to padding
        if d4.shape != e2.shape:
            d4 = nn.functional.interpolate(d4, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e2], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        if d3.shape != e1.shape:
            d3 = nn.functional.interpolate(d3, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e1], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        if d2.shape != e0.shape:
            d2 = nn.functional.interpolate(d2, size=e0.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e0], dim=1)
        d2 = self.dec2(d2)
        
        # Final upsample (Swin reduces by 4 initially, so we upsample by 4)
        d1 = self.up1(d2)
        if d1.shape != x.shape:
            d1 = nn.functional.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(d1)
        
        out = self.final_conv(d1)
        
        # Resize output back to original resolution if it was altered
        if out.shape[2:] != original_size:
            out = nn.functional.interpolate(out, size=original_size, mode='bilinear', align_corners=False)
            
        return out
