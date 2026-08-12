import torch.nn as nn
import segmentation_models_pytorch as smp

class UNet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_channels,
        )
        
    def forward(self, x):
        return self.model(x)
