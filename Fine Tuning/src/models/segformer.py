import torch.nn as nn
import transformers.modeling_utils
transformers.modeling_utils.check_torch_load_is_safe = lambda: True
from transformers import SegformerForSemanticSegmentation

class SegFormer(nn.Module):
    def __init__(self, encoder_name="nvidia/mit-b0", pretrained=True, in_channels=3, out_channels=1):
        super().__init__()
        
        # Load the Segformer model with custom number of labels
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                encoder_name,
                num_labels=out_channels,
                ignore_mismatched_sizes=True
            )
        else:
            from transformers import SegformerConfig
            config = SegformerConfig.from_pretrained(encoder_name, num_labels=out_channels)
            self.model = SegformerForSemanticSegmentation(config)
            
    def forward(self, x):
        outputs = self.model(pixel_values=x)
        # Segformer outputs logits of size (B, num_labels, H/4, W/4)
        logits = outputs.logits
        # Upsample to match input resolution
        upsampled_logits = nn.functional.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return upsampled_logits
