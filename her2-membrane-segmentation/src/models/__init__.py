from .unet import UNet
from .unet_plusplus import UNetPlusPlus
from .attention_unet import AttentionUNet
from .nnunet import nnUNet
from .segformer import SegFormer
from .unetformer import UNetFormer
from .swin_unet import SwinUNet

MODEL_REGISTRY = {
    "unet": UNet,
    "unet_plusplus": UNetPlusPlus,
    "attention_unet": AttentionUNet,
    "nnunet": nnUNet,
    "segformer": SegFormer,
    "unetformer": UNetFormer,
    "swin_unet": SwinUNet,
}

def get_model(config):
    model_config = config['model']
    model_name = model_config['name']
    
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry.")
        
    ModelClass = MODEL_REGISTRY[model_name]
    
    if model_name == 'nnunet':
        return ModelClass(
            in_channels=model_config.get('in_channels', 3),
            out_channels=model_config.get('out_channels', 1)
        )
    
    if model_name in ['segformer', 'unetformer', 'swin_unet']:
        return ModelClass(
            encoder_name=model_config.get('encoder', 'resnet34'),
            pretrained=model_config.get('pretrained', True),
            in_channels=model_config.get('in_channels', 3),
            out_channels=model_config.get('out_channels', 1)
        )
        
    return ModelClass(
        encoder_name=model_config.get('encoder', 'resnet34'),
        encoder_weights="imagenet" if model_config.get('pretrained', True) else None,
        in_channels=model_config.get('in_channels', 3),
        out_channels=model_config.get('out_channels', 1)
    )
