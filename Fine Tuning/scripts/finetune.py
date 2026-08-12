import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import HER2MembraneDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models import get_model
from src.losses.dice_bce import DiceBCELoss
from src.training.trainer import Trainer
from src.training.scheduler import get_scheduler

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    if '_base_' in config:
        with open(config['_base_'], 'r') as f:
            base_config = yaml.safe_load(f)
        # Simple merge, overriding base with child
        base_config.update(config)
        config = base_config
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to finetune config')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Prefix output_dir with output_base_dir if present
    output_base_dir = config.get('output_base_dir', '.')
    orig_output_dir = config['output_dir'] # e.g. outputs/unet
    config['output_dir'] = os.path.join(output_base_dir, orig_output_dir.replace('outputs/', ''))
    
    # Determine pretrained weights path
    pretrained_base_dir = config.get('pretrained_base_dir')
    if pretrained_base_dir:
        # e.g., d:/Annotated work/her2-membrane-segmentation/outputs/unet/checkpoints/best_model.pth
        pretrained_weights_path = os.path.join(pretrained_base_dir, orig_output_dir.replace('outputs/', ''), "checkpoints", "best_model.pth")
    else:
        raise ValueError("pretrained_base_dir not found in config")

    print(f"Fine-tuning {config['model']['name']}...")
    print(f"Loading weights from {pretrained_weights_path}")
    print(f"Output directory: {config['output_dir']}")
    
    # Set seed
    torch.manual_seed(config.get('seed', 42))
    
    # Dataloaders
    train_dataset = HER2MembraneDataset(
        data_dir=os.path.join(config['data_dir'], 'train'),
        split_file=None,
        transform=get_train_transforms(config['patch_size'])
    )
    val_dataset = HER2MembraneDataset(
        data_dir=os.path.join(config['data_dir'], 'val'),
        split_file=None,
        transform=get_val_transforms(config['patch_size'])
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    
    # Model
    model = get_model(config)
    
    # Load Pretrained Weights
    if os.path.exists(pretrained_weights_path):
        checkpoint = torch.load(pretrained_weights_path, map_location="cpu", weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Successfully loaded pre-trained weights.")
    else:
        raise FileNotFoundError(f"Pretrained weights not found at {pretrained_weights_path}")
    
    # Loss
    loss_fn = DiceBCELoss(dice_weight=config['dice_weight'], bce_weight=config['bce_weight'])
    
    # Optimizer (smaller learning rate for finetuning)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    # Scheduler
    scheduler = get_scheduler(optimizer, config)
    
    # WandB
    wandb.init(
        project=config['wandb_project'],
        name=config['wandb_run_name'],
        config=config,
        mode="disabled"
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config
    )
    
    # Train
    trainer.fit()
    wandb.finish()

if __name__ == "__main__":
    main()
