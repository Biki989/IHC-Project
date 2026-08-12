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
        base_config.update(config)
        config = base_config
    return config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/unetformer.yaml')
    args = parser.parse_args()
    
    config = load_config(args.config)
    torch.manual_seed(config.get('seed', 42))
    
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
    
    model = get_model(config)
    
    loss_fn = DiceBCELoss(dice_weight=config['dice_weight'], bce_weight=config['bce_weight'])
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = get_scheduler(optimizer, config)
    
    wandb.init(
        project=config['wandb_project'],
        name=config['wandb_run_name'],
        config=config,
        mode="disabled"
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config
    )
    
    trainer.fit()
    wandb.finish()

if __name__ == "__main__":
    main()
