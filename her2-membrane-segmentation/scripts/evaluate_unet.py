import argparse
import yaml
import torch
import os
import sys
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import HER2MembraneDataset
from src.data.transforms import get_val_transforms
from src.models import get_model
from src.evaluation.evaluate import evaluate_test_set

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
    parser.add_argument('--config', type=str, default='configs/unet.yaml')
    parser.add_argument('--checkpoint', type=str, default='outputs/unet/checkpoints/best_model.pth')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # Dataloader
    test_dataset = HER2MembraneDataset(
        data_dir=os.path.join(config['data_dir'], 'test'),
        split_file=None,
        transform=get_val_transforms(config['patch_size'])
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.get('num_workers', 2))
    
    # Model
    model = get_model(config)
    
    # Load checkpoint
    print(f"Loading checkpoint {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Evaluate
    evaluate_test_set(model, test_loader, device, config['output_dir'])

if __name__ == "__main__":
    main()
