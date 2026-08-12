import argparse
import yaml
import torch
import os
import sys
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import HER2MembraneDataset
from src.data.transforms import get_val_transforms
from src.models import get_model
from src.evaluation.visualize import visualize_prediction

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
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    output_dir = os.path.join(config['output_dir'], 'predictions')
    os.makedirs(output_dir, exist_ok=True)
    
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
    model.eval()
    
    count = 0
    with torch.no_grad():
        for i, (images, masks) in enumerate(tqdm(test_loader, desc="Visualizing")):
            if count >= args.num_samples:
                break
                
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            output_path = os.path.join(output_dir, f'pred_{i}.png')
            visualize_prediction(images[0], masks[0], preds[0], output_path)
            count += 1
            
    print(f"Saved {count} visualizations to {output_dir}")

if __name__ == "__main__":
    main()
