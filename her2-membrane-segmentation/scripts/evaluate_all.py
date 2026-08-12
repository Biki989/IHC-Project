import argparse
import yaml
import torch
import os
import sys
import pandas as pd
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
    configs = [
        'configs/unet.yaml',
        'configs/unet_plusplus.yaml',
        'configs/attention_unet.yaml',
        'configs/nnunet.yaml',
        'configs/segformer.yaml',
        'configs/unetformer.yaml',
        'configs/swin_unet.yaml'
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_metrics = []
    
    for config_path in configs:
        if not os.path.exists(config_path):
            print(f"Skipping {config_path} as it does not exist.")
            continue
            
        print(f"\n{'='*50}\nEvaluating model with {config_path}\n{'='*50}\n")
        config = load_config(config_path)
        
        # Check if checkpoint exists
        checkpoint_path = os.path.join(config['output_dir'], 'checkpoints', 'best_model.pth')
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}. Skipping.")
            continue
            
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
        print(f"Loading checkpoint {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Evaluate
        metrics = evaluate_test_set(model, test_loader, device, config['output_dir'])
        metrics['model_name'] = config['model']['name']
        all_metrics.append(metrics)
        
    if all_metrics:
        df = pd.DataFrame(all_metrics)
        # Reorder columns to have model_name first
        cols = ['model_name'] + [col for col in df.columns if col != 'model_name']
        df = df[cols]
        
        os.makedirs('outputs', exist_ok=True)
        csv_path = os.path.join('outputs', 'comparison_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nSuccessfully evaluated all available models. Results saved to {csv_path}")
        print(df.to_string(index=False))
    else:
        print("\nNo models were evaluated (checkpoints not found).")

if __name__ == "__main__":
    main()
