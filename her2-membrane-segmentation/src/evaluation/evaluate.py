import torch
from tqdm import tqdm
import json
import os
import pandas as pd
from src.metrics.metrics import MetricTracker

def evaluate_test_set(model, test_loader, device, output_dir):
    model.eval()
    metric_tracker = MetricTracker()
    
    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating"):
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            metric_tracker.update(logits, masks)
            
    metrics = metric_tracker.get_metrics()
    
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    with open(os.path.join(output_dir, "metrics", "final_results.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Test DSC: {metrics['dsc']:.4f}")
    print(f"Test IoU: {metrics['iou']:.4f}")
    return metrics
