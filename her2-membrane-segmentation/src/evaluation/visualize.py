import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def visualize_prediction(image, true_mask, pred_mask, output_path):
    """
    image: [3, H, W] tensor or numpy array
    true_mask: [1, H, W] tensor or numpy array
    pred_mask: [1, H, W] tensor or numpy array
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)
    if isinstance(true_mask, torch.Tensor):
        true_mask = true_mask.cpu().numpy()[0]
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.cpu().numpy()[0]
        
    # De-normalize image for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(true_mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    axes[2].imshow(pred_mask, cmap='gray')
    axes[2].set_title('Prediction')
    axes[2].axis('off')
    
    # Overlay
    overlay = image.copy()
    # Red for ground truth
    overlay[true_mask > 0.5] = [1, 0, 0]
    # Green for prediction
    overlay[pred_mask > 0.5] = [0, 1, 0]
    # Yellow for overlap
    overlap = (true_mask > 0.5) & (pred_mask > 0.5)
    overlay[overlap] = [1, 1, 0]
    
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (Red:GT, Green:Pred, Yellow:Both)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
