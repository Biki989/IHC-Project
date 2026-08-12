import torch
import numpy as np
from scipy.ndimage import distance_transform_edt

def compute_dsc(preds, targets, smooth=1e-6):
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)

def compute_iou(preds, targets, smooth=1e-6):
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def compute_hd95(preds, targets):
    # Calculate 95th percentile Hausdorff Distance
    # This expects batch dimension or single image numpy arrays
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    
    # We need to compute this per image in the batch and average
    b_size = preds.shape[0]
    hd95_sum = 0
    count = 0
    
    for i in range(b_size):
        pred = preds[i, 0]
        target = targets[i, 0]
        
        # If both are empty, distance is 0
        if pred.sum() == 0 and target.sum() == 0:
            count += 1
            continue
        
        # If one is empty and other is not, distance is max possible (or skip if we prefer)
        if pred.sum() == 0 or target.sum() == 0:
            hd95_sum += 256.0 # Max roughly across image
            count += 1
            continue
            
        # Get edges
        pred_edges = get_edges(pred)
        target_edges = get_edges(target)
        
        # Distance transforms
        pred_dt = distance_transform_edt(1 - target_edges)
        target_dt = distance_transform_edt(1 - pred_edges)
        
        distances_to_target = pred_dt[pred_edges > 0]
        distances_to_pred = target_dt[target_edges > 0]
        
        if len(distances_to_target) > 0 and len(distances_to_pred) > 0:
            hd95_sum += max(np.percentile(distances_to_target, 95), np.percentile(distances_to_pred, 95))
            count += 1
            
    if count == 0:
        return 0
    return hd95_sum / count

def get_edges(mask):
    # Simple edge detector using shifted arrays
    mask = mask.astype(bool)
    edge = np.zeros_like(mask)
    edge[1:] = edge[1:] | (mask[1:] != mask[:-1])
    edge[:-1] = edge[:-1] | (mask[:-1] != mask[1:])
    edge[:, 1:] = edge[:, 1:] | (mask[:, 1:] != mask[:, :-1])
    edge[:, :-1] = edge[:, :-1] | (mask[:, :-1] != mask[:, 1:])
    return edge

class MetricTracker:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.dsc_sum = 0
        self.iou_sum = 0
        self.hd95_sum = 0
        self.count = 0
        
    def update(self, logits, targets):
        preds = (torch.sigmoid(logits) > 0.5).float()
        targets = targets.float()
        
        self.dsc_sum += compute_dsc(preds, targets).item()
        self.iou_sum += compute_iou(preds, targets).item()
        # self.hd95_sum += compute_hd95(preds, targets)  # HD95 can be slow, maybe only in eval
        
        self.count += 1
        
    def get_metrics(self):
        return {
            "dsc": self.dsc_sum / self.count if self.count > 0 else 0,
            "iou": self.iou_sum / self.count if self.count > 0 else 0,
            # "hd95": self.hd95_sum / self.count if self.count > 0 else 0
        }
