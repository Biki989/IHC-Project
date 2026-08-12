import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import os
import json
from src.metrics.metrics import MetricTracker

class Trainer:
    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, scheduler, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.get('mixed_precision', True))
        self.grad_accum_steps = self.config.get('gradient_accumulation_steps', 1)
        self.output_dir = self.config['output_dir']
        
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        self.metric_tracker = MetricTracker()
        
    def train_epoch(self, epoch):
        self.model.train()
        self.metric_tracker.reset()
        epoch_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['epochs']}")
        for batch_idx, (images, masks) in enumerate(pbar):
            images, masks = images.to(self.device), masks.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.config.get('mixed_precision', True)):
                logits = self.model(images)
                loss = self.loss_fn(logits, masks)
                loss = loss / self.grad_accum_steps
                
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                
            epoch_loss += loss.item() * self.grad_accum_steps
            self.metric_tracker.update(logits.detach(), masks)
            
            pbar.set_postfix({'loss': loss.item() * self.grad_accum_steps})
            
        metrics = self.metric_tracker.get_metrics()
        metrics['train_loss'] = epoch_loss / len(self.train_loader)
        return metrics

    def validate(self):
        self.model.eval()
        self.metric_tracker.reset()
        val_loss = 0
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                
                with torch.cuda.amp.autocast(enabled=self.config.get('mixed_precision', True)):
                    logits = self.model(images)
                    loss = self.loss_fn(logits, masks)
                    
                val_loss += loss.item()
                self.metric_tracker.update(logits, masks)
                
        metrics = self.metric_tracker.get_metrics()
        metrics['val_loss'] = val_loss / len(self.val_loader)
        return metrics

    def fit(self):
        best_val_dsc = 0
        patience_counter = 0
        
        for epoch in range(1, self.config['epochs'] + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            
            self.scheduler.step()
            
            # Logging
            log_dict = {
                "epoch": epoch,
                "train_loss": train_metrics['train_loss'],
                "train_dsc": train_metrics['dsc'],
                "val_loss": val_metrics['val_loss'],
                "val_dsc": val_metrics['dsc'],
                "val_iou": val_metrics['iou'],
                "lr": self.optimizer.param_groups[0]['lr']
            }
            wandb.log(log_dict)
            
            print(f"Val DSC: {val_metrics['dsc']:.4f} | Val Loss: {val_metrics['val_loss']:.4f}")
            
            # Checkpointing
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_dsc": best_val_dsc,
                "config": self.config
            }
            
            torch.save(checkpoint, os.path.join(self.output_dir, "checkpoints", "last_model.pth"))
            
            if val_metrics['dsc'] > best_val_dsc:
                best_val_dsc = val_metrics['dsc']
                torch.save(checkpoint, os.path.join(self.output_dir, "checkpoints", "best_model.pth"))
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.get('early_stopping_patience', 15):
                print(f"Early stopping triggered at epoch {epoch}")
                break
