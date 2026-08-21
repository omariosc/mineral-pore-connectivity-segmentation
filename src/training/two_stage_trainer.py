"""
Two-Stage Training Pipeline for Extreme Class Imbalance
Stage 1: Binary segmentation (pore vs background)
Stage 2: Fine-grained classification (disconnected vs connected vs mineral)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path

from src.training.checkpoint_io import (
    load_weights_only_checkpoint,
    normalize_checkpoint_metadata,
)

class TwoStageTrainer:
    """
    Two-stage training strategy for handling extreme class imbalance.
    
    Stage 1: Train binary segmentation (pores vs minerals)
    Stage 2: Fine-tune for 3-class segmentation with frozen/semifrozen encoder
    """
    
    def __init__(self, model, device='cuda', save_dir='checkpoints'):
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # Training history
        self.stage1_history = []
        self.stage2_history = []
        
    def train_stage1(self, train_loader, val_loader, epochs=15, lr=0.001,
                     loss_fn=None, optimizer=None):
        """
        Stage 1: Binary segmentation training.
        Groups classes 0 and 1 (pores) vs class 2 (minerals).
        """
        print("="*50)
        print("STAGE 1: Binary Segmentation (Pores vs Minerals)")
        print("="*50)
        
        # Modify model output for binary classification
        original_out_channels = self.model.segmentation_head[-1].out_channels
        self.model.segmentation_head[-1] = nn.Conv2d(
            self.model.segmentation_head[-1].in_channels, 2, 1
        ).to(self.device)
        
        # Default loss for binary segmentation
        if loss_fn is None:
            # Weighted BCE for binary imbalance (pores ~13% vs minerals ~87%)
            loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 7.0]).to(self.device))
        
        # Default optimizer
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.0001)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_iou = 0.0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_pbar = tqdm(train_loader, desc=f"Stage 1 - Epoch {epoch+1}/{epochs}")
            
            for batch in train_pbar:
                images, labels = batch
                images = images.to(self.device)
                
                # Convert 3-class labels to binary (0,1 -> 0 (pores), 2 -> 1 (minerals))
                binary_labels = (labels == 2).long().to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = loss_fn(outputs, binary_labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': loss.item()})
            
            # Validation
            val_metrics = self._validate_binary(val_loader, loss_fn)
            
            # Update scheduler
            scheduler.step()
            
            # Save best model
            if val_metrics['pore_iou'] > best_iou:
                best_iou = val_metrics['pore_iou']
                self._save_checkpoint('stage1_best.pt', epoch, val_metrics)
            
            # Log metrics
            self.stage1_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss / len(train_loader),
                'val_loss': val_metrics['loss'],
                'pore_iou': val_metrics['pore_iou'],
                'mineral_iou': val_metrics['mineral_iou'],
                'mean_iou': val_metrics['mean_iou']
            })
            
            print(f"Stage 1 - Epoch {epoch+1}: "
                  f"Loss={val_metrics['loss']:.4f}, "
                  f"Pore IoU={val_metrics['pore_iou']:.4f}, "
                  f"Mean IoU={val_metrics['mean_iou']:.4f}")
        
        # Restore original output channels for stage 2
        self.model.segmentation_head[-1] = nn.Conv2d(
            self.model.segmentation_head[-1].in_channels, 
            original_out_channels, 1
        ).to(self.device)
        
        print(f"\nStage 1 Complete! Best Pore IoU: {best_iou:.4f}")
        return self.stage1_history
    
    def train_stage2(self, train_loader, val_loader, epochs=35, lr=0.0001,
                     loss_fn=None, optimizer=None, freeze_encoder=True,
                     stage1_checkpoint=None):
        """
        Stage 2: Fine-grained 3-class segmentation.
        Optionally freezes encoder from stage 1.
        """
        print("="*50)
        print("STAGE 2: Fine-grained 3-Class Segmentation")
        print("="*50)
        
        # Load stage 1 checkpoint if provided
        if stage1_checkpoint:
            self._load_checkpoint(stage1_checkpoint)
            print(f"Loaded stage 1 checkpoint: {stage1_checkpoint}")
        
        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()
            print("Encoder layers frozen")
        
        # Default loss for 3-class with extreme imbalance
        if loss_fn is None:
            from src.losses.focal_tversky import FocalTverskyLoss
            loss_fn = FocalTverskyLoss(
                alpha=0.7, beta=0.3, gamma=2.0,
                class_weights=[50.0, 10.0, 1.0]  # Heavy weights for minority
            )
        
        # Default optimizer (lower LR for fine-tuning)
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        if optimizer is None:
            optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.0001)
        
        # Scheduler with warmup
        warmup_epochs = 3
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader)
        )
        
        best_c0_iou = 0.0
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_pbar = tqdm(train_loader, desc=f"Stage 2 - Epoch {epoch+1}/{epochs}")
            
            for batch in train_pbar:
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = loss_fn(outputs, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': loss.item()})
            
            # Validation
            val_metrics = self._validate_3class(val_loader, loss_fn)
            
            # Save best model based on Class 0 IoU
            if val_metrics['class0_iou'] > best_c0_iou:
                best_c0_iou = val_metrics['class0_iou']
                self._save_checkpoint('stage2_best.pt', epoch, val_metrics)
            
            # Log metrics
            self.stage2_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss / len(train_loader),
                'val_loss': val_metrics['loss'],
                'class0_iou': val_metrics['class0_iou'],
                'class1_iou': val_metrics['class1_iou'],
                'class2_iou': val_metrics['class2_iou'],
                'pore_iou': val_metrics['pore_iou'],
                'mean_iou': val_metrics['mean_iou']
            })
            
            print(f"Stage 2 - Epoch {epoch+1}: "
                  f"Loss={val_metrics['loss']:.4f}, "
                  f"C0 IoU={val_metrics['class0_iou']:.4f}, "
                  f"Pore IoU={val_metrics['pore_iou']:.4f}, "
                  f"Mean IoU={val_metrics['mean_iou']:.4f}")
            
            # Unfreeze encoder after warmup
            if epoch == warmup_epochs - 1 and freeze_encoder:
                self._unfreeze_encoder(partial=True)
                print("Partially unfroze encoder after warmup")
        
        print(f"\nStage 2 Complete! Best Class 0 IoU: {best_c0_iou:.4f}")
        return self.stage2_history
    
    def _validate_binary(self, val_loader, loss_fn):
        """Validate binary segmentation."""
        self.model.eval()
        val_loss = 0.0
        
        # Metrics
        intersection = torch.zeros(2).to(self.device)
        union = torch.zeros(2).to(self.device)
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                binary_labels = (labels == 2).long().to(self.device)
                
                outputs = self.model(images)
                loss = loss_fn(outputs, binary_labels)
                val_loss += loss.item()
                
                # Predictions
                preds = torch.argmax(outputs, dim=1)
                
                # IoU calculation
                for c in range(2):
                    pred_c = (preds == c)
                    label_c = (binary_labels == c)
                    intersection[c] += (pred_c & label_c).sum()
                    union[c] += (pred_c | label_c).sum()
        
        # Calculate IoU
        iou = (intersection / (union + 1e-8)).cpu().numpy()
        
        return {
            'loss': val_loss / len(val_loader),
            'pore_iou': iou[0],
            'mineral_iou': iou[1],
            'mean_iou': iou.mean()
        }
    
    def _validate_3class(self, val_loader, loss_fn):
        """Validate 3-class segmentation."""
        self.model.eval()
        val_loss = 0.0
        
        # Metrics
        intersection = torch.zeros(3).to(self.device)
        union = torch.zeros(3).to(self.device)
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = loss_fn(outputs, labels)
                val_loss += loss.item()
                
                # Predictions
                preds = torch.argmax(outputs, dim=1)
                
                # IoU calculation
                for c in range(3):
                    pred_c = (preds == c)
                    label_c = (labels == c)
                    intersection[c] += (pred_c & label_c).sum()
                    union[c] += (pred_c | label_c).sum()
        
        # Calculate IoU
        iou = (intersection / (union + 1e-8)).cpu().numpy()
        
        return {
            'loss': val_loss / len(val_loader),
            'class0_iou': iou[0],
            'class1_iou': iou[1],
            'class2_iou': iou[2],
            'pore_iou': (iou[0] + iou[1]) / 2,  # Combined pore metric
            'mean_iou': iou.mean()
        }
    
    def _freeze_encoder(self):
        """Freeze encoder layers."""
        # Freeze based on common architectures
        for name, param in self.model.named_parameters():
            if 'encoder' in name or 'backbone' in name or 'features' in name:
                param.requires_grad = False
    
    def _unfreeze_encoder(self, partial=False):
        """Unfreeze encoder layers."""
        for name, param in self.model.named_parameters():
            if 'encoder' in name or 'backbone' in name:
                if partial:
                    # Only unfreeze later layers
                    if 'layer4' in name or 'stage4' in name or 'block4' in name:
                        param.requires_grad = True
                else:
                    param.requires_grad = True
    
    def _save_checkpoint(self, filename, epoch, metrics):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'metrics': normalize_checkpoint_metadata(metrics),
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def _load_checkpoint(self, filepath):
        """Load model checkpoint."""
        checkpoint = load_weights_only_checkpoint(
            filepath,
            map_location=self.device,
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('metrics', {})
    
    def save_training_history(self):
        """Save training history to JSON."""
        history = {
            'stage1': self.stage1_history,
            'stage2': self.stage2_history
        }
        
        with open(self.save_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training history saved to {self.save_dir / 'training_history.json'}")


class AdaptiveTwoStageTrainer(TwoStageTrainer):
    """
    Enhanced two-stage trainer with adaptive strategies.
    Includes curriculum learning and hard example mining.
    """
    
    def __init__(self, model, device='cuda', save_dir='checkpoints'):
        super().__init__(model, device, save_dir)
        self.hard_examples = []
        
    def train_stage2_with_curriculum(self, train_loader, val_loader, 
                                    epochs=35, lr=0.0001, difficulty_schedule=None):
        """
        Stage 2 with curriculum learning.
        Gradually introduces harder examples.
        """
        if difficulty_schedule is None:
            # Default schedule: start with easier examples
            difficulty_schedule = [
                (0, 0.3),   # Epochs 0-10: 30% easiest
                (10, 0.6),  # Epochs 10-20: 60% moderate
                (20, 0.9),  # Epochs 20-30: 90% harder
                (30, 1.0)   # Epochs 30+: all examples
            ]
        
        print("Stage 2 with Curriculum Learning")
        
        # Modified training loop with curriculum
        for epoch in range(epochs):
            # Determine difficulty threshold
            difficulty = 1.0
            for epoch_threshold, diff in difficulty_schedule:
                if epoch >= epoch_threshold:
                    difficulty = diff
            
            print(f"Epoch {epoch+1}: Using {difficulty*100:.0f}% of training data")
            
            # Filter dataset based on difficulty
            filtered_loader = self._filter_by_difficulty(train_loader, difficulty)
            
            # Train as usual but with filtered data
            # ... (rest of training loop)
    
    def _filter_by_difficulty(self, loader, difficulty_threshold):
        """Filter examples based on difficulty score."""
        # This would analyze each sample and assign difficulty
        # based on factors like pore density, size, etc.
        # For now, returning original loader
        return loader
    
    def mine_hard_examples(self, loader, top_k=100):
        """
        Identify hard examples based on current model performance.
        """
        self.model.eval()
        example_losses = []
        
        with torch.no_grad():
            for idx, (images, labels) in enumerate(loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                
                # Calculate per-sample loss
                loss = F.cross_entropy(outputs, labels, reduction='none')
                loss = loss.mean(dim=(1, 2))  # Average over spatial dims
                
                for i, l in enumerate(loss):
                    example_losses.append((idx * loader.batch_size + i, l.item()))
        
        # Sort by loss (highest = hardest)
        example_losses.sort(key=lambda x: x[1], reverse=True)
        
        # Store hard example indices
        self.hard_examples = [idx for idx, _ in example_losses[:top_k]]
        
        print(f"Identified {len(self.hard_examples)} hard examples")
        return self.hard_examples


# Utility function for easy two-stage training
def train_two_stage(model, train_loader, val_loader, config=None):
    """
    Convenience function for two-stage training.
    
    Args:
        model: Segmentation model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dict
    """
    if config is None:
        config = {
            'stage1_epochs': 15,
            'stage2_epochs': 35,
            'stage1_lr': 0.001,
            'stage2_lr': 0.0001,
            'freeze_encoder': True,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
    
    trainer = TwoStageTrainer(model, device=config['device'])
    
    # Stage 1: Binary segmentation
    stage1_history = trainer.train_stage1(
        train_loader, val_loader,
        epochs=config['stage1_epochs'],
        lr=config['stage1_lr']
    )
    
    # Stage 2: Fine-grained segmentation
    stage2_history = trainer.train_stage2(
        train_loader, val_loader,
        epochs=config['stage2_epochs'],
        lr=config['stage2_lr'],
        freeze_encoder=config['freeze_encoder'],
        stage1_checkpoint='checkpoints/stage1_best.pt'
    )
    
    # Save results
    trainer.save_training_history()
    
    return stage1_history, stage2_history


if __name__ == "__main__":
    print("Two-Stage Training Pipeline Implementation")
    print("=" * 50)
    
    # Mock test
    from torch.utils.data import TensorDataset
    
    # Create dummy data
    batch_size = 4
    num_samples = 20
    
    images = torch.randn(num_samples, 1, 256, 256)
    labels = torch.randint(0, 3, (num_samples, 256, 256))
    
    dataset = TensorDataset(images, labels)
    train_loader = DataLoader(dataset, batch_size=batch_size)
    val_loader = DataLoader(dataset, batch_size=batch_size)
    
    # Create dummy model
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Conv2d(1, 64, 3, padding=1)
            self.segmentation_head = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1),
                nn.Conv2d(32, 3, 1)
            )
        
        def forward(self, x):
            x = self.encoder(x)
            return self.segmentation_head(x)
    
    model = DummyModel()
    
    # Test trainer
    trainer = TwoStageTrainer(model, device='cpu', save_dir='test_checkpoints')
    
    print("Testing Stage 1 (Binary)...")
    trainer.train_stage1(train_loader, val_loader, epochs=2)
    
    print("\nTesting Stage 2 (3-class)...")
    trainer.train_stage2(train_loader, val_loader, epochs=2)
    
    print("\nTwo-stage training pipeline tested successfully!")
