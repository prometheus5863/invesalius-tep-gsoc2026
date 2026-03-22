"""
1D-CNN for TMS-EEG Artifact Rejection
=====================================

A complete deep learning pipeline for classifying TMS-EEG epochs as artifact or clean.

INPUT shape: (batch, 19, 701) — 19 EEG channels, 701 timepoints at 1kHz.
OUTPUT: binary probability (artifact=0, clean=1).

Author: Auto-generated
"""

import os
import warnings
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

import matplotlib.pyplot as plt


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class EEGArtifactCNN(nn.Module):
    """
    1D Convolutional Neural Network for TMS-EEG Artifact Classification.
    
    Architecture:
    - SpatialFilter: Conv1d(19, 32, kernel=1) — learns channel combinations
    - TemporalBlock1: Conv1d(32, 64, kernel=25, padding=12) + BN + GELU + MaxPool(4)
    - TemporalBlock2: Conv1d(64, 128, kernel=15, padding=7) + BN + GELU + MaxPool(4)
    - TemporalBlock3: Conv1d(128, 256, kernel=7, padding=3) + BN + GELU + AdaptiveAvgPool(1)
    - Classifier: Flatten → Dropout(0.4) → Linear(256,64) → GELU → Linear(64,1) → Sigmoid
    
    Args:
        n_channels: Number of EEG channels (default: 19)
        n_timepoints: Number of time samples (default: 701)
    """
    
    def __init__(self, n_channels: int = 19, n_timepoints: int = 701):
        super(EEGArtifactCNN, self).__init__()
        
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        
        # Spatial Filter: learns optimal channel combinations
        self.spatial_filter = nn.Conv1d(
            in_channels=n_channels, 
            out_channels=32, 
            kernel_size=1
        )
        
        # Temporal Block 1: captures long-range temporal patterns
        self.temporal_block1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=25, padding=12),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4)
        )
        
        # Temporal Block 2: captures medium-range patterns
        self.temporal_block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4)
        )
        
        # Temporal Block 3: captures fine-grained patterns
        self.temporal_block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Store intermediate activations for Grad-CAM
        self.activations = {}
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, n_channels, n_timepoints)
            
        Returns:
            Probability of clean epoch (artifact=0, clean=1)
        """
        # Spatial filtering
        x = self.spatial_filter(x)
        
        # Temporal processing
        x = self.temporal_block1(x)
        x = self.temporal_block2(x)
        x = self.temporal_block3(x)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def forward_with_hooks(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that stores intermediate activations for Grad-CAM.
        
        Args:
            x: Input tensor of shape (batch, n_channels, n_timepoints)
            
        Returns:
            Probability of clean epoch
        """
        # Spatial filtering
        x = self.spatial_filter(x)
        self.activations['spatial'] = x.detach()
        
        # Temporal processing
        x = self.temporal_block1(x)
        self.activations['temporal1'] = x.detach()
        
        x = self.temporal_block2(x)
        self.activations['temporal2'] = x.detach()
        
        x = self.temporal_block3(x)
        self.activations['temporal3'] = x.detach()
        
        # Classification
        x = self.classifier(x)
        
        return x


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def get_device() -> torch.device:
    """
    Get the best available device (GPU if available, else CPU).
    
    Returns:
        torch.device object
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
        print("GPU not available, using CPU")
    return device


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for imbalanced classification.
    
    Uses inverse frequency weighting to handle class imbalance.
    
    Args:
        y: Array of binary labels (0=artifact, 1=clean)
        
    Returns:
        Tensor of class weights [weight_for_0, weight_for_1]
    """
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    
    # Compute inverse frequency weights
    weights = n_samples / (n_classes * counts)
    
    # Normalize weights
    weights = weights / weights.sum() * n_classes
    
    print(f"Class distribution: {dict(zip(classes, counts))}")
    print(f"Class weights: artifact(0)={weights[0]:.3f}, clean(1)={weights[1]:.3f}")
    
    return torch.tensor(weights, dtype=torch.float32)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    patience: int = 8,
    checkpoint_path: str = "models/best_cnn1d.pt"
) -> Dict[str, list]:
    """
    Train the EEG artifact classification model.
    
    Uses:
    - BCELoss with class weights
    - Adam optimizer
    - CosineAnnealingLR scheduler
    - Early stopping on validation F1 score
    
    Args:
        model: The EEGArtifactCNN model
        X_train: Training data of shape (n_samples, n_channels, n_timepoints)
        y_train: Training labels (0=artifact, 1=clean)
        X_val: Validation data
        y_val: Validation labels
        epochs: Maximum number of training epochs
        lr: Learning rate
        batch_size: Batch size for training
        patience: Early stopping patience (epochs without improvement)
        checkpoint_path: Path to save best model checkpoint
        
    Returns:
        Dictionary containing training history:
        - train_loss: Training loss per epoch
        - val_loss: Validation loss per epoch
        - val_f1: Validation F1 score per epoch
        - val_auc: Validation AUC per epoch
    """
    device = get_device()
    model = model.to(device)
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(y_train)
    class_weights = class_weights.to(device)
    
    # Loss function with class weights
    # Weight for class i is applied to samples of class i
    pos_weight = class_weights[1] / class_weights[0] if class_weights[1] > class_weights[0] else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(device))
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_auc': []
    }
    
    # Early stopping
    best_f1 = 0.0
    epochs_without_improvement = 0
    
    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    print(f"\nStarting training for up to {epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (use raw logits for BCEWithLogitsLoss)
            # Temporarily bypass sigmoid for loss calculation
            logits = model.spatial_filter(batch_X)
            logits = model.temporal_block1(logits)
            logits = model.temporal_block2(logits)
            logits = model.temporal_block3(logits)
            logits = logits.flatten(1)
            logits = model.classifier[1](logits)  # Dropout
            logits = model.classifier[2](logits)  # Linear(256, 64)
            logits = model.classifier[3](logits)  # GELU
            logits = model.classifier[4](logits)  # Linear(64, 1) - raw logits
            
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            # Get predictions
            val_probs = model(X_val_tensor)
            val_logits = model.spatial_filter(X_val_tensor)
            val_logits = model.temporal_block1(val_logits)
            val_logits = model.temporal_block2(val_logits)
            val_logits = model.temporal_block3(val_logits)
            val_logits = val_logits.flatten(1)
            val_logits = model.classifier[1](val_logits)
            val_logits = model.classifier[2](val_logits)
            val_logits = model.classifier[3](val_logits)
            val_logits = model.classifier[4](val_logits)
            
            val_loss = criterion(val_logits, y_val_tensor).item()
            
            # Convert to numpy for metrics
            y_val_np = y_val_tensor.cpu().numpy()
            y_pred_np = val_probs.cpu().numpy()
            y_pred_binary = (y_pred_np >= 0.5).astype(int)
            
            val_f1 = f1_score(y_val_np, y_pred_binary, zero_division=0)
            try:
                val_auc = roc_auc_score(y_val_np, y_pred_np)
            except ValueError:
                val_auc = 0.5  # Only one class present
        
        # Update learning rate
        scheduler.step()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"Val AUC: {val_auc:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # Early stopping on F1
        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0
            
            # Save best checkpoint
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_auc': val_auc,
                'history': history
            }, checkpoint_path)
            print(f"  → New best F1: {val_f1:.4f} - Checkpoint saved to {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs "
                  f"(no F1 improvement for {patience} epochs)")
            break
    
    print("-" * 60)
    print(f"Training complete. Best Val F1: {best_f1:.4f}")
    
    # Load best model before returning
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return history


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate the trained model on test data.
    
    Computes comprehensive metrics including:
    - F1 score
    - AUC-ROC
    - Precision
    - Recall
    - False Rejection Rate (FRR): % of clean epochs incorrectly rejected
    - Confusion Matrix
    
    Args:
        model: Trained EEGArtifactCNN model
        X_test: Test data of shape (n_samples, n_channels, n_timepoints)
        y_test: Test labels (0=artifact, 1=clean)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    device = get_device()
    model = model.to(device)
    model.eval()
    
    # Convert to tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)
    
    with torch.no_grad():
        # Get predictions
        y_pred_probs = model(X_test_tensor).cpu().numpy()
    
    y_test_np = y_test.reshape(-1, 1)
    y_pred_binary = (y_pred_probs >= 0.5).astype(int)
    
    # Compute metrics
    f1 = f1_score(y_test_np, y_pred_binary, zero_division=0)
    precision = precision_score(y_test_np, y_pred_binary, zero_division=0)
    recall = recall_score(y_test_np, y_pred_binary, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test_np, y_pred_probs)
    except ValueError:
        auc = 0.5
    
    # Confusion matrix
    cm = confusion_matrix(y_test_np, y_pred_binary)
    
    # False Rejection Rate (FRR): clean epochs incorrectly classified as artifacts
    # FRR = FN / (FN + TP) = clean epochs rejected / total clean epochs
    tn, fp, fn, tp = cm.ravel()
    
    # FRR: proportion of actual clean (1) that were predicted as artifact (0)
    total_clean = tp + fn  # Actual clean epochs
    false_rejections = fn   # Clean epochs predicted as artifact
    frr = false_rejections / total_clean if total_clean > 0 else 0.0
    
    # Also compute False Acceptance Rate (FAR): artifacts accepted as clean
    total_artifacts = tn + fp  # Actual artifact epochs
    false_acceptances = fp      # Artifacts predicted as clean
    far = false_acceptances / total_artifacts if total_artifacts > 0 else 0.0
    
    results = {
        'f1': float(f1),
        'auc': float(auc),
        'precision': float(precision),
        'recall': float(recall),
        'false_rejection_rate': float(frr),
        'false_acceptance_rate': float(far),
        'confusion_matrix': cm.tolist(),
        'n_test_samples': len(y_test),
        'n_artifacts': int(total_artifacts),
        'n_clean': int(total_clean)
    }
    
    return results


def print_evaluation_table(results: Dict[str, float]) -> None:
    """
    Print evaluation results in a formatted table.
    
    Args:
        results: Dictionary from evaluate_model()
    """
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"\n{'Metric':<30} {'Value':>20}")
    print("-" * 50)
    print(f"{'F1 Score':<30} {results['f1']:>20.4f}")
    print(f"{'AUC-ROC':<30} {results['auc']:>20.4f}")
    print(f"{'Precision':<30} {results['precision']:>20.4f}")
    print(f"{'Recall (Sensitivity)':<30} {results['recall']:>20.4f}")
    print(f"{'False Rejection Rate':<30} {results['false_rejection_rate']:>20.4f}")
    print(f"{'False Acceptance Rate':<30} {results['false_acceptance_rate']:>20.4f}")
    print("-" * 50)
    print(f"{'Test Samples':<30} {results['n_test_samples']:>20d}")
    print(f"{'Artifact Epochs':<30} {results['n_artifacts']:>20d}")
    print(f"{'Clean Epochs':<30} {results['n_clean']:>20d}")
    
    # Print confusion matrix
    cm = np.array(results['confusion_matrix'])
    print("\nConfusion Matrix:")
    print("                 Predicted")
    print("                 Artifact  Clean")
    print(f"  Actual Artifact  {cm[0,0]:6d}   {cm[0,1]:6d}")
    print(f"  Actual Clean     {cm[1,0]:6d}   {cm[1,1]:6d}")
    print("=" * 60)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_training_history(
    history: Dict[str, list],
    save_path: str = "outputs/cnn_training.png"
) -> plt.Figure:
    """
    Plot training history metrics.
    
    Creates a 2x2 subplot showing:
    - Training and validation loss
    - Validation F1 score
    - Validation AUC
    - Learning rate schedule (if available)
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot 1: Loss curves
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: F1 Score
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['val_f1'], 'g-', label='Val F1', linewidth=2, marker='o', markersize=4)
    best_f1_idx = np.argmax(history['val_f1'])
    ax2.axhline(y=history['val_f1'][best_f1_idx], color='g', linestyle='--', alpha=0.5)
    ax2.scatter([best_f1_idx + 1], [history['val_f1'][best_f1_idx]], 
                color='red', s=100, zorder=5, label=f'Best: {history["val_f1"][best_f1_idx]:.4f}')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('F1 Score', fontsize=12)
    ax2.set_title('Validation F1 Score', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    # Plot 3: AUC
    ax3 = axes[1, 0]
    ax3.plot(epochs, history['val_auc'], 'm-', label='Val AUC', linewidth=2, marker='s', markersize=4)
    best_auc_idx = np.argmax(history['val_auc'])
    ax3.axhline(y=history['val_auc'][best_auc_idx], color='m', linestyle='--', alpha=0.5)
    ax3.scatter([best_auc_idx + 1], [history['val_auc'][best_auc_idx]], 
                color='red', s=100, zorder=5, label=f'Best: {history["val_auc"][best_auc_idx]:.4f}')
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('AUC-ROC', fontsize=12)
    ax3.set_title('Validation AUC-ROC', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # Plot 4: Combined metrics
    ax4 = axes[1, 1]
    ax4.plot(epochs, history['val_f1'], 'g-', label='F1', linewidth=2)
    ax4.plot(epochs, history['val_auc'], 'm-', label='AUC', linewidth=2)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Score', fontsize=12)
    ax4.set_title('F1 vs AUC Comparison', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nTraining history plot saved to: {save_path}")
    
    return fig


# =============================================================================
# INTERPRETABILITY: GRAD-CAM
# =============================================================================

def grad_cam_eeg(
    model: nn.Module,
    X_sample: np.ndarray,
    target_layer: str = 'spatial'
) -> np.ndarray:
    """
    Compute Grad-CAM channel importance weights for EEG data.
    
    This identifies which channels are most important for the model's
    artifact detection decision.
    
    Args:
        model: Trained EEGArtifactCNN model
        X_sample: Single sample of shape (n_channels, n_timepoints) or (1, n_channels, n_timepoints)
        target_layer: Which layer to compute Grad-CAM for ('spatial', 'temporal1', 'temporal2', 'temporal3')
        
    Returns:
        Array of channel importance weights (n_channels,)
    """
    device = get_device()
    model = model.to(device)
    model.eval()
    
    # Ensure correct shape
    if X_sample.ndim == 2:
        X_sample = X_sample[np.newaxis, ...]  # Add batch dimension
    
    X_tensor = torch.tensor(X_sample, dtype=torch.float32, requires_grad=True).to(device)
    
    # Forward pass with hooks
    gradients = {}
    activations = {}
    
    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]
    
    def forward_hook(module, input, output):
        activations['value'] = output
    
    # Register hooks based on target layer
    if target_layer == 'spatial':
        target_module = model.spatial_filter
    elif target_layer == 'temporal1':
        target_module = model.temporal_block1[0]  # First Conv1d in block
    elif target_layer == 'temporal2':
        target_module = model.temporal_block2[0]
    elif target_layer == 'temporal3':
        target_module = model.temporal_block3[0]
    else:
        raise ValueError(f"Unknown target layer: {target_layer}")
    
    forward_handle = target_module.register_forward_hook(forward_hook)
    backward_handle = target_module.register_full_backward_hook(backward_hook)
    
    try:
        # Forward pass
        output = model(X_tensor)
        
        # Backward pass to get gradients
        model.zero_grad()
        output.backward()
        
        # Get gradients and activations
        grads = gradients['value'].detach().cpu().numpy()
        acts = activations['value'].detach().cpu().numpy()
        
        # Compute Grad-CAM weights
        # Global average pooling of gradients
        weights = np.mean(grads, axis=(0, 2))  # Average over batch and time
        
        # For spatial layer, weights directly correspond to input channels
        if target_layer == 'spatial':
            # Weight shape is (32,) - one weight per output channel
            # We need to map this back to input channels
            # Get the spatial filter weights
            spatial_weights = model.spatial_filter.weight.detach().cpu().numpy()
            # spatial_weights shape: (32, 19, 1)
            # Combine with Grad-CAM weights
            channel_importance = np.abs(spatial_weights.squeeze(-1).T @ weights)
        else:
            # For temporal layers, we get channel importance for the layer's output
            # This is already computed in weights
            channel_importance = weights
        
        # Normalize
        channel_importance = channel_importance / (channel_importance.sum() + 1e-8)
        
        return channel_importance
        
    finally:
        # Remove hooks
        forward_handle.remove()
        backward_handle.remove()


def visualize_channel_importance(
    model: nn.Module,
    X_sample: np.ndarray,
    channel_names: Optional[list] = None,
    save_path: str = "outputs/channel_importance.png"
) -> plt.Figure:
    """
    Visualize channel importance from Grad-CAM analysis.
    
    Args:
        model: Trained EEGArtifactCNN model
        X_sample: Single EEG sample
        channel_names: List of channel names (default: standard 10-20 names)
        save_path: Path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    # Default 10-20 channel names for 19-channel EEG
    if channel_names is None:
        channel_names = [
            'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
            'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 
            'Fz', 'Cz', 'Pz'
        ]
    
    # Compute channel importance
    importance = grad_cam_eeg(model, X_sample, target_layer='spatial')
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by importance
    sorted_idx = np.argsort(importance)[::-1]
    sorted_names = [channel_names[i] for i in sorted_idx]
    sorted_importance = importance[sorted_idx]
    
    # Bar plot
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(importance)))[::-1]
    bars = ax.barh(range(len(sorted_importance)), sorted_importance, color=colors)
    
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names)
    ax.set_xlabel('Importance Weight', fontsize=12)
    ax.set_title('EEG Channel Importance for Artifact Detection (Grad-CAM)', 
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, sorted_importance)):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nChannel importance plot saved to: {save_path}")
    
    return fig


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(data_path: str = "data/augmented.npz") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load TMS-EEG data from numpy file.
    
    Args:
        data_path: Path to the .npz file containing X and y arrays
        
    Returns:
        Tuple of (X, y) where X is (n_samples, n_channels, n_timepoints)
        and y is (n_samples,) binary labels
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Please ensure the augmented data file exists."
        )
    
    data = np.load(data_path)
    
    # Handle different possible key names
    if 'X' in data and 'y' in data:
        X, y = data['X'], data['y']
    elif 'data' in data and 'labels' in data:
        X, y = data['data'], data['labels']
    elif 'X_train' in data and 'y_train' in data:
        # If data is already split, just use train
        X, y = data['X_train'], data['y_train']
    else:
        # Try to infer keys
        keys = list(data.keys())
        X = data[keys[0]]
        y = data[keys[1]]
    
    print(f"Loaded data from {data_path}")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Label distribution: 0 (artifact)={np.sum(y==0)}, 1 (clean)={np.sum(y==1)}")
    
    return X, y


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets with stratification.
    
    Args:
        X: Input data
        y: Labels
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    from sklearn.model_selection import train_test_split
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1"
    
    # First split: train vs (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1-train_ratio), random_state=random_state, stratify=y
    )
    
    # Second split: val vs test
    val_test_ratio = test_ratio / (val_ratio + test_ratio)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_test_ratio, random_state=random_state, stratify=y_temp
    )
    
    print(f"\nData split:")
    print(f"  Train: {X_train.shape[0]} samples")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main training and evaluation pipeline.
    
    Steps:
    1. Load data from data/augmented.npz
    2. Split into train/val/test sets
    3. Initialize model
    4. Train model with early stopping
    5. Evaluate on test set
    6. Save training history plot
    7. Compute Grad-CAM channel importance
    """
    print("=" * 60)
    print("TMS-EEG Artifact Rejection with 1D-CNN")
    print("=" * 60)
    
    # Paths
    data_path = "data/augmented.npz"
    checkpoint_path = "models/best_cnn1d.pt"
    plot_path = "outputs/cnn_training.png"
    channel_importance_path = "outputs/channel_importance.png"
    
    # Check for data file
    if not os.path.exists(data_path):
        # Try alternative paths
        alt_paths = [
            "/home/z/my-project/data/augmented.npz",
            "./data/augmented.npz",
            "../data/augmented.npz"
        ]
        for alt_path in alt_paths:
            if os.path.exists(alt_path):
                data_path = alt_path
                break
        else:
            print(f"\nWarning: Data file not found at expected locations.")
            print("Creating synthetic data for demonstration...")
            
            # Create synthetic data for demonstration
            np.random.seed(42)
            n_samples = 1000
            n_channels = 19
            n_timepoints = 701
            
            # Generate synthetic EEG-like data
            X = np.random.randn(n_samples, n_channels, n_timepoints).astype(np.float32)
            
            # Add some structure (simulate artifacts with higher amplitude)
            artifact_idx = np.random.choice(n_samples, size=n_samples//2, replace=False)
            for idx in artifact_idx:
                # Add high-amplitude noise to simulate artifacts
                X[idx] += np.random.randn(n_channels, n_timepoints) * 5
            
            # Labels: 0 = artifact, 1 = clean
            y = np.ones(n_samples, dtype=np.float32)
            y[artifact_idx] = 0
            
            # Save synthetic data
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            np.savez(data_path, X=X, y=y)
            print(f"Created synthetic data: {X.shape}, labels: {y.shape}")
    
    # Load data
    print("\n[1/6] Loading data...")
    X, y = load_data(data_path)
    
    # Ensure correct data type and shape
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    if X.ndim == 2:
        # If data is (n_samples, n_features), need to check expected shape
        if X.shape[1] == 19 * 701:
            X = X.reshape(-1, 19, 701)
        else:
            raise ValueError(f"Unexpected data shape: {X.shape}")
    
    # Split data
    print("\n[2/6] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # Initialize model
    print("\n[3/6] Initializing model...")
    n_channels = X.shape[1]
    n_timepoints = X.shape[2]
    model = EEGArtifactCNN(n_channels=n_channels, n_timepoints=n_timepoints)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\n[4/6] Training model...")
    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=50,
        lr=1e-3,
        batch_size=32,
        patience=8,
        checkpoint_path=checkpoint_path
    )
    
    # Plot training history
    print("\n[5/6] Plotting training history...")
    plot_training_history(history, save_path=plot_path)
    
    # Load best model for evaluation
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    print("\n[6/6] Evaluating on test set...")
    results = evaluate_model(model, X_test, y_test)
    print_evaluation_table(results)
    
    # Channel importance analysis
    print("\n" + "=" * 60)
    print("CHANNEL IMPORTANCE ANALYSIS (Grad-CAM)")
    print("=" * 60)
    
    # Get a sample for Grad-CAM analysis
    sample_idx = 0
    X_sample = X_test[sample_idx]
    true_label = y_test[sample_idx]
    
    # Get model prediction
    model.eval()
    with torch.no_grad():
        pred_prob = model(torch.tensor(X_sample[np.newaxis, ...], dtype=torch.float32)).item()
    
    print(f"\nSample {sample_idx}:")
    print(f"  True label: {'Clean' if true_label == 1 else 'Artifact'}")
    print(f"  Predicted probability (clean): {pred_prob:.4f}")
    
    # Visualize channel importance
    visualize_channel_importance(model, X_sample, save_path=channel_importance_path)
    
    # Print top important channels
    importance = grad_cam_eeg(model, X_sample, target_layer='spatial')
    channel_names = [
        'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 
        'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 
        'Fz', 'Cz', 'Pz'
    ]
    sorted_idx = np.argsort(importance)[::-1]
    
    print("\nTop 5 most important channels:")
    for i, idx in enumerate(sorted_idx[:5]):
        print(f"  {i+1}. {channel_names[idx]}: {importance[idx]:.4f}")
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nOutputs saved:")
    print(f"  - Model checkpoint: {checkpoint_path}")
    print(f"  - Training history: {plot_path}")
    print(f"  - Channel importance: {channel_importance_path}")
    
    return model, history, results


if __name__ == "__main__":
    # Suppress warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Run main pipeline
    model, history, results = main()
