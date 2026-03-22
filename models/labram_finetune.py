"""
LaBraM-Style Transformer for TMS-EEG Artifact Detection

This module implements a LaBraM-inspired architecture for finetuning on
TMS-EEG artifact detection. LaBraM (Large Brain Model) is a foundation
model pretrained on 80,000+ channel-hours of EEG data.

Architecture:
1. EEGPatchEmbedding: Splits EEG signals into patches and projects to embeddings
2. EEGTransformerEncoder: Transformer backbone with CLS token (like BERT)
3. LaBraMFinetune: Complete model with finetuning head

Finetuning Protocol (3-phase):
- Phase 1 (epochs 1-10): Only head trainable, lr=1e-3
- Phase 2 (epochs 11-30): Unfreeze last transformer layer, lr=3e-4
- Phase 3 (epochs 31-50): Unfreeze all, lr=1e-4

Reference: LaBraM: A Foundation Model for EEG (2024)

Author: TMS-EEG Signal Processing Pipeline
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import warnings


# =============================================================================
# STEP 1: Patch Embedding Layer
# =============================================================================

class EEGPatchEmbedding(nn.Module):
    """
    Patch embedding for EEG signals.
    
    Splits the multi-channel EEG signal into non-overlapping patches
    and projects each patch to a learned embedding space.
    
    For TMS-EEG data with shape (batch, 19, 701):
    - Patch size: 50 timepoints
    - Number of patches: ceil(701/50) = 15
    - Each patch: 19 channels × 50 timepoints flattened to 950 → projected to 256
    
    Parameters
    ----------
    n_channels : int
        Number of EEG channels. Default: 19.
    patch_size : int
        Size of each patch in timepoints. Default: 50.
    embed_dim : int
        Embedding dimension. Default: 256.
    n_times : int
        Total number of timepoints. Default: 701.
    """
    
    def __init__(
        self,
        n_channels: int = 19,
        patch_size: int = 50,
        embed_dim: int = 256,
        n_times: int = 701
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_times = n_times
        
        # Calculate number of patches (ceil division)
        self.n_patches = math.ceil(n_times / patch_size)
        
        # Linear projection: (n_channels * patch_size) -> embed_dim
        # Flattens all channels within each patch
        self.patch_projection = nn.Linear(n_channels * patch_size, embed_dim)
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.n_patches, embed_dim)
        )
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # CLS token (like BERT)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for patch embedding.
        
        Parameters
        ----------
        x : torch.Tensor
            Input EEG data of shape (batch, n_channels, n_times).
        
        Returns
        -------
        embeddings : torch.Tensor
            Patch embeddings of shape (batch, n_patches + 1, embed_dim).
            The +1 is for the CLS token prepended to the sequence.
        """
        batch_size = x.shape[0]
        
        # Pad input if necessary to make it divisible by patch_size
        if self.n_times % self.patch_size != 0:
            pad_size = self.n_patches * self.patch_size - self.n_times
            x = F.pad(x, (0, pad_size))  # Pad on the right (time dimension)
        
        # Reshape into patches: (batch, n_channels, n_patches, patch_size)
        x = x.unfold(dimension=2, size=self.patch_size, step=self.patch_size)
        
        # Flatten channels and patch: (batch, n_patches, n_channels * patch_size)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, self.n_patches, -1)
        
        # Project patches to embedding dimension
        x = self.patch_projection(x)  # (batch, n_patches, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embedding
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, n_patches + 1, embed_dim)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x


# =============================================================================
# STEP 2: Transformer Encoder
# =============================================================================

class EEGTransformerEncoder(nn.Module):
    """
    Transformer encoder for EEG with CLS token extraction.
    
    Uses standard transformer encoder layers with multi-head self-attention.
    The CLS token representation is extracted as the output for downstream tasks.
    
    Parameters
    ----------
    embed_dim : int
        Embedding dimension. Default: 256.
    n_heads : int
        Number of attention heads. Default: 8.
    n_layers : int
        Number of transformer layers. Default: 4.
    ff_dim : int
        Feed-forward network dimension. Default: 512.
    dropout : float
        Dropout rate. Default: 0.1.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better training stability
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(embed_dim)
        )
        
        # Store attention weights for visualization
        self.attention_weights = []
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through transformer encoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input embeddings of shape (batch, n_patches + 1, embed_dim).
        return_attention : bool
            Whether to return attention weights. Default: False.
        
        Returns
        -------
        output : torch.Tensor
            CLS token representation of shape (batch, embed_dim).
            If return_attention=True, also returns attention weights.
        """
        # Clear stored attention weights
        self.attention_weights = []
        
        # Register hook to capture attention weights
        if return_attention:
            hooks = []
            for layer_idx, layer in enumerate(self.encoder.layers):
                def make_hook(idx):
                    def hook(module, input, output):
                        # For TransformerEncoderLayer, we need to capture attention manually
                        pass
                    return hook
                hooks.append(layer.register_forward_hook(make_hook(layer_idx)))
        
        # Pass through encoder
        encoded = self.encoder(x)
        
        # Extract CLS token (first token)
        cls_output = encoded[:, 0, :]  # (batch, embed_dim)
        
        # Remove hooks
        if return_attention:
            for hook in hooks:
                hook.remove()
        
        if return_attention:
            return cls_output, self.attention_weights
        
        return cls_output
    
    def get_attention_weights(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract attention weights from all layers.
        
        This method manually computes attention weights for visualization
        since PyTorch's TransformerEncoder doesn't return them by default.
        
        Parameters
        ----------
        x : torch.Tensor
            Input embeddings of shape (batch, n_patches + 1, embed_dim).
        
        Returns
        -------
        attention_weights : torch.Tensor
            Attention weights of shape (n_layers, batch, n_heads, n_patches+1, n_patches+1).
        """
        all_attention_weights = []
        
        for layer in self.encoder.layers:
            # Get query, key, value projections
            # In TransformerEncoderLayer, self_attn is MultiheadAttention
            self_attn = layer.self_attn
            
            # Compute attention manually
            # MultiheadAttention expects: query, key, value
            q = k = v = x
            
            # Get weights using the attention module
            with torch.no_grad():
                attn_output, attn_weights = self_attn(
                    q, k, v,
                    need_weights=True,
                    average_attn_weights=False  # Get per-head weights
                )
            
            all_attention_weights.append(attn_weights)
            
            # Continue forward pass through this layer
            # Apply attention output
            x = x + layer.dropout1(attn_output)
            x = layer.norm1(x)
            
            # Feed-forward network
            ff_output = layer.linear2(
                layer.dropout(layer.activation(layer.linear1(x)))
            )
            x = x + layer.dropout2(ff_output)
            x = layer.norm2(x)
        
        # Stack all attention weights: (n_layers, batch, n_heads, seq_len, seq_len)
        attention_weights = torch.stack(all_attention_weights, dim=0)
        
        return attention_weights


# =============================================================================
# STEP 3: Complete Finetuning Model
# =============================================================================

class LaBraMFinetune(nn.Module):
    """
    LaBraM-style model for TMS-EEG artifact detection finetuning.
    
    Combines patch embedding, transformer encoder, and a classification head.
    Supports layer freezing for gradual finetuning.
    
    Parameters
    ----------
    n_channels : int
        Number of EEG channels. Default: 19.
    n_times : int
        Number of timepoints. Default: 701.
    patch_size : int
        Size of each patch. Default: 50.
    embed_dim : int
        Embedding dimension. Default: 256.
    n_heads : int
        Number of attention heads. Default: 8.
    n_layers : int
        Number of transformer layers. Default: 4.
    ff_dim : int
        Feed-forward dimension. Default: 512.
    dropout : float
        Dropout rate. Default: 0.1.
    """
    
    def __init__(
        self,
        n_channels: int = 19,
        n_times: int = 701,
        patch_size: int = 50,
        embed_dim: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        ff_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_times = n_times
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Patch embedding layer
        self.patch_embed = EEGPatchEmbedding(
            n_channels=n_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            n_times=n_times
        )
        
        # Transformer encoder
        self.encoder = EEGTransformerEncoder(
            embed_dim=embed_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_dim=ff_dim,
            dropout=dropout
        )
        
        # Classification head for finetuning
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize classification head
        self._init_head()
    
    def _init_head(self):
        """Initialize classification head weights."""
        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def freeze_encoder_layers(self, n_layers: int):
        """
        Freeze first n transformer encoder layers.
        
        Parameters
        ----------
        n_layers : int
            Number of layers to freeze (from the start).
        """
        # Freeze patch embedding
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        
        # Freeze specified transformer layers
        for idx, layer in enumerate(self.encoder.encoder.layers):
            for param in layer.parameters():
                param.requires_grad = idx >= n_layers
        
        # Keep encoder norm unfrozen
        for param in self.encoder.encoder.norm.parameters():
            param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all layers for full finetuning."""
        for param in self.parameters():
            param.requires_grad = True
    
    def freeze_head_only(self):
        """Freeze all layers except the classification head."""
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        for param in self.head.parameters():
            param.requires_grad = True
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Parameters
        ----------
        x : torch.Tensor
            Input EEG data of shape (batch, n_channels, n_times).
        return_attention : bool
            Whether to return attention weights. Default: False.
        
        Returns
        -------
        output : torch.Tensor
            Predicted probabilities of shape (batch, 1).
        """
        # Patch embedding
        x = self.patch_embed(x)  # (batch, n_patches + 1, embed_dim)
        
        # Transformer encoding
        if return_attention:
            cls_output, attention = self.encoder(x, return_attention=True)
        else:
            cls_output = self.encoder(x)
        
        # Classification head
        output = self.head(cls_output)
        
        if return_attention:
            return output, attention
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for visualization.
        
        Parameters
        ----------
        x : torch.Tensor
            Input EEG data of shape (batch, n_channels, n_times).
        
        Returns
        -------
        attention_weights : torch.Tensor
            Attention weights of shape (n_layers, batch, n_heads, seq_len, seq_len).
        """
        # Patch embedding
        x = self.patch_embed(x)
        
        # Get attention weights
        attention = self.encoder.get_attention_weights(x)
        
        return attention


# =============================================================================
# STEP 4: Finetuning Strategy
# =============================================================================

class FinetuneTrainer:
    """
    Trainer for LaBraM finetuning with 3-phase protocol.
    
    Phase 1 (epochs 1-10): Only classification head trainable, lr=1e-3
    Phase 2 (epochs 11-30): Last transformer layer unfrozen, lr=3e-4
    Phase 3 (epochs 31-50): All layers trainable, lr=1e-4
    
    Parameters
    ----------
    model : LaBraMFinetune
        Model to finetune.
    device : torch.device
        Device for training.
    freeze_layers : int
        Number of transformer layers to freeze in phase 1. Default: 2.
    """
    
    def __init__(
        self,
        model: LaBraMFinetune,
        device: torch.device,
        freeze_layers: int = 2
    ):
        self.model = model.to(device)
        self.device = device
        self.freeze_layers = freeze_layers
        
        # Phase boundaries
        self.phase1_end = 10
        self.phase2_end = 30
        self.total_epochs = 50
        
        # Learning rates for each phase
        self.lr_phase1 = 1e-3
        self.lr_phase2 = 3e-4
        self.lr_phase3 = 1e-4
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'lr': [],
            'phase': []
        }
    
    def _get_current_phase(self, epoch: int) -> int:
        """Determine current training phase based on epoch."""
        if epoch < self.phase1_end:
            return 1
        elif epoch < self.phase2_end:
            return 2
        else:
            return 3
    
    def _configure_phase(self, phase: int) -> Tuple[float, bool]:
        """
        Configure model for a specific training phase.
        
        Returns
        -------
        lr : float
            Learning rate for this phase.
        needs_optimizer : bool
            Whether optimizer needs to be recreated.
        """
        needs_optimizer = False
        
        if phase == 1:
            # Freeze all encoder layers, only train head
            self.model.freeze_head_only()
            lr = self.lr_phase1
            needs_optimizer = True
            
        elif phase == 2:
            # Unfreeze last transformer layers
            self.model.freeze_encoder_layers(self.freeze_layers)
            lr = self.lr_phase2
            needs_optimizer = True
            
        else:  # phase == 3
            # Unfreeze all layers
            self.model.unfreeze_all()
            lr = self.lr_phase3
            needs_optimizer = True
        
        return lr, needs_optimizer
    
    def _train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.float().to(self.device)
            
            optimizer.zero_grad()
            
            outputs = self.model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            predictions = (outputs.squeeze() > 0.5).float()
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def _validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.float().to(self.device)
            
            outputs = self.model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            predictions = (outputs.squeeze() > 0.5).float()
            correct += (predictions == batch_y).sum().item()
            total += batch_y.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def finetune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        batch_size: int = 32,
        verbose: bool = True
    ) -> Dict[str, List]:
        """
        Execute the 3-phase finetuning protocol.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training data of shape (n_samples, n_channels, n_times).
        y_train : np.ndarray
            Training labels.
        X_val : np.ndarray
            Validation data.
        y_val : np.ndarray
            Validation labels.
        batch_size : int
            Batch size for training. Default: 32.
        verbose : bool
            Whether to print progress. Default: True.
        
        Returns
        -------
        history : dict
            Training history with losses, accuracies, and learning rates.
        """
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.LongTensor(y_val)
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )
        
        # Loss function
        criterion = nn.BCELoss()
        
        # Training state
        current_phase = 0
        optimizer = None
        
        if verbose:
            print("=" * 60)
            print("LaBraM Finetuning - 3-Phase Protocol")
            print("=" * 60)
        
        for epoch in range(self.total_epochs):
            # Check for phase transition
            phase = self._get_current_phase(epoch)
            
            if phase != current_phase:
                current_phase = phase
                lr, needs_optimizer = self._configure_phase(phase)
                
                if needs_optimizer:
                    optimizer = AdamW(
                        self.model.parameters(),
                        lr=lr,
                        weight_decay=0.01
                    )
                
                if verbose:
                    phase_names = {1: "Head-only", 2: "Partial", 3: "Full"}
                    print(f"\n>>> Phase {phase}: {phase_names[phase]} finetuning (lr={lr})")
            
            # Train and validate
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self._validate(val_loader, criterion)
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            self.history['phase'].append(phase)
            
            if verbose and (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1:3d}: "
                      f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                      f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"Finetuning complete. Best Val Acc: {max(self.history['val_acc']):.4f}")
            print("=" * 60)
        
        return self.history
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Parameters
        ----------
        X_test : np.ndarray
            Test data.
        y_test : np.ndarray
            Test labels.
        
        Returns
        -------
        metrics : dict
            Evaluation metrics including accuracy, precision, recall, F1.
        """
        self.model.eval()
        
        X_tensor = torch.FloatTensor(X_test).to(self.device)
        y_tensor = torch.LongTensor(y_test).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X_tensor).squeeze()
            predictions = (outputs > 0.5).float()
            
            # Calculate metrics
            tp = ((predictions == 1) & (y_tensor == 1)).sum().item()
            tn = ((predictions == 0) & (y_tensor == 0)).sum().item()
            fp = ((predictions == 1) & (y_tensor == 0)).sum().item()
            fn = ((predictions == 0) & (y_tensor == 1)).sum().item()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics


# =============================================================================
# STEP 5: Attention Visualization
# =============================================================================

def get_attention_weights(
    model: LaBraMFinetune,
    X_sample: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Extract attention weights from the model.
    
    Parameters
    ----------
    model : LaBraMFinetune
        Trained model.
    X_sample : torch.Tensor
        Input sample of shape (batch, n_channels, n_times).
    device : torch.device
        Device for computation.
    
    Returns
    -------
    attention_weights : np.ndarray
        Attention weights of shape (n_layers, n_heads, n_patches+1, n_patches+1).
    """
    model.eval()
    X_sample = X_sample.to(device)
    
    with torch.no_grad():
        attention = model.get_attention_weights(X_sample)
    
    # Convert to numpy: (n_layers, batch, n_heads, seq_len, seq_len)
    attention_np = attention.cpu().numpy()
    
    # Average over batch dimension
    attention_np = attention_np.mean(axis=1)
    
    return attention_np


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    layer_idx: int = -1,
    head_idx: int = 0,
    patch_times: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Plot attention heatmap for visualization.
    
    Parameters
    ----------
    attention_weights : np.ndarray
        Attention weights of shape (n_layers, n_heads, seq_len, seq_len).
    layer_idx : int
        Index of layer to visualize. -1 for last layer. Default: -1.
    head_idx : int
        Index of attention head to visualize. Default: 0.
    patch_times : list, optional
        Labels for patch time ranges.
    save_path : str, optional
        Path to save the figure.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    import matplotlib.pyplot as plt
    
    n_layers, n_heads, seq_len, _ = attention_weights.shape
    
    # Get the specified layer
    if layer_idx < 0:
        layer_idx = n_layers + layer_idx
    
    attn = attention_weights[layer_idx, head_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap
    im = ax.imshow(attn, cmap='viridis', aspect='auto')
    
    # Labels
    ax.set_xlabel('Key Patches', fontsize=12)
    ax.set_ylabel('Query Patches', fontsize=12)
    ax.set_title(f'Attention Weights - Layer {layer_idx+1}, Head {head_idx+1}', fontsize=14)
    
    # Create tick labels
    if patch_times is None:
        labels = ['CLS'] + [f'P{i}' for i in range(seq_len - 1)]
    else:
        labels = ['CLS'] + patch_times
    
    # Show every other label if too many
    if seq_len > 10:
        step = 2
        tick_indices = list(range(0, seq_len, step))
        ax.set_xticks(tick_indices)
        ax.set_yticks(tick_indices)
        ax.set_xticklabels([labels[i] for i in tick_indices], rotation=45, ha='right')
        ax.set_yticklabels([labels[i] for i in tick_indices])
    else:
        ax.set_xticks(range(seq_len))
        ax.set_yticks(range(seq_len))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention heatmap to {save_path}")
    
    return fig


def plot_attention_over_time(
    attention_weights: np.ndarray,
    tmin: float = -0.2,
    tmax: float = 0.5,
    n_patches: int = 15,
    layer_idx: int = -1,
    save_path: Optional[str] = None
):
    """
    Plot CLS token attention over time patches.
    
    Shows which time segments the model attends to for classification.
    
    Parameters
    ----------
    attention_weights : np.ndarray
        Attention weights of shape (n_layers, n_heads, seq_len, seq_len).
    tmin : float
        Start time in seconds. Default: -0.2.
    tmax : float
        End time in seconds. Default: 0.5.
    n_patches : int
        Number of patches. Default: 15.
    layer_idx : int
        Layer to visualize. Default: -1 (last layer).
    save_path : str, optional
        Path to save the figure.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    import matplotlib.pyplot as plt
    
    n_layers, n_heads, seq_len, _ = attention_weights.shape
    
    if layer_idx < 0:
        layer_idx = n_layers + layer_idx
    
    # Get CLS token attention (first row), averaged over heads
    cls_attention = attention_weights[layer_idx, :, 0, 1:].mean(axis=0)  # (seq_len-1,)
    
    # Create time labels for patches
    patch_size = (tmax - tmin) / n_patches
    patch_centers = [tmin + (i + 0.5) * patch_size for i in range(n_patches)]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Plot attention over time
    bars = ax.bar(range(len(cls_attention)), cls_attention, 
                  color='steelblue', alpha=0.8)
    
    # Highlight artifact window (5-50ms)
    artifact_start_patch = int((0.005 - tmin) / patch_size)
    artifact_end_patch = int((0.050 - tmin) / patch_size)
    
    for i in range(max(0, artifact_start_patch), min(len(bars), artifact_end_patch + 1)):
        bars[i].set_color('crimson')
        bars[i].set_alpha(0.9)
    
    # Labels
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Attention Weight', fontsize=12)
    ax.set_title(f'CLS Token Attention Over Time - Layer {layer_idx+1}', fontsize=14)
    
    # Custom x-axis labels
    ax.set_xticks(range(len(cls_attention)))
    ax.set_xticklabels([f'{t:.2f}' for t in patch_centers], rotation=45, ha='right')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.8, label='Normal'),
        Patch(facecolor='crimson', alpha=0.9, label='Artifact Window (5-50ms)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention over time plot to {save_path}")
    
    return fig


# =============================================================================
# Comparison Baseline: 1D CNN
# =============================================================================

class CNN1D(nn.Module):
    """
    1D CNN baseline for TMS-EEG artifact detection.
    
    A simple convolutional network for comparison with the transformer.
    
    Parameters
    ----------
    n_channels : int
        Number of EEG channels. Default: 19.
    """
    
    def __init__(self, n_channels: int = 19):
        super().__init__()
        
        # Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # Calculate flattened size
        # Input: 701 -> pool1: 350 -> pool2: 175 -> pool3: 87
        self.flat_size = 128 * 87
        
        # Classification head
        self.head = nn.Sequential(
            nn.Linear(self.flat_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.view(x.size(0), -1)
        x = self.head(x)
        
        return x


def train_cnn(
    cnn_model: CNN1D,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    verbose: bool = True
) -> Dict[str, List]:
    """
    Train the CNN baseline.
    
    Parameters
    ----------
    cnn_model : CNN1D
        CNN model to train.
    X_train, y_train : np.ndarray
        Training data and labels.
    X_val, y_val : np.ndarray
        Validation data and labels.
    device : torch.device
        Device for training.
    epochs : int
        Number of training epochs. Default: 50.
    batch_size : int
        Batch size. Default: 32.
    lr : float
        Learning rate. Default: 1e-3.
    verbose : bool
        Print progress. Default: True.
    
    Returns
    -------
    history : dict
        Training history.
    """
    cnn_model = cnn_model.to(device)
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.LongTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.LongTensor(y_val)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    criterion = nn.BCELoss()
    optimizer = AdamW(cnn_model.parameters(), lr=lr, weight_decay=0.01)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    if verbose:
        print("\n" + "=" * 60)
        print("Training CNN1D Baseline")
        print("=" * 60)
    
    for epoch in range(epochs):
        # Train
        cnn_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.float().to(device)
            
            optimizer.zero_grad()
            outputs = cnn_model(batch_x)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_x.size(0)
            predictions = (outputs.squeeze() > 0.5).float()
            train_correct += (predictions == batch_y).sum().item()
            train_total += batch_y.size(0)
        
        # Validate
        cnn_model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.float().to(device)
                
                outputs = cnn_model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                
                val_loss += loss.item() * batch_x.size(0)
                predictions = (outputs.squeeze() > 0.5).float()
                val_correct += (predictions == batch_y).sum().item()
                val_total += batch_y.size(0)
        
        history['train_loss'].append(train_loss / train_total)
        history['val_loss'].append(val_loss / val_total)
        history['train_acc'].append(train_correct / train_total)
        history['val_acc'].append(val_correct / val_total)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}: "
                  f"Train Loss={history['train_loss'][-1]:.4f}, Acc={history['train_acc'][-1]:.4f} | "
                  f"Val Loss={history['val_loss'][-1]:.4f}, Acc={history['val_acc'][-1]:.4f}")
    
    if verbose:
        print(f"\nCNN Training complete. Best Val Acc: {max(history['val_acc']):.4f}")
    
    return history


# =============================================================================
# Main Function
# =============================================================================

def main():
    """
    Main function to finetune LaBraM on TMS-EEG data.
    
    Loads augmented data, trains both LaBraM and CNN baseline,
    compares performance, and saves results.
    """
    import matplotlib.pyplot as plt
    
    print("=" * 70)
    print("LaBraM Finetuning for TMS-EEG Artifact Detection")
    print("=" * 70)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load augmented data
    data_path = Path("/home/z/my-project/data/augmented.npz")
    if not data_path.exists():
        raise FileNotFoundError(
            f"Augmented data not found at {data_path}. "
            "Please run data/augment.py first."
        )
    
    print(f"\nLoading data from {data_path}")
    data = np.load(data_path)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Val:   {X_val.shape[0]} samples")
    print(f"Test:  {X_test.shape[0]} samples")
    
    # Create output directory
    output_dir = Path("/home/z/my-project/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ==========================================================================
    # Train LaBraM model
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Training LaBraM Transformer")
    print("=" * 70)
    
    labram_model = LaBraMFinetune(
        n_channels=19,
        n_times=701,
        patch_size=50,
        embed_dim=256,
        n_heads=8,
        n_layers=4,
        ff_dim=512,
        dropout=0.1
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in labram_model.parameters())
    trainable_params = sum(p.numel() for p in labram_model.parameters() if p.requires_grad)
    print(f"\nLaBraM Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    trainer = FinetuneTrainer(
        model=labram_model,
        device=device,
        freeze_layers=2
    )
    
    labram_history = trainer.finetune(
        X_train, y_train,
        X_val, y_val,
        batch_size=32,
        verbose=True
    )
    
    # Evaluate on test set
    labram_metrics = trainer.evaluate(X_test, y_test)
    
    print("\nLaBraM Test Metrics:")
    for metric, value in labram_metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    # ==========================================================================
    # Train CNN baseline
    # ==========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Training CNN1D Baseline")
    print("=" * 70)
    
    cnn_model = CNN1D(n_channels=19)
    
    cnn_params = sum(p.numel() for p in cnn_model.parameters())
    print(f"\nCNN Parameters: {cnn_params:,}")
    
    cnn_history = train_cnn(
        cnn_model,
        X_train, y_train,
        X_val, y_val,
        device=device,
        epochs=50,
        batch_size=32,
        lr=1e-3,
        verbose=True
    )
    
    # Evaluate CNN on test set
    cnn_model.eval()
    X_test_tensor = torch.FloatTensor(X_test).to(device)
    y_test_tensor = torch.LongTensor(y_test).to(device)
    
    with torch.no_grad():
        cnn_outputs = cnn_model(X_test_tensor).squeeze()
        cnn_predictions = (cnn_outputs > 0.5).float()
        
        cnn_tp = ((cnn_predictions == 1) & (y_test_tensor == 1)).sum().item()
        cnn_tn = ((cnn_predictions == 0) & (y_test_tensor == 0)).sum().item()
        cnn_fp = ((cnn_predictions == 1) & (y_test_tensor == 0)).sum().item()
        cnn_fn = ((cnn_predictions == 0) & (y_test_tensor == 1)).sum().item()
        
        cnn_accuracy = (cnn_tp + cnn_tn) / (cnn_tp + cnn_tn + cnn_fp + cnn_fn)
        cnn_precision = cnn_tp / (cnn_tp + cnn_fp) if (cnn_tp + cnn_fp) > 0 else 0.0
        cnn_recall = cnn_tp / (cnn_tp + cnn_fn) if (cnn_tp + cnn_fn) > 0 else 0.0
        cnn_f1 = 2 * cnn_precision * cnn_recall / (cnn_precision + cnn_recall) \
            if (cnn_precision + cnn_recall) > 0 else 0.0
    
    cnn_metrics = {
        'accuracy': cnn_accuracy,
        'precision': cnn_precision,
        'recall': cnn_recall,
        'f1': cnn_f1
    }
    
    print("\nCNN Test Metrics:")
    for metric, value in cnn_metrics.items():
        print(f"  {metric.capitalize()}: {value:.4f}")
    
    # ==========================================================================
    # Comparison
    # ==========================================================================
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    
    print("\n{:<15} {:<12} {:<12}".format("Metric", "LaBraM", "CNN1D"))
    print("-" * 40)
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        print("{:<15} {:<12.4f} {:<12.4f}".format(
            metric.capitalize(),
            labram_metrics[metric],
            cnn_metrics[metric]
        ))
    
    # ==========================================================================
    # Attention Visualization
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ATTENTION VISUALIZATION")
    print("=" * 70)
    
    # Get attention weights for a sample
    sample_idx = 0
    X_sample = torch.FloatTensor(X_test[sample_idx:sample_idx+1])
    
    attention_weights = get_attention_weights(labram_model, X_sample, device)
    print(f"Attention weights shape: {attention_weights.shape}")
    
    # Plot attention heatmap
    heatmap_path = output_dir / "attention_heatmap.png"
    plot_attention_heatmap(
        attention_weights,
        layer_idx=-1,
        head_idx=0,
        save_path=str(heatmap_path)
    )
    
    # Plot attention over time
    time_path = output_dir / "attention_over_time.png"
    plot_attention_over_time(
        attention_weights,
        tmin=-0.2,
        tmax=0.5,
        n_patches=15,
        layer_idx=-1,
        save_path=str(time_path)
    )
    
    # ==========================================================================
    # Training Curves
    # ==========================================================================
    print("\n" + "=" * 70)
    print("GENERATING TRAINING CURVES")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curves
    ax1 = axes[0]
    ax1.plot(labram_history['train_loss'], label='LaBraM Train', color='blue', alpha=0.7)
    ax1.plot(labram_history['val_loss'], label='LaBraM Val', color='blue', linestyle='--')
    ax1.plot(cnn_history['train_loss'], label='CNN Train', color='orange', alpha=0.7)
    ax1.plot(cnn_history['val_loss'], label='CNN Val', color='orange', linestyle='--')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Accuracy curves
    ax2 = axes[1]
    ax2.plot(labram_history['val_acc'], label='LaBraM Val', color='blue')
    ax2.plot(cnn_history['val_acc'], label='CNN Val', color='orange')
    ax2.axhline(y=max(labram_history['val_acc']), color='blue', linestyle=':', alpha=0.5)
    ax2.axhline(y=max(cnn_history['val_acc']), color='orange', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    curves_path = output_dir / "training_curves.png"
    fig.savefig(curves_path, dpi=150, bbox_inches='tight')
    print(f"Saved training curves to {curves_path}")
    
    plt.close('all')
    
    # ==========================================================================
    # Save Model
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    
    model_path = output_dir / "labram_finetuned.pt"
    torch.save({
        'model_state_dict': labram_model.state_dict(),
        'model_config': {
            'n_channels': 19,
            'n_times': 701,
            'patch_size': 50,
            'embed_dim': 256,
            'n_heads': 8,
            'n_layers': 4,
            'ff_dim': 512,
            'dropout': 0.1
        },
        'test_metrics': labram_metrics,
        'history': labram_history
    }, model_path)
    
    print(f"Saved finetuned model to {model_path}")
    
    # Save CNN for comparison
    cnn_path = output_dir / "cnn_baseline.pt"
    torch.save({
        'model_state_dict': cnn_model.state_dict(),
        'test_metrics': cnn_metrics,
        'history': cnn_history
    }, cnn_path)
    print(f"Saved CNN baseline to {cnn_path}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nLaBraM Architecture:")
    print(f"  - Patch size: 50 timepoints")
    print(f"  - Number of patches: 15")
    print(f"  - Embedding dimension: 256")
    print(f"  - Transformer layers: 4")
    print(f"  - Attention heads: 8")
    print(f"  - Total parameters: {total_params:,}")
    
    print(f"\nFinetuning Protocol:")
    print(f"  - Phase 1 (epochs 1-10): Head-only, lr=1e-3")
    print(f"  - Phase 2 (epochs 11-30): Partial unfreeze, lr=3e-4")
    print(f"  - Phase 3 (epochs 31-50): Full finetuning, lr=1e-4")
    
    print(f"\nFinal Test Accuracy:")
    print(f"  - LaBraM: {labram_metrics['accuracy']:.4f}")
    print(f"  - CNN1D:  {cnn_metrics['accuracy']:.4f}")
    improvement = (labram_metrics['accuracy'] - cnn_metrics['accuracy']) / cnn_metrics['accuracy'] * 100
    print(f"  - Improvement: {improvement:+.2f}%")
    
    print("\n" + "=" * 70)
    print("Pipeline Complete!")
    print("=" * 70)
    
    return labram_model, cnn_model, labram_history, cnn_history


if __name__ == "__main__":
    main()
