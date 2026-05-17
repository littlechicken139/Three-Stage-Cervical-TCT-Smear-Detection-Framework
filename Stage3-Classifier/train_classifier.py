#!/usr/bin/env python3
"""
Unified Classifier Training Script

Supports multiple classifier heads: AvgPool, AttentionPooling, SE-Adapter, MLP, GCN
Switch between different classifier heads by setting CLASSIFIER_TYPE or using --classifier_type argument.

Features:
- Multiple classifier architectures optimized for medical image classification
- Early stopping mechanism to prevent overfitting
- Overfitting detection and diagnostic report
- Comprehensive training history tracking
- Support for various regularization techniques
"""
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
import pickle
from tqdm import tqdm
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# ==================== Configuration Parameters ====================
# Training data path
FEATURE_DIR = r"Stage3-Classifier/features/train"

# Model output path
OUTPUT_DIR = r"Stage3-Classifier/models"

# Classifier type configuration
# Available options: 'AVGPOOL', 'ATTENTION', 'SE_ADAPTER', 'MLP', 'GCN'
CLASSIFIER_TYPE = 'MLP'
# ================================================================


class H5FeatureDataset(Dataset):
    """HDF5 Feature Dataset"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ==================== 1. Average Pooling Classifier Head ====================
class AvgPoolHead(nn.Module):
    """
    Average Pooling Classifier Head optimized for medical cell image classification.
    
    Uses global average pooling for feature aggregation, suitable for small sample medical data.
    """
    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout_rate=0.3):
        super(AvgPoolHead, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Feature reduction layer - reduces parameters to prevent overfitting
        self.feature_reduce = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Global average pooling - aggregates features globally
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 4, num_classes)
        )
    
    def forward(self, x):
        # Feature reduction
        x = self.feature_reduce(x)  # (batch_size, hidden_dim)
        
        # Add dimension for pooling
        x = x.unsqueeze(-1)  # (batch_size, hidden_dim, 1)
        
        # Global average pooling
        x = self.global_pool(x).squeeze(-1)  # (batch_size, hidden_dim)
        
        # Classification
        logits = self.classifier(x)
        return logits


# ==================== 2. Attention Pooling Classifier Head ====================
class AttentionPoolingHead(nn.Module):
    """
    Attention Pooling Classifier Head optimized for medical cell image classification.
    
    Uses attention mechanism to adaptively aggregate features, focusing on important regions.
    """
    def __init__(self, input_dim, num_classes, hidden_dim=256, num_heads=8, dropout_rate=0.3):
        super(AttentionPoolingHead, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_heads = num_heads
        
        # Feature preprocessing layer
        self.feature_preprocess = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Multi-head attention mechanism - models feature relationships
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout_rate, batch_first=True)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # LayerNorm layers
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        # Feature preprocessing
        x = self.feature_preprocess(x)  # (batch_size, hidden_dim)
        
        # Add sequence dimension for attention
        x = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Self-attention mechanism
        attn_output, _ = self.attention(x, x, x)  # (batch_size, 1, hidden_dim)
        x = self.norm1(x + attn_output)
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        # Remove sequence dimension
        x = x.squeeze(1)  # (batch_size, hidden_dim)
        
        # Classification
        logits = self.classifier(x)
        return logits


# ==================== 3. MLP Classifier Head ====================
class MLPHead(nn.Module):
    """
    MLP Classifier Head optimized for medical cell image classification.
    
    Uses multi-layer perceptron for non-linear feature transformation.
    """
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256, 128], dropout_rate=0.3):
        super(MLPHead, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Classification layer
        self.classifier = nn.Linear(prev_dim, num_classes)
    
    def forward(self, x):
        # Feature extraction
        features = self.feature_extractor(x)
        
        # Classification
        logits = self.classifier(features)
        return logits


# ==================== 4. SE-Adapter Classifier Head ====================
class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.channels = channels
        self.reduction = reduction
    
    def forward(self, x):
        batch_size, channels = x.size(0), x.size(1)
        y = self.global_avg_pool(x)
        y = y.view(batch_size, channels)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        y = y.view(batch_size, channels, 1)
        return x * y


class SEAdapterHead(nn.Module):
    """
    SE-Adapter Classifier Head optimized for medical cell image classification.
    
    Uses SE modules for adaptive channel feature recalibration.
    """
    def __init__(self, input_dim, num_classes, hidden_dim=128, reduction=16, dropout_rate=0.2):
        super(SEAdapterHead, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Feature preprocessing
        self.feature_preprocess = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Three SE blocks
        self.se_block1 = SEBlock(hidden_dim, reduction)
        self.se_block2 = SEBlock(hidden_dim, reduction)
        self.se_block3 = SEBlock(hidden_dim // 2, reduction)
        
        # Feature transformation and reduction
        self.feature_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.feature_reduce = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, num_classes)
        )
    
    def forward(self, x):
        # Feature preprocessing
        x = self.feature_preprocess(x)
        
        # First SE block
        x_se = x.unsqueeze(-1)
        x_se = self.se_block1(x_se)
        x_se = x_se.squeeze(-1)
        x = x + x_se
        
        # Feature transformation
        x = self.feature_transform(x)
        
        # Second SE block
        x_se = x.unsqueeze(-1)
        x_se = self.se_block2(x_se)
        x_se = x_se.squeeze(-1)
        x = x + x_se
        
        # Feature reduction
        x = self.feature_reduce(x)
        
        # Third SE block
        x_se = x.unsqueeze(-1)
        x_se = self.se_block3(x_se)
        x_se = x_se.squeeze(-1)
        x = x + x_se
        
        # Classification
        logits = self.classifier(x)
        return logits


# ==================== 5. Graph Convolutional Classifier Head ====================
class GraphConvolution(nn.Module):
    """Graph Convolution Layer"""
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCNHead(nn.Module):
    """
    Graph Convolutional Classifier Head optimized for medical cell image classification.
    
    Uses graph convolutional networks for structured feature learning.
    """
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_graph_layers=3, dropout_rate=0.2):
        super(GCNHead, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_graph_layers = num_graph_layers
        
        # Feature preprocessing
        self.feature_preprocess = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Graph convolution layers
        self.graph_layers = nn.ModuleList()
        for i in range(num_graph_layers):
            self.graph_layers.append(GraphConvolution(hidden_dim, hidden_dim))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 4, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # Feature preprocessing
        x = self.feature_preprocess(x)
        
        # Build graph structure
        x_graph = x.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        adj = torch.eye(1, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Graph convolution layers
        for gcn_layer in self.graph_layers:
            x_graph = gcn_layer(x_graph, adj)
            x_graph = F.relu(x_graph)
            x_graph = F.dropout(x_graph, p=0.2, training=self.training)
        
        # Pooling
        x_pooled = x_graph.squeeze(1)
        
        # Classification
        logits = self.classifier(x_pooled)
        return logits


# ==================== Unified Model Factory ====================
def create_model(classifier_type, input_dim, num_classes, dropout_rate=None):
    """
    Create model based on classifier type.
    
    Args:
        classifier_type: Type of classifier head ('AVGPOOL', 'ATTENTION', 'SE_ADAPTER', 'MLP', 'GCN')
        input_dim: Input feature dimension
        num_classes: Number of classes
        dropout_rate: If specified, overrides default dropout rate (for dynamic overfitting control)
    
    Returns:
        model: Created model instance
    """
    dropout_kw = {}
    if dropout_rate is not None:
        dropout_kw['dropout_rate'] = dropout_rate
    
    if classifier_type == 'AVGPOOL':
        return AvgPoolHead(input_dim, num_classes, hidden_dim=256, **dropout_kw)
    
    elif classifier_type == 'ATTENTION':
        return AttentionPoolingHead(input_dim, num_classes, hidden_dim=256, num_heads=8, **dropout_kw)
    
    elif classifier_type == 'SE_ADAPTER':
        return SEAdapterHead(input_dim, num_classes, hidden_dim=128, reduction=16, **dropout_kw)
    
    elif classifier_type == 'MLP':
        return MLPHead(input_dim, num_classes, hidden_dims=[512, 256, 128], **dropout_kw)
    
    elif classifier_type == 'GCN':
        return GCNHead(input_dim, num_classes, hidden_dim=128, num_graph_layers=3, **dropout_kw)
    
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")


# ==================== Data Loading Functions ====================
def load_features_from_h5(feature_dir):
    """
    Load feature vectors and labels from H5 files, excluding folders starting with ".".
    
    Args:
        feature_dir: Path to feature folder
        
    Returns:
        features: Feature vector array
        labels: Label array
        label_names: List of label names
    """
    feature_path = Path(feature_dir)
    
    # Get all subfolders (categories), excluding folders starting with "."
    categories = [d for d in feature_path.iterdir() if d.is_dir() and not d.name.startswith('.')]
    categories.sort()
    
    if not categories:
        raise ValueError(f"No valid category folders found in {feature_dir}")
    
    print(f"Found {len(categories)} valid categories:")
    for i, cat in enumerate(categories):
        print(f"  {i}: {cat.name}")
    
    # Show excluded folders
    excluded_dirs = [d for d in feature_path.iterdir() if d.is_dir() and d.name.startswith('.')]
    if excluded_dirs:
        print(f"Excluded folders: {[d.name for d in excluded_dirs]}")
    
    all_features = []
    all_labels = []
    label_names = []
    
    # Load features for each category
    for label_idx, category_dir in enumerate(categories):
        category_name = category_dir.name
        label_names.append(category_name)
        
        # Get all .h5 files in this category
        h5_files = list(category_dir.glob('*.h5'))
        
        if not h5_files:
            print(f"[WARNING] No .h5 files found in category {category_name}")
            continue
        
        print(f"\nLoading features for category '{category_name}' ({len(h5_files)} files)...")
        
        for h5_file in tqdm(h5_files, desc=category_name):
            try:
                with h5py.File(h5_file, 'r') as h5f:
                    if 'features' in h5f:
                        features = h5f['features'][:]
                        all_features.append(features)
                        all_labels.append(label_idx)
                    else:
                        print(f"  [WARNING] 'features' dataset not found in file {h5_file.name}")
                        
            except Exception as e:
                print(f"  [WARNING] Failed to load file {h5_file.name}: {e}")
                continue
    
    if not all_features:
        raise ValueError("No feature vectors were successfully loaded")
    
    # Convert to numpy arrays
    features = np.array(all_features)
    labels = np.array(all_labels)
    
    print(f"\nFeature loading complete:")
    print(f"  Feature shape: {features.shape}")
    print(f"  Label shape: {labels.shape}")
    print(f"  Number of classes: {len(label_names)}")
    
    return features, labels, label_names


# ==================== Training Functions ====================
def train_classifier(features, labels, label_names, output_dir, classifier_type,
                    learning_rate=0.001, batch_size=32, num_epochs=100,
                    test_size=0.2, random_state=42, weight_decay=1e-4,
                    patience=15, min_delta=1e-4, dropout_rate=None, label_smoothing=0.05):
    """
    Train classifier model.
    
    Args:
        features: Feature vectors
        labels: Labels
        label_names: Label names
        output_dir: Output directory
        classifier_type: Type of classifier head
        learning_rate: Learning rate
        batch_size: Batch size
        num_epochs: Number of training epochs
        test_size: Test set ratio
        random_state: Random seed
        weight_decay: Weight decay for regularization
        patience: Early stopping patience (stop if validation loss doesn't improve for N epochs)
        min_delta: Minimum improvement threshold for early stopping
        dropout_rate: Override default dropout rate (increase to combat overfitting)
        label_smoothing: Label smoothing coefficient
    """
    # Auto-select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"  GPU model: {torch.cuda.get_device_name(0)}")
    
    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state, stratify=labels
    )
    
    print(f"\nDataset split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Create data loaders
    train_dataset = H5FeatureDataset(X_train, y_train)
    test_dataset = H5FeatureDataset(X_test, y_test)
    
    # pin_memory speeds up GPU data transfer
    pin_memory = (device.type == 'cuda')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory)
    
    # Get input dimension and number of classes
    input_dim = features.shape[1]
    num_classes = len(label_names)
    
    print(f"\nClassifier type: {classifier_type}")
    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {num_classes}")
    
    # Create model and move to device
    model = create_model(classifier_type, input_dim, num_classes, dropout_rate=dropout_rate)
    model = model.to(device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [],
        'learning_rate': []
    }
    
    # ========== Early Stopping Initialization ==========
    best_test_loss = float('inf')
    best_test_acc = 0.0
    best_model_state = None
    best_epoch = 0
    early_stop_counter = 0
    
    print(f"\nStarting training (max {num_epochs} epochs)...")
    print(f"Early stopping: Stop if validation loss doesn't improve for {patience} consecutive epochs (> {min_delta:.6f})")
    print("-" * 70)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features_batch, labels_batch in train_loader:
            features_batch = features_batch.to(device, non_blocking=pin_memory)
            labels_batch = labels_batch.to(device, non_blocking=pin_memory)
            
            optimizer.zero_grad()
            outputs = model(features_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels_batch.size(0)
            train_correct += (predicted == labels_batch).sum().item()
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for features_batch, labels_batch in test_loader:
                features_batch = features_batch.to(device, non_blocking=pin_memory)
                labels_batch = labels_batch.to(device, non_blocking=pin_memory)
                
                outputs = model(features_batch)
                loss = criterion(outputs, labels_batch)
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels_batch.size(0)
                test_correct += (predicted == labels_batch).sum().item()
        
        # Calculate test metrics
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = test_correct / test_total
        
        # Update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_accuracy)
        history['test_loss'].append(avg_test_loss)
        history['test_acc'].append(test_accuracy)
        history['learning_rate'].append(current_lr)
        
        # ========== Early Stopping Check (based on validation loss) ==========
        if avg_test_loss < (best_test_loss - min_delta):
            best_test_loss = avg_test_loss
            best_test_acc = test_accuracy
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            early_stop_counter = 0
            improve_flag = "*"
        else:
            early_stop_counter += 1
            improve_flag = f" ({early_stop_counter}/{patience})"
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0 or early_stop_counter == 0:
            gap = train_accuracy - test_accuracy
            print(f"Epoch {epoch + 1:3d}/{num_epochs}:  [Loss] Train: {avg_train_loss:.4f} / Test: {avg_test_loss:.4f}  "
                  f"[Accuracy] Train: {train_accuracy:.4f} / Test: {test_accuracy:.4f}  "
                  f"[Gap: {gap:.4f}]  LR: {current_lr:.6f}{improve_flag}")
        
        # Trigger early stopping
        if early_stop_counter >= patience:
            print(f"\n[Early Stopping] Validation loss hasn't improved for {patience} consecutive epochs, stopping training!")
            print(f"  Best epoch: {best_epoch}  (Validation loss: {best_test_loss:.4f}, Accuracy: {best_test_acc:.4f})")
            break
    
    print("-" * 70)
    
    # Restore best model (based on validation loss, not accuracy)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"[OK] Restored best model from epoch {best_epoch}")
    else:
        print("[WARNING] No improved model found, using final epoch state")
    
    # ========== Overfitting Diagnostic Report ==========
    final_train_acc = history['train_acc'][-1]
    final_test_acc = history['test_acc'][-1]
    final_train_loss = history['train_loss'][-1]
    final_test_loss = history['test_loss'][-1]
    overfit_gap = final_train_acc - final_test_acc
    loss_gap = final_test_loss - final_train_loss
    
    print(f"\n{'='*70}")
    print("Overfitting Diagnostic Report")
    print(f"{'='*70}")
    print(f"  Final Train Accuracy: {final_train_acc:.4f}  |  Final Test Accuracy: {final_test_acc:.4f}")
    print(f"  Final Train Loss:     {final_train_loss:.4f}  |  Final Test Loss:     {final_test_loss:.4f}")
    print(f"  Accuracy Gap (Train-Test): {overfit_gap:.4f}")
    print(f"  Loss Gap (Test-Train):     {loss_gap:.4f}")
    
    if overfit_gap > 0.10:
        print(f"\n  [WARNING] Overfitting detected! Accuracy gap > 10%")
        print(f"  Suggestions: Increase --dropout_rate (e.g., 0.5), increase --weight_decay (e.g., 1e-3), or reduce --num_epochs")
    elif overfit_gap > 0.05:
        print(f"\n  [NOTICE] Slight overfitting detected (gap > 5%)")
        print(f"  Suggestions: Increase --weight_decay or --dropout_rate")
    else:
        print(f"\n  [OK] Good generalization, no significant overfitting")
    
    print(f"{'='*70}")
    
    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_filename = f"{classifier_type.lower()}_classifier.pth"
    model_path = output_path / model_filename
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_type': classifier_type,
        'input_dim': input_dim,
        'num_classes': num_classes,
        'label_names': label_names,
        'best_test_acc': best_test_acc if best_test_acc else test_accuracy,
        'best_test_loss': best_test_loss if best_test_loss != float('inf') else avg_test_loss,
        'best_epoch': best_epoch,
        'overfit_gap': overfit_gap,
        'timestamp': datetime.now().isoformat()
    }, model_path)
    
    # Save labels
    label_path = output_path / 'label_names.pkl'
    with open(label_path, 'wb') as f:
        pickle.dump(label_names, f)
    
    # Save training history
    history_path = output_path / f'training_history_{classifier_type.lower()}.pkl'
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    print(f"\nModel saved to: {output_dir}")
    print(f"  Model file: {model_path}")
    print(f"  Label file: {label_path}")
    print(f"  Training history: {history_path}")
    
    return model, history


def main():
    parser = argparse.ArgumentParser(description="Unified classifier training script")
    
    parser.add_argument("--feature_dir", default=FEATURE_DIR, help="Path to feature folder")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Path to output folder")
    parser.add_argument("--classifier_type", default=CLASSIFIER_TYPE, 
                       choices=['AVGPOOL', 'ATTENTION', 'SE_ADAPTER', 'MLP', 'GCN'],
                       help="Type of classifier head")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs (early stopping may stop earlier)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test set ratio")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    
    # ========== Overfitting Prevention Parameters ==========
    parser.add_argument("--weight_decay", type=float, default=1e-4, 
                       help="Weight decay (suggest increasing to 1e-3 ~ 5e-3 when overfitting)")
    parser.add_argument("--dropout_rate", type=float, default=None, 
                       help="Override default dropout rate (suggest setting to 0.4~0.5 when overfitting)")
    parser.add_argument("--label_smoothing", type=float, default=0.05, 
                       help="Label smoothing coefficient (suggest 0.05~0.1)")
    parser.add_argument("--patience", type=int, default=15, 
                       help="Early stopping patience: stop if validation loss doesn't improve for N epochs (suggest 10~20)")
    parser.add_argument("--min_delta", type=float, default=1e-4, 
                       help="Minimum improvement threshold for early stopping: validation loss must decrease by more than this value")
    # ==================================================
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Unified Classifier Training")
    print("=" * 70)
    print(f"Feature folder: {args.feature_dir}")
    print(f"Output folder: {args.output_dir}")
    print(f"Classifier type: {args.classifier_type}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Epochs: {args.num_epochs} (early stopping patience={args.patience})")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Label smoothing: {args.label_smoothing}")
    if args.dropout_rate is not None:
        print(f"Override dropout: {args.dropout_rate}")
    print("=" * 70)
    
    # Load feature data
    try:
        print(f"\nLoading feature data from: {args.feature_dir}")
        features, labels, label_names = load_features_from_h5(args.feature_dir)
    except Exception as e:
        print(f"[ERROR] Failed to load feature data: {e}")
        return
    
    # Train model
    try:
        model, history = train_classifier(
            features=features,
            labels=labels,
            label_names=label_names,
            output_dir=args.output_dir,
            classifier_type=args.classifier_type,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            num_epochs=args.num_epochs,
            test_size=args.test_size,
            random_state=args.random_state,
            weight_decay=args.weight_decay,
            patience=args.patience,
            min_delta=args.min_delta,
            dropout_rate=args.dropout_rate,
            label_smoothing=args.label_smoothing
        )
        
        print(f"\n[OK] {args.classifier_type} classifier training complete!")
        
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return


if __name__ == "__main__":
    main()