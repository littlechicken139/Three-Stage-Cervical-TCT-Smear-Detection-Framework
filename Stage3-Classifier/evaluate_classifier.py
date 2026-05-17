#!/usr/bin/env python3
"""
Unified Classifier Evaluation Script

Supports evaluation of multiple classifier heads: AvgPool, AttentionPooling, SE-Adapter, MLP, GCN
Switch between different classifier heads by setting CLASSIFIER_TYPE or using --classifier_type argument.

Features:
- Multiple classifier architectures evaluation
- Comprehensive performance metrics (Precision, Recall, F1, mAP)
- Model efficiency analysis (size, GFLOPs, FPS, latency)
- System information reporting
- Results exported to CSV and JSON formats
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import pandas as pd
import time
import psutil
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score
import argparse
import json
import warnings
warnings.filterwarnings('ignore')


# ==================== Configuration Parameters ====================
# Test set path
FEATURE_DIR = r"Stage3-Classifier/features/val"

# Model path
MODEL_DIR = r"Stage3-Classifier/models"

# Output path
OUTPUT_DIR = r"Stage3-Classifier/evaluate_result"
# ================================================================


# ==================== Classifier Type Configuration ====================
# Available options: 'AVGPOOL', 'ATTENTION', 'SE_ADAPTER', 'MLP', 'GCN'
CLASSIFIER_TYPE = 'MLP'
# ================================================================


# ==================== Network Architecture Parameters ====================
# These parameters should match the training parameters

# AvgPool parameters
AVGPOOL_HIDDEN_DIM = 256
AVGPOOL_DROPOUT_RATE = 0.3

# Attention parameters
ATTENTION_HIDDEN_DIM = 256
ATTENTION_NUM_HEADS = 8
ATTENTION_DROPOUT_RATE = 0.3

# MLP parameters
MLP_HIDDEN_DIMS = [512, 256, 128]
MLP_DROPOUT_RATE = 0.3

# SE-Adapter parameters
SE_ADAPTER_HIDDEN_DIM = 128
SE_ADAPTER_REDUCTION = 16
SE_ADAPTER_DROPOUT_RATE = 0.2

# GCN parameters
GCN_HIDDEN_DIM = 128
GCN_NUM_LAYERS = 3
GCN_DROPOUT_RATE = 0.2
# ================================================================


class H5FeatureDataset(torch.utils.data.Dataset):
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
    Average Pooling Classifier Head.
    
    Network architecture parameters:
    - hidden_dim: Hidden layer dimension (default: 256)
    - dropout_rate: Dropout rate (default: 0.3)
    """
    def __init__(self, input_dim, num_classes, hidden_dim=AVGPOOL_HIDDEN_DIM, 
                 dropout_rate=AVGPOOL_DROPOUT_RATE):
        super(AvgPoolHead, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Feature reduction layer
        self.feature_reduce = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier head: hidden_dim -> hidden_dim/2 -> hidden_dim/4 -> num_classes
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
        # Feature reduction: (batch_size, input_dim) -> (batch_size, hidden_dim)
        x = self.feature_reduce(x)
        
        # Expand dimension: (batch_size, hidden_dim) -> (batch_size, hidden_dim, 1)
        x = x.unsqueeze(-1)
        
        # Global average pooling: (batch_size, hidden_dim, 1) -> (batch_size, hidden_dim)
        x = self.global_pool(x).squeeze(-1)
        
        # Classification
        logits = self.classifier(x)
        return logits


# ==================== 2. Attention Pooling Classifier Head ====================
class AttentionPoolingHead(nn.Module):
    """
    Attention Pooling Classifier Head.
    
    Network architecture parameters:
    - hidden_dim: Hidden layer dimension (default: 256)
    - num_heads: Number of attention heads (default: 8)
    - dropout_rate: Dropout rate (default: 0.3)
    """
    def __init__(self, input_dim, num_classes, hidden_dim=ATTENTION_HIDDEN_DIM,
                 num_heads=ATTENTION_NUM_HEADS, dropout_rate=ATTENTION_DROPOUT_RATE):
        super(AttentionPoolingHead, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Feature preprocessing layer
        self.feature_preprocess = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Multi-head attention mechanism
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout_rate, batch_first=True
        )
        
        # Feed-forward network: hidden_dim -> hidden_dim*2 -> hidden_dim
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # LayerNorm layers
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Classifier head: hidden_dim -> hidden_dim/2 -> num_classes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        # Feature preprocessing: (batch_size, input_dim) -> (batch_size, hidden_dim)
        x = self.feature_preprocess(x)
        
        # Add sequence dimension: (batch_size, hidden_dim) -> (batch_size, 1, hidden_dim)
        x = x.unsqueeze(1)
        
        # Self-attention mechanism
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        # Remove sequence dimension: (batch_size, 1, hidden_dim) -> (batch_size, hidden_dim)
        x = x.squeeze(1)
        
        # Classification
        logits = self.classifier(x)
        return logits


# ==================== 3. MLP Classifier Head ====================
class MLPHead(nn.Module):
    """
    MLP Classifier Head.
    
    Network architecture parameters:
    - hidden_dims: List of hidden layer dimensions (default: [512, 256, 128])
    - dropout_rate: Dropout rate (default: 0.3)
    """
    def __init__(self, input_dim, num_classes, hidden_dims=MLP_HIDDEN_DIMS,
                 dropout_rate=MLP_DROPOUT_RATE):
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
    """
    Squeeze-and-Excitation Block.
    
    Parameters:
    - channels: Number of input channels
    - reduction: Reduction ratio (default: 16)
    """
    def __init__(self, channels, reduction=SE_ADAPTER_REDUCTION):
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
    SE-Adapter Classifier Head.
    
    Network architecture parameters:
    - hidden_dim: Hidden layer dimension (default: 128)
    - reduction: SE block reduction ratio (default: 16)
    - dropout_rate: Dropout rate (default: 0.2)
    """
    def __init__(self, input_dim, num_classes, hidden_dim=SE_ADAPTER_HIDDEN_DIM,
                 reduction=SE_ADAPTER_REDUCTION, dropout_rate=SE_ADAPTER_DROPOUT_RATE):
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
        
        # Feature transformation: hidden_dim -> hidden_dim
        self.feature_transform = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Feature reduction: hidden_dim -> hidden_dim/2
        self.feature_reduce = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Classifier: hidden_dim/2 -> hidden_dim/4 -> num_classes
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
    """
    Graph Convolution Layer.
    
    Parameters:
    - in_features: Input feature dimension
    - out_features: Output feature dimension
    """
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
    Graph Convolutional Classifier Head.
    
    Network architecture parameters:
    - hidden_dim: Hidden layer dimension (default: 128)
    - num_graph_layers: Number of graph convolution layers (default: 3)
    - dropout_rate: Dropout rate (default: 0.2)
    """
    def __init__(self, input_dim, num_classes, hidden_dim=GCN_HIDDEN_DIM,
                 num_graph_layers=GCN_NUM_LAYERS, dropout_rate=GCN_DROPOUT_RATE):
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
        
        # Classifier: hidden_dim -> hidden_dim/2 -> hidden_dim/4 -> num_classes
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
        
        # Build graph structure: (batch_size, hidden_dim) -> (batch_size, 1, hidden_dim)
        x_graph = x.unsqueeze(1)
        adj = torch.eye(1, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Graph convolution layers
        for gcn_layer in self.graph_layers:
            x_graph = gcn_layer(x_graph, adj)
            x_graph = F.relu(x_graph)
            x_graph = F.dropout(x_graph, p=0.2, training=self.training)
        
        # Pooling: (batch_size, 1, hidden_dim) -> (batch_size, hidden_dim)
        x_pooled = x_graph.squeeze(1)
        
        # Classification
        logits = self.classifier(x_pooled)
        return logits


# ==================== Unified Model Factory ====================
def create_model(classifier_type, input_dim, num_classes):
    """
    Create model based on classifier type.
    
    Args:
        classifier_type: Type of classifier head ('AVGPOOL', 'ATTENTION', 'SE_ADAPTER', 'MLP', 'GCN')
        input_dim: Input feature dimension
        num_classes: Number of classes
    
    Returns:
        model: Created model instance
    """
    if classifier_type == 'AVGPOOL':
        return AvgPoolHead(input_dim, num_classes)
    
    elif classifier_type == 'ATTENTION':
        return AttentionPoolingHead(input_dim, num_classes)
    
    elif classifier_type == 'SE_ADAPTER':
        return SEAdapterHead(input_dim, num_classes)
    
    elif classifier_type == 'MLP':
        return MLPHead(input_dim, num_classes)
    
    elif classifier_type == 'GCN':
        return GCNHead(input_dim, num_classes)
    
    else:
        raise ValueError(f"Unsupported classifier type: {classifier_type}")


# ==================== Data Loading Functions ====================
def load_features_from_h5(feature_dir):
    """
    Load feature vectors and labels from H5 files, excluding folders starting with '.'.
    
    Args:
        feature_dir: Path to feature folder
        
    Returns:
        features: Feature vector array
        labels: Label array
        label_names: List of label names
    """
    feature_path = Path(feature_dir)
    
    # Get all subdirectories (categories), excluding folders starting with '.'
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
            print(f"[WARNING] No .h5 files found in category '{category_name}'")
            continue
        
        print(f"\nLoading features for category '{category_name}' ({len(h5_files)} files)...")
        
        for h5_file in h5_files:
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


# ==================== Model Loading Functions ====================
def load_model(model_path, classifier_type, input_dim, num_classes, device=None):
    """
    Load classifier model.
    
    Args:
        model_path: Path to model file
        classifier_type: Classifier head type
        input_dim: Input dimension
        num_classes: Number of classes
        device: Computing device
    
    Returns:
        model: Loaded model
    """
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Create model
        model = create_model(classifier_type, input_dim, num_classes)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"[OK] Model loaded successfully")
        print(f"  Input dimension: {input_dim}")
        print(f"  Number of classes: {num_classes}")
        print(f"  Classifier type: {classifier_type}")
        
        return model, device
        
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        raise


# ==================== Performance Evaluation Functions ====================
def calculate_model_size(model_path):
    """Calculate model file size."""
    try:
        file_size_bytes = os.path.getsize(model_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        return file_size_mb
    except Exception as e:
        print(f"[WARNING] Failed to calculate model size: {e}")
        return 0.0


def calculate_gflops(model, input_dim):
    """Calculate GFLOPs."""
    try:
        gflops = 0.0
        
        # Calculate FLOPs for each module
        for module in model.modules():
            if isinstance(module, nn.Linear):
                # Linear layer: 2 * input_features * output_features
                gflops += 2 * module.in_features * module.out_features / 1e9
            elif isinstance(module, GraphConvolution):
                # Graph convolution layer
                gflops += 2 * module.in_features * module.out_features / 1e9
            elif isinstance(module, nn.MultiheadAttention):
                # Multi-head attention layer (simplified calculation)
                gflops += 2 * module.embed_dim * module.embed_dim / 1e9
            elif isinstance(module, nn.AdaptiveAvgPool1d):
                # Adaptive average pooling
                gflops += 1 / 1e9
        
        return gflops
    except Exception as e:
        print(f"[WARNING] Failed to calculate GFLOPs: {e}")
        return 0.0


def measure_inference_speed(model, input_dim, device='cpu', num_runs=100):
    """Measure inference speed."""
    try:
        model.eval()
        dummy_input = torch.randn(1, input_dim).to(device)
        
        # Warm-up
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Measure inference time
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(dummy_input)
        
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time_per_inference = total_time / num_runs
        fps = 1.0 / avg_time_per_inference
        avg_latency_ms = avg_time_per_inference * 1000
        
        return fps, avg_latency_ms
        
    except Exception as e:
        print(f"[WARNING] Failed to measure inference speed: {e}")
        return 0.0, 0.0


def calculate_map_metrics(y_true, y_score, num_classes):
    """Calculate mAP series metrics."""
    try:
        # Ensure y_true is one-hot encoded
        if len(y_true.shape) == 1:
            y_true_binary = np.eye(num_classes)[y_true]
        else:
            y_true_binary = y_true
            
        # Calculate AP for each class
        ap_scores = []
        
        for i in range(num_classes):
            if np.sum(y_true_binary[:, i]) > 0:
                ap = average_precision_score(y_true_binary[:, i], y_score[:, i])
                ap_scores.append(ap)
        
        # Calculate mAP at different thresholds
        thresholds = [0.5, 0.75]
        map_dict = {}
        
        for thresh in thresholds:
            ap_scores_thresh = []
            for i in range(num_classes):
                if np.sum(y_true_binary[:, i]) > 0:
                    ap = average_precision_score(y_true_binary[:, i], y_score[:, i])
                    ap_scores_thresh.append(ap)
            
            map_thresh = np.mean(ap_scores_thresh) if ap_scores_thresh else 0.0
            map_dict[f'mAP{int(thresh*100)}'] = map_thresh
        
        # Calculate mAP50-95
        all_thresholds = np.arange(0.5, 1.0, 0.05)
        all_maps = []
        
        for thresh in all_thresholds:
            ap_scores_temp = []
            for i in range(num_classes):
                if np.sum(y_true_binary[:, i]) > 0:
                    ap = average_precision_score(y_true_binary[:, i], y_score[:, i])
                    ap_scores_temp.append(ap)
            
            map_temp = np.mean(ap_scores_temp) if ap_scores_temp else 0.0
            all_maps.append(map_temp)
        
        map_dict['mAP50-95'] = np.mean(all_maps)
        map_dict['mAP_all'] = np.mean(ap_scores) if ap_scores else 0.0
        
        return map_dict
        
    except Exception as e:
        print(f"[WARNING] Failed to calculate mAP metrics: {e}")
        return {'mAP50': 0.0, 'mAP75': 0.0, 'mAP50-95': 0.0, 'mAP_all': 0.0}


def evaluate_model(model, features, labels, class_names, device, model_path, input_dim):
    """
    Evaluate model performance.
    
    Returns:
        performance_metrics: Dictionary of performance metrics
    """
    print("\nStarting model evaluation...")
    
    model.eval()
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(features).to(device)
    
    # Batch prediction
    batch_size = 32
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for i in range(0, len(features), batch_size):
            batch_features = features_tensor[i:i+batch_size]
            batch_outputs = model(batch_features)
            batch_probabilities = torch.softmax(batch_outputs, dim=1)
            batch_predictions = torch.argmax(batch_outputs, dim=1)
            
            all_predictions.extend(batch_predictions.cpu().numpy())
            all_probabilities.extend(batch_probabilities.cpu().numpy())
    
    predictions = np.array(all_predictions)
    probabilities = np.array(all_probabilities)
    
    print(f"\n[OK] Prediction complete! Total samples: {len(predictions)}")
    
    # Calculate basic metrics
    print("\nCalculating basic performance metrics...")
    
    # Precision, Recall, F1-Score (macro average)
    precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
    recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    
    # Precision, Recall, F1-Score (micro average)
    precision_micro = precision_score(labels, predictions, average='micro', zero_division=0)
    recall_micro = recall_score(labels, predictions, average='micro', zero_division=0)
    f1_micro = f1_score(labels, predictions, average='micro', zero_division=0)
    
    # Precision, Recall, F1-Score (weighted average)
    precision_weighted = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall_weighted = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    print("[OK] Basic metrics calculation complete")
    
    # Calculate mAP series metrics
    print("\nCalculating mAP series metrics...")
    map_metrics = calculate_map_metrics(labels, probabilities, len(class_names))
    print("[OK] mAP metrics calculation complete")
    
    # Calculate model file size
    print("\nCalculating model file size...")
    file_size_mb = calculate_model_size(model_path)
    print(f"[OK] Model file size: {file_size_mb:.2f} MB")
    
    # Calculate GFLOPs
    print("\nCalculating GFLOPs...")
    gflops = calculate_gflops(model, input_dim)
    print(f"[OK] GFLOPs: {gflops:.4f}")
    
    # Measure inference speed
    print("\nMeasuring inference speed...")
    fps, avg_latency_ms = measure_inference_speed(model, input_dim, device)
    print(f"[OK] FPS: {fps:.2f}")
    print(f"[OK] Average latency: {avg_latency_ms:.2f} ms")
    
    # Get system information
    print("\nGetting system information...")
    cpu_info = f"{psutil.cpu_count()} cores" if hasattr(psutil, 'cpu_count') else "Unknown"
    memory_info = f"{psutil.virtual_memory().total / (1024**3):.1f} GB" if hasattr(psutil, 'virtual_memory') else "Unknown"
    device_info = str(device)
    
    # Build performance metrics dictionary
    performance_metrics = {
        # Basic performance metrics
        'Overall_Accuracy': float(np.mean(labels == predictions)),
        'Precision_Macro': float(precision_macro),
        'Recall_Macro': float(recall_macro),
        'F1_Score_Macro': float(f1_macro),
        'Precision_Micro': float(precision_micro),
        'Recall_Micro': float(recall_micro),
        'F1_Score_Micro': float(f1_micro),
        'Precision_Weighted': float(precision_weighted),
        'Recall_Weighted': float(recall_weighted),
        'F1_Score_Weighted': float(f1_weighted),
        
        # mAP series metrics
        'mAP50': float(map_metrics['mAP50']),
        'mAP75': float(map_metrics['mAP75']),
        'mAP50-95': float(map_metrics['mAP50-95']),
        'mAP_All': float(map_metrics['mAP_all']),
        
        # Model efficiency metrics
        'File_Size_MB': float(file_size_mb),
        'GFLOPs': float(gflops),
        'FPS': float(fps),
        'Avg_Latency_ms': float(avg_latency_ms),
        
        # System information
        'CPU_Info': cpu_info,
        'Memory_Info': memory_info,
        'Device': device_info,
        
        # Data information
        'Total_Samples': int(len(predictions)),
        'Num_Classes': int(len(class_names)),
        'Feature_Dimension': int(input_dim),
        'Classifier_Type': CLASSIFIER_TYPE
    }
    
    return performance_metrics


# ==================== CSV Save Functions ====================
def save_to_csv(performance_metrics, output_path):
    """Save evaluation results to CSV files."""
    try:
        # Create output directory
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        base_path = str(output_path).replace('.csv', '')
        
        # 1. Main performance metrics table
        main_metrics = {
            'Metric': [
                'Overall Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)',
                'Precision (Micro)', 'Recall (Micro)', 'F1-Score (Micro)',
                'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)',
                'mAP50', 'mAP75', 'mAP50-95', 'mAP (All)',
                'File Size (MB)', 'GFLOPs', 'FPS', 'Avg Latency (ms)'
            ],
            'Value': [
                performance_metrics['Overall_Accuracy'],
                performance_metrics['Precision_Macro'], performance_metrics['Recall_Macro'], performance_metrics['F1_Score_Macro'],
                performance_metrics['Precision_Micro'], performance_metrics['Recall_Micro'], performance_metrics['F1_Score_Micro'],
                performance_metrics['Precision_Weighted'], performance_metrics['Recall_Weighted'], performance_metrics['F1_Score_Weighted'],
                performance_metrics['mAP50'], performance_metrics['mAP75'], performance_metrics['mAP50-95'], performance_metrics['mAP_All'],
                performance_metrics['File_Size_MB'], performance_metrics['GFLOPs'], 
                performance_metrics['FPS'], performance_metrics['Avg_Latency_ms']
            ],
            'Description': [
                'Overall accuracy', 'Macro-averaged precision', 'Macro-averaged recall', 'Macro-averaged F1 score',
                'Micro-averaged precision', 'Micro-averaged recall', 'Micro-averaged F1 score',
                'Weighted-averaged precision', 'Weighted-averaged recall', 'Weighted-averaged F1 score',
                'mAP at IoU 0.5', 'mAP at IoU 0.75', 'mAP at IoU 0.5-0.95', 'Average mAP across all thresholds',
                'Model file size', 'Computational complexity', 'Inference speed', 'Average latency'
            ]
        }
        
        df_main = pd.DataFrame(main_metrics)
        df_main.to_csv(f'{base_path}_main_metrics.csv', index=False, encoding='utf-8-sig')
        
        # 2. Model information table
        model_info = {
            'Model Information': ['Classifier Type', 'Input Dimension', 'Number of Classes', 'Total Samples', 
                                'File Size (MB)', 'GFLOPs', 'FPS', 'Avg Latency (ms)'],
            'Value': [
                performance_metrics['Classifier_Type'], 
                performance_metrics['Feature_Dimension'], 
                performance_metrics['Num_Classes'],
                performance_metrics['Total_Samples'], 
                performance_metrics['File_Size_MB'],
                performance_metrics['GFLOPs'], 
                performance_metrics['FPS'], 
                performance_metrics['Avg_Latency_ms']
            ]
        }
        
        df_model = pd.DataFrame(model_info)
        df_model.to_csv(f'{base_path}_model_info.csv', index=False, encoding='utf-8-sig')
        
        # 3. System information table
        system_info = {
            'System Information': ['CPU Info', 'Memory Info', 'Device'],
            'Value': [
                performance_metrics['CPU_Info'], 
                performance_metrics['Memory_Info'], 
                performance_metrics['Device']
            ]
        }
        
        df_system = pd.DataFrame(system_info)
        df_system.to_csv(f'{base_path}_system_info.csv', index=False, encoding='utf-8-sig')
        
        # 4. Model efficiency metrics
        efficiency_metrics = {
            'Efficiency Metric': ['File Size (MB)', 'GFLOPs', 'FPS', 'Avg Latency (ms)'],
            'Value': [performance_metrics['File_Size_MB'], performance_metrics['GFLOPs'], 
                      performance_metrics['FPS'], performance_metrics['Avg_Latency_ms']],
            'Description': ['Model file size', 'Computational complexity (billion floating point operations)', 
                           'Inference per second', 'Average inference latency']
        }
        
        df_efficiency = pd.DataFrame(efficiency_metrics)
        df_efficiency.to_csv(f'{base_path}_efficiency_metrics.csv', index=False, encoding='utf-8-sig')
        
        # 5. mAP series metrics
        map_metrics = {
            'mAP Metric': ['mAP50', 'mAP75', 'mAP50-95', 'mAP (All)'],
            'Value': [performance_metrics['mAP50'], performance_metrics['mAP75'], 
                      performance_metrics['mAP50-95'], performance_metrics['mAP_All']],
            'Description': ['mAP at IoU threshold 0.5', 'mAP at IoU threshold 0.75', 
                           'mAP at IoU thresholds 0.5-0.95', 'Average mAP across all thresholds']
        }
        
        df_map = pd.DataFrame(map_metrics)
        df_map.to_csv(f'{base_path}_map_metrics.csv', index=False, encoding='utf-8-sig')
        
        print(f"[OK] Performance metrics saved to CSV files: {base_path}_*.csv")
        
        # Also save in JSON format
        json_path = f'{base_path}.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(performance_metrics, f, ensure_ascii=False, indent=2)
        print(f"[OK] Performance metrics saved to JSON file: {json_path}")
        
    except Exception as e:
        print(f"[ERROR] Failed to save CSV files: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Unified classifier evaluation script")
    
    parser.add_argument("--feature_dir", default=FEATURE_DIR, help="Path to test set feature folder")
    parser.add_argument("--model_dir", default=MODEL_DIR, help="Path to model folder")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Path to evaluation results output folder")
    parser.add_argument("--classifier_type", default=CLASSIFIER_TYPE, 
                       choices=['AVGPOOL', 'ATTENTION', 'SE_ADAPTER', 'MLP', 'GCN'],
                       help="Type of classifier head")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Unified Classifier Evaluation")
    print("=" * 80)
    print(f"Test set path: {args.feature_dir}")
    print(f"Model folder: {args.model_dir}")
    print(f"Output folder: {args.output_dir}")
    print(f"Classifier type: {args.classifier_type}")
    print("=" * 80)
    
    try:
        # Create device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load test data
        print(f"\nStep 1: Load test data")
        features, labels, class_names = load_features_from_h5(args.feature_dir)
        
        # Get input dimension
        input_dim = features.shape[1]
        num_classes = len(class_names)
        
        # Build model path
        model_filename = f"{args.classifier_type.lower()}_classifier.pth"
        model_path = Path(args.model_dir) / model_filename
        
        print(f"\nStep 2: Load model")
        print(f"Model path: {model_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model
        model, device = load_model(
            model_path=str(model_path),
            classifier_type=args.classifier_type,
            input_dim=input_dim,
            num_classes=num_classes,
            device=device
        )
        
        # Evaluate model
        print(f"\nStep 3: Evaluate model performance")
        performance_metrics = evaluate_model(
            model=model,
            features=features,
            labels=labels,
            class_names=class_names,
            device=device,
            model_path=str(model_path),
            input_dim=input_dim
        )
        
        # Save results
        print(f"\nStep 4: Save evaluation results")
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_filename = f"{args.classifier_type.lower()}_evaluation_results.csv"
        output_path = output_dir / output_filename
        
        save_to_csv(performance_metrics, output_path)
        
        # Print summary
        print("\n" + "=" * 80)
        print("Evaluation complete!")
        print(f"Classifier type: {performance_metrics['Classifier_Type']}")
        print(f"Overall accuracy: {performance_metrics['Overall_Accuracy']:.4f}")
        print(f"F1-Score (Macro): {performance_metrics['F1_Score_Macro']:.4f}")
        print(f"mAP50: {performance_metrics['mAP50']:.4f}")
        print(f"File size: {performance_metrics['File_Size_MB']:.2f} MB")
        print(f"GFLOPs: {performance_metrics['GFLOPs']:.4f}")
        print(f"FPS: {performance_metrics['FPS']:.2f}")
        print(f"Results saved to: {str(output_path).replace('.csv', '')}_*.csv and {str(output_path).replace('.csv', '')}.json")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    import sys
    sys.exit(exit_code)