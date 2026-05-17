#!/usr/bin/env python3
"""
DINOv3 Feature Extraction Script

This script extracts DINOv3 feature vectors from images and saves them as .h5 files.
It preserves the original folder structure, with one .h5 file per image.

Features:
- Supports multiple image formats (jpg, jpeg, png, bmp, tiff, webp)
- Recursively processes all subdirectories
- Saves features in compressed HDF5 format with metadata
- Supports command-line argument configuration
- Automatic device selection (CUDA preferred)
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import h5py
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')


# ==================== Configuration Parameters ====================
# Input folder containing images
INPUT_FOLDER = "Stage2-DINOv3/dataset-image"

# Output folder for extracted features
OUTPUT_FOLDER = "Stage3-Classifier/features"

# Path to DINOv3 model weights
WEIGHTS_PATH = "Stage2-DINOv3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# Model name
MODEL_NAME = "dinov3_vits16"

# Device configuration (None = auto: CUDA if available, otherwise CPU)
DEVICE = None
# ================================================================


class DinoV3FeatureExtractor:
    """DINOv3 Feature Extractor
    
    This class provides methods to load the DINOv3 model and extract feature vectors
    from images.
    """
    
    def __init__(self, model_name='dinov3_vits16', weights_path=None, device=None):
        """
        Initialize the DINOv3 feature extractor.
        
        Args:
            model_name: Name of the DINOv3 model to use
            weights_path: Path to local weights file
            device: Computing device (torch.device)
        """
        self.model_name = model_name
        self.weights_path = weights_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading DINOv3 model: {model_name}")
        print(f"Using device: {self.device}")
        self._load_model()
        
    def _load_model(self):
        """Load DINOv3 model and weights"""
        try:
            # Import local dinov3 model
            sys.path.insert(0, str(Path(__file__).parent))
            from dinov3.hub.backbones import dinov3_vits16
            
            # Create model
            self.model = dinov3_vits16(pretrained=False)
            
            # Load local weights file
            if self.weights_path and os.path.exists(self.weights_path):
                print(f"Loading weights from: {self.weights_path}")
                checkpoint = torch.load(self.weights_path, map_location=self.device)
                self.model.load_state_dict(checkpoint, strict=True)
                print("✓ Weights loaded successfully")
            else:
                print(f"⚠️ Warning: Weights file not found: {self.weights_path}")
                print("Using randomly initialized weights!")
            
            self.model.to(self.device)
            self.model.eval()
            print(f"✓ DINOv3 model loaded successfully")
            
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for feature extraction.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed tensor ready for model input
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Resize to 224x224 (DINOv3 default input size)
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize to [0, 1]
            image_array = np.array(image).astype(np.float32) / 255.0
            
            # Convert to tensor and transpose dimensions (H,W,C) -> (C,H,W)
            image_tensor = torch.from_numpy(image_array).permute(2, 0, 1)
            
            # Apply ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
            
            # Add batch dimension and move to device
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            
            return image_tensor
            
        except Exception as e:
            print(f"Failed to preprocess image {image_path}: {e}")
            return None
    
    def extract_features(self, image_tensor):
        """
        Extract feature vectors from a preprocessed image tensor.
        
        Args:
            image_tensor: Preprocessed image tensor
            
        Returns:
            Feature vector as numpy array
        """
        try:
            with torch.no_grad():
                # Extract features
                features = self.model(image_tensor)
                
                # Handle different output formats
                if len(features.shape) == 3:  # (batch, tokens, dim)
                    features = features[:, 0, :]  # Get CLS token
                elif len(features.shape) == 2:  # (batch, dim)
                    pass  # Already 2D features
                
                # Convert to numpy array and flatten
                features = features.cpu().numpy().flatten()
                
                return features
                
        except Exception as e:
            print(f"Failed to extract features: {e}")
            return None
    
    def process_single_image(self, image_path):
        """
        Process a single image: preprocess and extract features.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Feature vector or None if processing fails
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_path)
        if image_tensor is None:
            return None
        
        # Extract features
        features = self.extract_features(image_tensor)
        return features


def save_features_to_h5(features, output_path, image_path):
    """
    Save feature vector to HDF5 file.
    
    Args:
        features: Feature vector to save
        output_path: Output .h5 file path
        image_path: Original image path (for metadata)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(output_path, 'w') as h5f:
            # Save feature vector with compression
            h5f.create_dataset('features', data=features, compression='gzip')
            
            # Save metadata
            h5f.attrs['image_path'] = str(image_path)
            h5f.attrs['feature_shape'] = features.shape
            h5f.attrs['feature_dtype'] = str(features.dtype)
            
        return True
        
    except Exception as e:
        print(f"Failed to save .h5 file {output_path}: {e}")
        return False


def get_all_images(input_folder):
    """
    Get all image files in folder and subfolders.
    
    Args:
        input_folder: Input folder path
        
    Returns:
        List of image file paths
    """
    input_path = Path(input_folder)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    image_files = []
    
    # Recursively find all image files
    for file_path in input_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.append(file_path)
    
    return sorted(image_files)


def process_images_recursive(input_folder, output_folder, extractor):
    """
    Process all images recursively while preserving folder structure.
    
    Args:
        input_folder: Input folder containing images
        output_folder: Output folder for features
        extractor: Feature extractor instance
        
    Returns:
        Number of successfully processed images
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    # Get all image files
    image_files = get_all_images(input_folder)
    
    if not image_files:
        print(f"⚠️ No images found in {input_folder}")
        return 0
    
    print(f"Found {len(image_files)} images")
    print("=" * 70)
    
    success_count = 0
    failed_files = []
    
    # Process each image
    for image_file in tqdm(image_files, desc="Extracting features"):
        try:
            # Calculate relative path to preserve folder structure
            relative_path = image_file.relative_to(input_path)
            
            # Generate output file path (same relative path but with .h5 extension)
            output_file_path = output_path / relative_path.with_suffix('.h5')
            
            # Skip if output already exists
            if output_file_path.exists():
                continue
            
            # Extract features
            features = extractor.process_single_image(image_file)
            if features is None:
                failed_files.append(str(image_file))
                continue
            
            # Save features
            if save_features_to_h5(features, output_file_path, image_file):
                success_count += 1
            else:
                failed_files.append(str(image_file))
                
        except Exception as e:
            print(f"\nFailed to process image {image_file}: {e}")
            failed_files.append(str(image_file))
            continue
    
    # Print failed files
    if failed_files:
        print("\n" + "=" * 70)
        print(f"Failed files ({len(failed_files)}):")
        for f in failed_files[:10]:  # Show first 10 only
            print(f"  - {f}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")
    
    return success_count


def main():
    """Main function for DINOv3 feature extraction."""
    print("=" * 70)
    print("DINOv3 Feature Extraction Tool")
    print("=" * 70)
    
    # Command line argument parsing
    parser = argparse.ArgumentParser(description="Extract DINOv3 feature vectors from images")
    parser.add_argument("--input", default=INPUT_FOLDER, 
                        help="Input folder containing images")
    parser.add_argument("--output", default=OUTPUT_FOLDER, 
                        help="Output folder for extracted features")
    parser.add_argument("--weights", default=WEIGHTS_PATH, 
                        help="Path to model weights file")
    args = parser.parse_args()
    
    input_folder = args.input
    output_folder = args.output
    weights_path = args.weights
    
    print(f"Input folder: {input_folder}")
    print(f"Output folder: {output_folder}")
    print(f"Weights file: {weights_path}")
    print("=" * 70)
    
    # Validate input folder
    if not os.path.exists(input_folder):
        print(f"❌ Input folder does not exist: {input_folder}")
        return
    
    # Check weights file
    if not os.path.exists(weights_path):
        print(f"⚠️ Warning: Weights file not found: {weights_path}")
        print("Continuing with randomly initialized weights...")
    
    # Create feature extractor
    try:
        extractor = DinoV3FeatureExtractor(
            model_name=MODEL_NAME,
            weights_path=weights_path,
            device=DEVICE
        )
    except Exception as e:
        print(f"❌ Failed to create feature extractor: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Process images
    print("\nProcessing images...")
    total_success = process_images_recursive(input_folder, output_folder, extractor)
    
    print("\n" + "=" * 70)
    print(f"Processing complete!")
    print(f"Successfully extracted features: {total_success} images")
    print(f"Output folder: {output_folder}")
    print("=" * 70)


if __name__ == "__main__":
    main()
