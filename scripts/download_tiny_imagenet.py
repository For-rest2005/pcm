#!/usr/bin/env python3
"""
Script to download and prepare Tiny-ImageNet-200 dataset.
"""
import os
import sys
import urllib.request
import zipfile
import shutil
from pathlib import Path


def download_tiny_imagenet(data_dir: str = "./data"):
    """
    Download and extract Tiny-ImageNet-200 dataset.
    
    Args:
        data_dir: Directory to download and extract data to
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    tiny_imagenet_dir = data_dir / "tiny-imagenet-200"
    
    # Check if already downloaded
    if tiny_imagenet_dir.exists():
        print(f"Tiny-ImageNet already exists at {tiny_imagenet_dir}")
        print("If you want to re-download, please delete the directory first.")
        return
    
    # Download URL
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_path = data_dir / "tiny-imagenet-200.zip"
    
    print("=" * 60)
    print("Downloading Tiny-ImageNet-200 dataset")
    print("=" * 60)
    print(f"URL: {url}")
    print(f"Destination: {data_dir}")
    print()
    
    # Download with progress
    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\rDownloading: {percent}%")
        sys.stdout.flush()
    
    try:
        print("Downloading... (this may take several minutes)")
        urllib.request.urlretrieve(url, zip_path, progress_hook)
        print("\nDownload completed!")
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("Please download manually from:")
        print(url)
        return
    
    # Extract
    print("\nExtracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction completed!")
    except Exception as e:
        print(f"Error extracting dataset: {e}")
        return
    
    # Clean up zip file
    print("\nCleaning up...")
    zip_path.unlink()
    
    # Reorganize validation directory
    print("Reorganizing validation directory...")
    val_dir = tiny_imagenet_dir / "val"
    
    # Read val annotations
    val_annotations_file = val_dir / "val_annotations.txt"
    
    if val_annotations_file.exists():
        # Create class directories in val
        with open(val_annotations_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_name = parts[0]
                    class_name = parts[1]
                    
                    # Create class directory if it doesn't exist
                    class_dir = val_dir / class_name
                    class_dir.mkdir(exist_ok=True)
                    
                    # Move image to class directory
                    src = val_dir / "images" / img_name
                    dst = class_dir / img_name
                    
                    if src.exists():
                        shutil.move(str(src), str(dst))
        
        # Remove empty images directory and annotations file
        images_dir = val_dir / "images"
        if images_dir.exists() and not any(images_dir.iterdir()):
            images_dir.rmdir()
        if val_annotations_file.exists():
            val_annotations_file.unlink()
    
    print("\nDataset structure:")
    print(f"  Training data: {tiny_imagenet_dir / 'train'}")
    print(f"  Validation data: {tiny_imagenet_dir / 'val'}")
    print(f"  Test data: {tiny_imagenet_dir / 'test'}")
    
    # Print statistics
    train_dir = tiny_imagenet_dir / "train"
    val_dir = tiny_imagenet_dir / "val"
    
    if train_dir.exists():
        num_train_classes = len([d for d in train_dir.iterdir() if d.is_dir()])
        print(f"\n  Number of training classes: {num_train_classes}")
        
        # Count training images
        num_train_images = sum(
            len(list((train_dir / class_dir / "images").glob("*.JPEG")))
            for class_dir in train_dir.iterdir()
            if class_dir.is_dir() and (class_dir / "images").exists()
        )
        print(f"  Number of training images: {num_train_images}")
    
    if val_dir.exists():
        num_val_classes = len([d for d in val_dir.iterdir() if d.is_dir()])
        num_val_images = sum(
            len(list(class_dir.glob("*.JPEG")))
            for class_dir in val_dir.iterdir()
            if class_dir.is_dir()
        )
        print(f"  Number of validation classes: {num_val_classes}")
        print(f"  Number of validation images: {num_val_images}")
    
    print("\n" + "=" * 60)
    print("Tiny-ImageNet-200 setup completed!")
    print("=" * 60)
    print("\nYou can now use the dataset with:")
    print("  from pcm.vit import load_tiny_imagenet_data")
    print("  train_data, val_data = load_tiny_imagenet_data()")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Tiny-ImageNet-200 dataset")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Directory to download data to (default: ./data)"
    )
    
    args = parser.parse_args()
    download_tiny_imagenet(args.data_dir)





