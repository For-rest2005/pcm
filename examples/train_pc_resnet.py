#!/usr/bin/env python3
"""
Training script for PC-ResNet56 on Tiny-ImageNet.
This demonstrates the full training pipeline for ResNet-56 in the PC framework.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import jax.random as jr
from pcm.resnet import PCResNet56
from pcm.datasets import get_tiny_imagenet_dataloaders


def main(save_path=None, load_path=None):
    """
    Main training function for PC-ResNet56.
    
    Args:
        save_path: Path to save the trained model (optional)
        load_path: Path to load a pre-trained model (optional)
    """
    print("=" * 70)
    print("Predictive Coding ResNet-56 Training on Tiny-ImageNet")
    print("=" * 70)
    
    # Configuration
    config = {
        "img_size": 128,
        "in_channels": 3,
        "num_classes": 200,
        "batch_size": 32,
        "epochs": 5,
        "train_lr": 2e-4,
        "inf_lr": 2e-3,
        "inf_epoch": 200,
        "print_times_every_epoch": 4,
        "seed": 42,
        "num_train_samples": 128,  # Use 20k samples (out of 100k total)
        "num_val_samples": 10000,     # Use 4k validation samples
    }
    
    print("Loading Tiny-ImageNet dataset...")
    print("-" * 70)
    train_loader, val_loader = get_tiny_imagenet_dataloaders(
        data_dir='./data/tiny-imagenet-200',
        batch_size=config["batch_size"],
        img_size=config["img_size"],
        num_train_samples=config["num_train_samples"],
        num_val_samples=config["num_val_samples"],
    )
    
    key = jr.PRNGKey(config["seed"])
    model = PCResNet56(
        img_size=config["img_size"],
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
        key=key,
    )
    
    print("Network Architecture:")
    print("  Model: ResNet-56")
    print(f"  Input: {config['in_channels']} x {config['img_size']} x {config['img_size']}")
    print("  Layers:")
    print("    - Initial Conv: 3x3, 16 filters")
    print("    - Stage 1: 9 residual blocks, 16 filters, 64x64")
    print("    - Stage 2: 9 residual blocks, 32 filters, 32x32 (downsampled)")
    print("    - Stage 3: 9 residual blocks, 64 filters, 16x16 (downsampled)")
    print("    - Global Average Pooling + FC")
    print(f"  Output: {config['num_classes']} classes")
    print(f"  Total depth: 56 layers (1 + 9*2 + 9*2 + 9*2 + 1)")
    print()
    
    if load_path:
        print(f"Loading pre-trained model from: {load_path}")
        print("-" * 70)
        model.network.load(load_path)
        print("  Model loaded successfully!")
        print()
    
    key, subkey = jr.split(key)
    print(f"batches_per_epoch: {len(train_loader)}")

    energy_history = model.train(
        train_data=train_loader,
        key=subkey,
        epochs=config["epochs"],
        train_lr=config["train_lr"],
        inf_lr=config["inf_lr"],
        inf_epoch=config["inf_epoch"],
        verbose=True,
        print_times_every_epoch=config["print_times_every_epoch"],
    )
    print("\nTraining completed!")
    print()
    
    # Save model if specified
    if save_path:
        print(f"Saving model to: {save_path}")
        print("-" * 70)
        model.network.save(save_path)
        print("  Model saved successfully!")
        print()
    
    # Evaluate model
    print("Step 4: Evaluating on validation set...")
    print("-" * 70)
    val_accuracy = model.evaluate_accuracy(
        test_data=val_loader,
        verbose=True,
    )
    print()
    
    # Print summary
    print("=" * 70)
    print("Training Summary")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Training samples: {config['num_train_samples']}")
    print(f"  Validation samples: {config['num_val_samples']}")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Learning rate: {config['train_lr']}")
    print()
    print(f"Results:")
    print(f"  Initial Energy: {energy_history[0]:.6f}")
    print(f"  Final Energy: {energy_history[-1]:.6f}")
    print(f"  Energy Reduction: {energy_history[0] - energy_history[-1]:.6f}")
    print(f"  Validation Accuracy: {val_accuracy * 100:.2f}%")
    print("=" * 70)
    
    return model, energy_history, val_accuracy


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train PC-ResNet56 on Tiny-ImageNet")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save the trained model (e.g., './checkpoints/resnet_model.eqx')")
    parser.add_argument("--load_path", type=str, default=None,
                        help="Path to load a pre-trained model (e.g., './checkpoints/resnet_model.eqx')")
    args = parser.parse_args()
    
    main(save_path=args.save_path, load_path=args.load_path)

