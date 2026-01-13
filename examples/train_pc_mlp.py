#!/usr/bin/env python3
"""
Example script to train Predictive Coding MLP on MNIST dataset.
Run this script to train and evaluate the PC-MLP model.
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path to import pcm
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import jax.random as jr
from pcm.mlp import PCMLP
from pcm.datasets import get_mnist_dataloaders

def main(save_path=None, load_path=None):
    """
    Main function to train and evaluate PC-MLP on MNIST.
    
    Args:
        save_path: Path to save the trained model (optional)
        load_path: Path to load a pre-trained model (optional)
    """
    print("=" * 70)
    print("Predictive Coding MLP - MNIST Training")
    print("=" * 70)
    
    # Hyperparameters
    batch_size = 64
    epochs = 1
    train_lr = 1e-3
    inf_lr = 0.1
    inf_epoch = 80
    hidden_dims = [256, 128]
    
    print(f"\nHyperparameters:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Training LR: {train_lr}")
    print(f"  Inference LR: {inf_lr}")
    print(f"  Inference epochs: {inf_epoch}")
    print(f"  Hidden dimensions: {hidden_dims}")
    
    # Load MNIST data
    print("\n" + "-" * 70)
    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=batch_size,
        flatten=True,
        one_hot=True
    )
    print(f"  Training batches: {len(train_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Initialize model
    print("\n" + "-" * 70)
    print("Initializing PC-MLP model...")
    key = jr.PRNGKey(42)
    model = PCMLP(
        input_dim=784,
        hidden_dims=hidden_dims,
        output_dim=10,
        key=key
    )
    print(f"  Model architecture: {[784] + hidden_dims + [10]}")
    
    # Load pre-trained model if specified
    if load_path:
        print(f"\nLoading pre-trained model from: {load_path}")
        model.network.load(load_path)
        print("  Model loaded successfully!")
    
    # Train model
    print("\n" + "-" * 70)
    print("Training model...")
    print("-" * 70)
    key, train_key = jr.split(key)
    energy_history = model.train(
        train_data=train_loader,
        key=train_key,
        epochs=epochs,
        train_lr=train_lr,
        inf_lr=inf_lr,
        inf_epoch=inf_epoch,
        verbose=True,
        print_times_every_epoch=10
    )
    
    # Save model if specified
    if save_path:
        print(f"\nSaving model to: {save_path}")
        model.network.save(save_path)
        print("  Model saved successfully!")
    
    print("\n" + "-" * 70)
    print("Evaluating on test set...")
    print("-" * 70)
    test_accuracy = model.evaluate_accuracy(
        test_data=test_loader
    )
    
    # Final results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  Test Accuracy: {test_accuracy * 100:.2f}%")
    print("=" * 70)
    
    return model, energy_history, test_accuracy

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train Predictive Coding MLP on MNIST")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save the trained model (e.g., './checkpoints/mlp_model.eqx')")
    parser.add_argument("--load_path", type=str, default=None,
                        help="Path to load a pre-trained model (e.g., './checkpoints/mlp_model.eqx')")
    args = parser.parse_args()
    
    # Run the main training and evaluation
    model, energy_history, test_accuracy = main(
        save_path=args.save_path,
        load_path=args.load_path
    )
    
    print("\n\nYou can now use the trained model for predictions!")
    print("Example:")
    print("  predictions = model.predict(input_data)")

