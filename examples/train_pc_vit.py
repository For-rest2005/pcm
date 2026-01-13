"""
Example script to train PC-ViT on Tiny-ImageNet.
"""
import argparse
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
from pcm.vit import PCViT
from pcm.datasets import get_tiny_imagenet_dataloaders


def main(save_path=None, load_path=None):
    """Train PC-ViT on Tiny-ImageNet dataset.
    
    Args:
        save_path: Path to save the trained model (optional)
        load_path: Path to load a pre-trained model (optional)
    """
    
    print("=" * 60)
    print("PC-ViT Training on Tiny-ImageNet")
    print("=" * 60)
    print()
    
    # Configuration
    config = {
        # Model architecture
        "img_size": 64,
        "patch_size": 8,
        "in_channels": 3,
        "num_classes": 200,
        "embed_dim": 256,          # Smaller for faster training
        "num_layers": 8,           # 4 transformer layers
        "num_heads": 4,            # 4 attention heads
        "mlp_ratio": 2,            # MLP hidden dim = 2 * embed_dim
        
        # Training settings
        "batch_size": 64,          # Smaller batch for memory efficiency
        "epochs": 1,               # Start with fewer epochs
        "train_lr": 2e-3,          # Learning rate for weight updates
        "inf_lr": 5e-3,             # Inference learning rate
        "inf_epoch": 300,           # Max inference iterations
        
        # Data settings
        "num_train_samples": 128 * 500, # Use subset for quick testing
        "num_val_samples": 10000,   # Use subset for quick testing
        
        "seed": 42,
        "print_times_every_epoch": 50,
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Load data
    print("Loading Tiny-ImageNet dataset...")
    print("Note: Make sure you have downloaded the dataset first!")
    print("Run: python scripts/download_tiny_imagenet.py")
    print()
    
    try:
        train_loader, val_loader = get_tiny_imagenet_dataloaders(
            data_dir='./data/tiny-imagenet-200',
            batch_size=config["batch_size"],
            img_size=config["img_size"],
            num_train_samples=config["num_train_samples"],
            num_val_samples=config["num_val_samples"],
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nPlease make sure:")
        print("1. You have downloaded Tiny-ImageNet using: python scripts/download_tiny_imagenet.py")
        print("2. The dataset is in ./data/tiny-imagenet-200/")
        return
    
    print()
    
    # Create model
    print("Initializing PC-ViT...")
    key = jr.PRNGKey(config["seed"])
    model = PCViT(
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
        embed_dim=config["embed_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        key=key,
    )
    
    print(f"Model architecture:")
    print(f"  Image size: {config['img_size']}x{config['img_size']}")
    print(f"  Patches: {model.num_patches} ({config['patch_size']}x{config['patch_size']} each)")
    print(f"  Embedding dimension: {config['embed_dim']}")
    print(f"  Transformer layers: {config['num_layers']}")
    print(f"  Attention heads: {config['num_heads']}")
    print(f"  Output classes: {config['num_classes']}")
    print()
    
    # Load pre-trained model if specified
    if load_path:
        print(f"Loading pre-trained model from: {load_path}")
        print("-" * 60)
        model.network.load(load_path)
        print("  Model loaded successfully!")
        print()
    
    # Train with custom loop for periodic accuracy evaluation
    print("Starting training...")
    print("This may take a while depending on your hardware...")
    print(f"Will evaluate accuracy every 100 batches")
    print()
    
    key, subkey = jr.split(key)
    
    train_opt = optax.adam(config["train_lr"])
    weights = [edge.forward_fn for edge in model.network.edges]
    train_opt_state = train_opt.init(eqx.filter(weights, eqx.is_array))
    
    energy_history = []
    total_batches = 0
    batches_per_epoch = len(train_loader)
    
    for epoch in range(config["epochs"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print(f"{'='*60}")
        
        epoch_energy = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            key, subkey = jr.split(key)
            
            # print(batch)
            train_opt_state, energy, _ = model.network.train_step(
                input_states=batch,
                key=subkey,
                returned_vertices=None,
                init_fun=jr.normal,
                train_opt=train_opt,
                train_opt_state=train_opt_state,
                inf_lr=config["inf_lr"],
                inf_epoch=config["inf_epoch"],
            )
            
            energy_history.append(float(energy))
            epoch_energy += energy
            total_batches += 1
            # print(f"  Batch {batch_idx + 1}/{batches_per_epoch}")

            if (batch_idx + 1) % 1 == 0:
                print(f"  Batch {batch_idx + 1}/{batches_per_epoch}, "
                      f"Energy: {energy:.4f}")
            
            if total_batches % 50 == 0:
                print(f"\n{'─'*60}")
                print(f"Evaluation at {total_batches} batches (Epoch {epoch+1}, Batch {batch_idx+1})")
                print(f"{'─'*60}")
                
                val_subset = val_loader
                
                correct = 0
                total = 0
                # Track predictions per class
                pred_count = {}  # How many times each class was predicted
                correct_per_class = {}  # How many correct predictions per class
                
                for val_batch in val_subset:
                    images = val_batch["input"]
                    targets = val_batch["output"]
                    
                    predictions = model.predict(images)
                    pred_labels = jnp.argmax(predictions, axis=1)
                    true_labels = jnp.argmax(targets, axis=1)
                    
                    # Count predictions per class
                    for pred_label in pred_labels:
                        pred_label_int = int(pred_label)
                        pred_count[pred_label_int] = pred_count.get(pred_label_int, 0) + 1
                    
                    # Count correct predictions per class
                    for pred_label, true_label in zip(pred_labels, true_labels):
                        pred_label_int = int(pred_label)
                        true_label_int = int(true_label)
                        if pred_label_int == true_label_int:
                            correct_per_class[pred_label_int] = correct_per_class.get(pred_label_int, 0) + 1
                    
                    correct += jnp.sum(pred_labels == true_labels)
                    total += images.shape[0]
                
                accuracy = float(correct) / total
                
                # Get top 5 most predicted classes
                sorted_pred = sorted(pred_count.items(), key=lambda x: x[1], reverse=True)[:5]
                
                print(f"Validation Accuracy: {accuracy * 100:.2f}% ({correct}/{total} correct)")
                print(f"Current Energy: {energy:.6f}")
                print(f"\nTop 5 Most Predicted Classes:")
                for class_id, count in sorted_pred:
                    correct_count = correct_per_class.get(class_id, 0)
                    percentage = (count / total) * 100
                    correct_rate = (correct_count / count * 100) if count > 0 else 0
                    print(f"  Class {class_id:3d}: {count:4d} predictions ({percentage:5.2f}%), "
                          f"{correct_count:3d} correct ({correct_rate:5.2f}%)")
                
                # Show classes with most correct predictions
                if correct_per_class:
                    sorted_correct = sorted(correct_per_class.items(), key=lambda x: x[1], reverse=True)[:5]
                    print(f"\nTop 5 Classes by Correct Predictions:")
                    for class_id, correct_count in sorted_correct:
                        pred_total = pred_count.get(class_id, 0)
                        correct_rate = (correct_count / pred_total * 100) if pred_total > 0 else 0
                        print(f"  Class {class_id:3d}: {correct_count:3d} correct out of {pred_total:4d} predictions ({correct_rate:5.2f}%)")
                
                print(f"{'─'*60}\n")
        
        avg_epoch_energy = epoch_energy / batches_per_epoch
        print(f"\nEpoch {epoch + 1} completed. Average energy: {avg_epoch_energy:.6f}")
    
    print("\nTraining completed!")
    print()
    
    # Save model if specified
    if save_path:
        print(f"Saving model to: {save_path}")
        print("-" * 60)
        model.network.save(save_path)
        print("  Model saved successfully!")
        print()
    
    print("Evaluating on training set...")
    train_accuracy = model.evaluate_accuracy(
        test_data=train_loader,
        verbose=True,
    )
    print()
    
    print("Evaluating on validation set...")
    val_accuracy = model.evaluate_accuracy(
        test_data=val_loader,
        verbose=True,
    )
    print()
    
    # Summary
    print("=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Configuration: {config['num_layers']} layers, {config['embed_dim']} dim, {config['num_heads']} heads")
    print(f"Training samples: {config['num_train_samples']}")
    print(f"Validation samples: {config['num_val_samples']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Final training energy: {energy_history[-1]:.6f}")
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Validation accuracy: {val_accuracy * 100:.2f}%")
    print("=" * 60)
    
    return model, energy_history, train_accuracy, val_accuracy


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train PC-ViT on Tiny-ImageNet")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save the trained model (e.g., './checkpoints/vit_model.eqx')")
    parser.add_argument("--load_path", type=str, default=None,
                        help="Path to load a pre-trained model (e.g., './checkpoints/vit_model.eqx')")
    args = parser.parse_args()
    
    main(save_path=args.save_path, load_path=args.load_path)

