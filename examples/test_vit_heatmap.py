"""
Test script to visualize PC-ViT predictions on Tiny-ImageNet using heatmaps.
This script initializes a ViT model, performs forward passes on TinyImageNet samples,
and visualizes the predicted probabilities vs actual one-hot labels as heatmaps.
"""
import argparse
import jax.random as jr
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from pcm.vit import PCViT
from pcm.datasets import get_tiny_imagenet_dataloaders


def plot_prediction_heatmap(predictions, targets, num_samples=16, num_classes_to_show=50):
    predictions = predictions[:num_samples]
    targets = targets[:num_samples]
    
    selected_classes = set()
    
    for i in range(len(targets)):
        true_class = int(np.argmax(targets[i]))
        selected_classes.add(true_class)
    
    top_k = max(5, num_classes_to_show // num_samples)
    for i in range(len(predictions)):
        top_classes = np.argsort(predictions[i])[-top_k:]
        selected_classes.update(top_classes.tolist())

    selected_classes = sorted(list(selected_classes))
    if len(selected_classes) > num_classes_to_show:
        true_labels = set(int(np.argmax(targets[i])) for i in range(len(targets)))
        predicted_classes = [c for c in selected_classes if c not in true_labels]
        
        remaining_slots = num_classes_to_show - len(true_labels)
        if len(predicted_classes) > remaining_slots:
            step = len(predicted_classes) // remaining_slots
            predicted_classes = [predicted_classes[i * step] for i in range(remaining_slots)]
        
        selected_classes = sorted(list(true_labels) + predicted_classes)
    
    predictions_subset = predictions[:, selected_classes]
    targets_subset = targets[:, selected_classes]
    
    num_displayed_samples = len(predictions_subset)
    num_displayed_classes = len(selected_classes)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    im1 = ax1.imshow(predictions_subset, aspect='auto', cmap='viridis', interpolation='nearest')
    ax1.set_title('Model Predictions (Forward Pass)', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Class Index', fontsize=12)
    ax1.set_ylabel('Sample Index', fontsize=12)
    ax1.set_yticks(range(num_displayed_samples))
    ax1.set_yticklabels([f'Sample {i}' for i in range(num_displayed_samples)])
    
    if num_displayed_classes <= 50:
        ax1.set_xticks(range(num_displayed_classes))
        ax1.set_xticklabels([str(selected_classes[i]) for i in range(num_displayed_classes)], 
                           rotation=45, ha='right', fontsize=8)
    else:
        tick_step = max(1, num_displayed_classes // 20)
        tick_positions = range(0, num_displayed_classes, tick_step)
        ax1.set_xticks(tick_positions)
        ax1.set_xticklabels([str(selected_classes[i]) for i in tick_positions], 
                           rotation=45, ha='right', fontsize=8)
    
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Predicted Probability', rotation=270, labelpad=20, fontsize=11)
    
    im2 = ax2.imshow(targets_subset, aspect='auto', cmap='RdYlGn', interpolation='nearest')
    ax2.set_title('Actual Labels (One-Hot)', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Class Index', fontsize=12)
    ax2.set_ylabel('Sample Index', fontsize=12)
    ax2.set_yticks(range(num_displayed_samples))
    ax2.set_yticklabels([f'Sample {i}' for i in range(num_displayed_samples)])
    
    if num_displayed_classes <= 50:
        ax2.set_xticks(range(num_displayed_classes))
        ax2.set_xticklabels([str(selected_classes[i]) for i in range(num_displayed_classes)], 
                           rotation=45, ha='right', fontsize=8)
    else:
        tick_step = max(1, num_displayed_classes // 20)
        tick_positions = range(0, num_displayed_classes, tick_step)
        ax2.set_xticks(tick_positions)
        ax2.set_xticklabels([str(selected_classes[i]) for i in tick_positions], 
                           rotation=45, ha='right', fontsize=8)
    
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('One-Hot Label', rotation=270, labelpad=20, fontsize=11)
    
    ax1.grid(False)
    ax2.grid(False)
    
    for i in range(num_displayed_samples):
        pred_class_idx = int(np.argmax(predictions_subset[i]))
        true_class_idx = int(np.argmax(targets_subset[i]))
        
        ax1.plot(pred_class_idx, i, 'r*', markersize=12, markeredgecolor='white', markeredgewidth=1.5)
        ax2.plot(true_class_idx, i, 'b*', markersize=12, markeredgecolor='white', markeredgewidth=1.5)
    
    info_text = f'Showing {num_displayed_classes} classes (includes all true labels + top predicted classes)'
    plt.suptitle(f'ViT Model Predictions vs Actual Labels on TinyImageNet\n{info_text}', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    return fig, selected_classes


def plot_vertex_state_heatmap(vertex_state, sample_idx=None):
    """
    Plot detailed heatmap of a single sample's state from the second-to-last vertex.
    Shows each scalar value in the (num_patches+1, embed_dim) representation.
    
    Args:
        vertex_state: Array of shape (num_samples, num_patches+1, embed_dim) 
                     representing the transformer layer output
        sample_idx: Index of sample to visualize (if None, randomly select one)
    """
    # Randomly select a sample if not specified
    if sample_idx is None:
        sample_idx = np.random.randint(0, len(vertex_state))
    
    # Extract single sample: shape (num_patches+1, embed_dim)
    single_sample_state = vertex_state[sample_idx]
    
    num_patches_plus_cls = single_sample_state.shape[0]  # num_patches + 1 (CLS token)
    embed_dim = single_sample_state.shape[1]
    
    # Create figure with single large heatmap
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Plot the full state matrix
    im = ax.imshow(single_sample_state, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    
    ax.set_title(f'Transformer Layer State - Sample {sample_idx}\nEach scalar value shown', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Embedding Dimension', fontsize=12)
    ax.set_ylabel('Patch Index (0=CLS token)', fontsize=12)
    
    # Y-axis: patches (highlight CLS token at position 0)
    if num_patches_plus_cls <= 65:
        tick_positions_y = [0] + list(range(8, num_patches_plus_cls, 8))
        ax.set_yticks(tick_positions_y)
        ax.set_yticklabels([f'{i}' for i in tick_positions_y])
    else:
        tick_step = max(1, num_patches_plus_cls // 20)
        tick_positions_y = range(0, num_patches_plus_cls, tick_step)
        ax.set_yticks(tick_positions_y)
        ax.set_yticklabels([f'{i}' for i in tick_positions_y])
    
    # X-axis: embedding dimensions
    if embed_dim <= 200:
        tick_step_x = max(10, embed_dim // 20)
        tick_positions_x = range(0, embed_dim, tick_step_x)
        ax.set_xticks(tick_positions_x)
        ax.set_xticklabels([f'{i}' for i in tick_positions_x])
    else:
        tick_step_x = embed_dim // 20
        tick_positions_x = range(0, embed_dim, tick_step_x)
        ax.set_xticks(tick_positions_x)
        ax.set_xticklabels([f'{i}' for i in tick_positions_x])
    
    # Add horizontal line to separate CLS token from patches
    ax.axhline(y=0.5, color='yellow', linestyle='--', linewidth=2, alpha=0.8, label='CLS token boundary')
    ax.legend(loc='upper right', fontsize=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Activation Value', rotation=270, labelpad=20, fontsize=11)
    
    # Add statistics
    stats_text = (f'Shape: ({num_patches_plus_cls}, {embed_dim})\n'
                  f'Min: {np.min(single_sample_state):.4f}, '
                  f'Max: {np.max(single_sample_state):.4f}, '
                  f'Mean: {np.mean(single_sample_state):.4f}\n'
                  f'CLS token mean: {np.mean(single_sample_state[0]):.4f}, '
                  f'Patches mean: {np.mean(single_sample_state[1:]):.4f}')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    return fig, sample_idx


def main(load_path=None):
    figures_dir = Path("./figures")
    figures_dir.mkdir(exist_ok=True)
    print(f"Figures will be saved to: {figures_dir.absolute()}")
    print()
    
    config = {
        "img_size": 64,
        "patch_size": 8,
        "in_channels": 3,
        "num_classes": 200,
        "embed_dim": 192,
        "num_layers": 4,
        "num_heads": 4,
        "mlp_ratio": 2,
        
        "batch_size": 16,  # Get 16 samples for visualization
        "num_samples": 16,
        "num_classes_to_show": 50,  # Max number of classes to display (smart selection)
        
        "seed": 42,
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    print("Loading Tiny-ImageNet dataset...")
    
    try:
        _, val_loader = get_tiny_imagenet_dataloaders(
            data_dir='./data/tiny-imagenet-200',
            batch_size=config["batch_size"],
            img_size=config["img_size"],
            num_train_samples=0,  # Don't load training data
            num_val_samples=config["num_samples"],
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nPlease make sure:")
        print("1. You have downloaded Tiny-ImageNet")
        print("2. The dataset is in ./data/tiny-imagenet-200/")
        return
    
    if len(val_loader) == 0:
        print("Error: No validation data loaded!")
        return
    
    batch = next(iter(val_loader))
    images = batch["input"]
    targets = batch["output"]
    
    print(f"Loaded batch shape:")
    print(f"  Images: {images.shape}")
    print(f"  Targets: {targets.shape}")
    print()
    
    if load_path:
        print(f"Loading pre-trained model from: {load_path}")
    else:
        print("Initializing PC-ViT with random weights...")
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
    
    # Load pre-trained weights if specified
    if load_path:
        model.network.load(load_path)
        print("  Model loaded successfully!")
    
    print(f"Model architecture:")
    print(f"  Image size: {config['img_size']}x{config['img_size']}")
    print(f"  Patches: {model.num_patches} ({config['patch_size']}x{config['patch_size']} each)")
    print(f"  Embedding dimension: {config['embed_dim']}")
    print(f"  Transformer layers: {config['num_layers']}")
    print(f"  Attention heads: {config['num_heads']}")
    print(f"  Output classes: {config['num_classes']}")
    print()
    
    # Perform forward pass and get intermediate states
    print("Performing forward pass...")
    predictions = model.predict(images)
    
    # Get the second-to-last vertex state (last transformer layer)
    # The vertices are: input, patches, layer_0, layer_1, ..., layer_{n-1}, output
    # Second-to-last is layer_{n-1}
    second_to_last_vertex_name = f"layer_{config['num_layers'] - 1}"
    print(f"Getting state from vertex: {second_to_last_vertex_name}")
    
    input_states = {"input": images}
    all_states = model.network.forward(
        input_states=input_states,
        returned_vertices=[second_to_last_vertex_name, "output"]
    )
    vertex_state = all_states[second_to_last_vertex_name]
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions are probabilities (softmax applied): "
          f"min={float(jnp.min(predictions)):.6f}, max={float(jnp.max(predictions)):.6f}")
    print(f"{second_to_last_vertex_name} state shape: {vertex_state.shape}")
    print(f"  (batch_size={vertex_state.shape[0]}, num_patches+1={vertex_state.shape[1]}, embed_dim={vertex_state.shape[2]})")
    print()
    
    pred_labels = jnp.argmax(predictions, axis=1)
    true_labels = jnp.argmax(targets, axis=1)
    accuracy = float(jnp.mean(pred_labels == true_labels))
    
    if load_path:
        print(f"Model accuracy on samples: {accuracy * 100:.2f}% ({int(jnp.sum(pred_labels == true_labels))}/{len(pred_labels)} correct)")
    else:
        print(f"Random initialization accuracy: {accuracy * 100:.2f}% ({int(jnp.sum(pred_labels == true_labels))}/{len(pred_labels)} correct)")
        print("(Expected ~0.5% for random 200-class predictions)")
    print()
    
    print("Sample predictions:")
    for i in range(min(5, len(pred_labels))):
        pred_class = int(pred_labels[i])
        true_class = int(true_labels[i])
        confidence = float(predictions[i, pred_class])
        print(f"  Sample {i}: Predicted={pred_class:3d} (conf={confidence:.4f})")
    print()
    
    print("Creating heatmap visualization...")
    fig, selected_classes = plot_prediction_heatmap(
        predictions=np.array(predictions),
        targets=np.array(targets),
        num_samples=min(config["num_samples"], len(predictions)),
        num_classes_to_show=config["num_classes_to_show"]
    )
    
    print(f"Displaying {len(selected_classes)} classes: {selected_classes[:10]}{'...' if len(selected_classes) > 10 else ''}")
    print(f"  (Includes all true labels + top predicted classes for each sample)")
    
    output_path = figures_dir / 'vit_prediction_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved to: {output_path}")
    print()
    
    print("Creating vertex state heatmap for a single sample...")
    fig2, selected_sample_idx = plot_vertex_state_heatmap(
        vertex_state=np.array(vertex_state),
        sample_idx=None  # Randomly select
    )
    print(f"Displaying detailed state for Sample {selected_sample_idx}")
    
    output_path2 = figures_dir / 'vit_vertex_state_heatmap.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"Vertex state heatmap saved to: {output_path2}")
    print()
    
    print("Displaying heatmaps...")
    print("Close the plot windows to exit.")
    plt.show()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize PC-ViT predictions on Tiny-ImageNet")
    parser.add_argument("--load_path", type=str, default=None,
                        help="Path to load a pre-trained model (e.g., './checkpoints/vit_model.eqx')")
    args = parser.parse_args()
    
    main(load_path=args.load_path)

