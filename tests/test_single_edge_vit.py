"""
Test PC training with a single-edge ChainNetwork.
The single edge contains a complete ViT model as its forward function.
This helps debug whether PC can work in the simplest possible setup.
"""
import argparse
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
import sys
sys.path.insert(0, '/home/lu/Code/AtoL_proj/pcm/src')

from pcm.network import Vertex, Edge, ChainNetwork
from pcm.datasets import get_mnist_dataloaders
from pcm.blocks import (
    PatchEmbedding,
    InputTransformerBlock,
    TransformerBlock,
)


class SimpleViT(eqx.Module):
    """
    A simple ViT that takes images and outputs class logits.
    This will be used as a single forward_fn in a ChainNetwork edge.
    """
    patch_embed: PatchEmbedding
    input_transformer: InputTransformerBlock
    transformer_layers: list
    norm: eqx.nn.LayerNorm
    head: eqx.nn.Linear
    
    def __init__(
        self,
        img_size: int = 28,
        patch_size: int = 4,
        in_channels: int = 1,
        num_classes: int = 10,
        embed_dim: int = 128,
        num_layers: int = 1,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        key: jax.Array = jr.PRNGKey(42),
    ):
        num_patches = (img_size // patch_size) ** 2
        total_layers = num_layers
        
        keys = jr.split(key, num_layers + 3)
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            key=keys[0],
        )
        
        # Input transformer (adds CLS token and positional embedding)
        self.input_transformer = InputTransformerBlock(
            num_patches=num_patches,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            total_layers=total_layers,
            key=keys[1],
        )
        
        # Additional transformer layers
        self.transformer_layers = []
        for i in range(num_layers - 1):
            layer = TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                total_layers=total_layers,
                key=keys[i + 2],
            )
            self.transformer_layers.append(layer)
        
        # Classification head
        self.norm = eqx.nn.LayerNorm(embed_dim)
        self.head = eqx.nn.Linear(embed_dim, num_classes, use_bias=True, key=keys[-1])
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass: images -> class probabilities
        
        Args:
            x: Input images of shape (batch_size, channels, height, width)
        
        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Input transformer (adds CLS token and applies first transformer)
        x = self.input_transformer(x)  # (B, num_patches+1, embed_dim)
        
        # Additional transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Extract CLS token and classify
        cls_token = x[:, 0, :]  # (B, embed_dim)
        cls_token = jax.vmap(self.norm)(cls_token)
        logits = jax.vmap(self.head)(cls_token)  # (B, num_classes)
        
        # Apply softmax to get probabilities (to match PC-ViT output format)
        probs = jax.nn.softmax(logits, axis=-1)
        
        return probs


def main(save_path=None, load_path=None):
    print("=" * 60)
    print("Single-Edge PC Network Test - MNIST")
    print("=" * 60)
    print()
    
    config = {
        "img_size": 28,
        "patch_size": 4,
        "in_channels": 1,
        "num_classes": 10,
        "embed_dim": 128,
        "num_layers": 1,
        "num_heads": 4,
        "mlp_ratio": 4,
        
        "batch_size": 128,
        "epochs": 10,
        "train_lr": 1e-2,
        "inf_lr": 1e-1,
        "inf_epoch": 1,
        
        "seed": 42,
        "print_times_every_epoch": 10,
        
        # Overfit test mode
        "overfit_test": True,
        "overfit_samples": 64,
    }
    
    print("Configuration parameters:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    num_patches = (config["img_size"] // config["patch_size"]) ** 2
    print(f"Image will be split into {num_patches} patches (each {config['patch_size']}x{config['patch_size']})")
    print()
    
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=config["batch_size"],
        flatten=False,
        one_hot=True,
        seed=config["seed"]
    )
    
    # Overfit test mode: use only a tiny subset of data
    if config["overfit_test"]:
        print(f"OVERFIT TEST MODE ENABLED")
        print(f"Using only {config['overfit_samples']} samples for training and testing")
        print(f"This is designed to test if the model can overfit on a tiny dataset")
        print()
        
        first_batch = next(iter(train_loader))
        images = first_batch["input"][:config["overfit_samples"]]
        labels = first_batch["output"][:config["overfit_samples"]]
        
        small_dataset = [{"input": images, "output": labels}]
        train_data = small_dataset
        test_data = small_dataset
        
        print(f"Dataset size: {images.shape[0]} samples")
    else:
        train_data = train_loader
        test_data = test_loader
    
    print()
    
    print("Initializing Single-Edge PC Network...")
    print("Architecture: Input -> [ViT] -> Output")
    print()
    
    key = jr.PRNGKey(config["seed"])
    key, model_key = jr.split(key)
    
    # Create the ViT model that will be the forward function
    vit_model = SimpleViT(
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
        embed_dim=config["embed_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        key=model_key,
    )
    
    # Create vertices
    input_vertex = Vertex(
        name="input",
        shape=(config["in_channels"], config["img_size"], config["img_size"]),
        fixed=True
    )
    
    output_vertex = Vertex(
        name="output",
        shape=(config["num_classes"],),
        fixed=True
    )
    
    # Create the single edge with ViT as forward function
    edge = Edge(
        from_v=input_vertex,
        to_v=output_vertex,
        forward_fn=vit_model,
        energy_ratio=1.0,
    )
    
    # Create ChainNetwork with single edge
    network = ChainNetwork(edges=[edge])
    
    print(f"Network created:")
    print(f"  Vertices: {list(network.vertices.keys())}")
    print(f"  Edges: 1 (input -> output)")
    print(f"  Forward function: Complete ViT model")
    print()
    
    if load_path:
        print(f"Loading pre-trained model from: {load_path}")
        print("-" * 60)
        network.load(load_path)
        print("  Model loaded successfully!")
        print()
    
    print("Starting training with Predictive Coding...")
    if config["overfit_test"]:
        print(f"OVERFIT MODE: Training on {config['overfit_samples']} samples only")
    print()
    
    key, subkey = jr.split(key)
    
    train_opt = optax.adamw(config["train_lr"])
    weights = [edge.forward_fn for edge in network.edges]
    train_opt_state = train_opt.init(eqx.filter(weights, eqx.is_array))
    
    energy_history = []
    total_batches = 0
    batches_per_epoch = len(train_data)
    
    for epoch in range(config["epochs"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print(f"{'='*60}")
        
        epoch_energy = 0.0
        
        for batch_idx, batch in enumerate(train_data):
            key, subkey = jr.split(key)
            
            train_opt_state, energy, _ = network.train_step(
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
            
            if config["overfit_test"] or (batch_idx + 1) % max(1, batches_per_epoch // config["print_times_every_epoch"]) == 0:
                print(f"  Batch {batch_idx + 1}/{batches_per_epoch}, "
                      f"Energy: {energy:.6f}")
            
            # Evaluation
            eval_frequency = 1 if config["overfit_test"] else 50
            if total_batches % eval_frequency == 0:
                print(f"\n{'─'*60}")
                print(f"Evaluate - {total_batches} batches (Epoch {epoch+1}, Batch {batch_idx+1})")
                print(f"{'─'*60}")
                
                # Get validation subset
                if config["overfit_test"]:
                    val_subset = test_data
                else:
                    val_subset = []
                    for i, val_batch in enumerate(test_data):
                        val_subset.append(val_batch)
                        if i >= 19:
                            break
                
                correct = 0
                total = 0
                pred_count = [0] * 10
                correct_per_class = [0] * 10
                
                for val_batch in val_subset:
                    images = val_batch["input"]
                    targets = val_batch["output"]
                    
                    # Use forward pass to get predictions
                    output_states = network.forward(
                        input_states={"input": images},
                        returned_vertices=["output"]
                    )
                    predictions = output_states["output"]
                    
                    pred_labels = jnp.argmax(predictions, axis=1)
                    true_labels = jnp.argmax(targets, axis=1)
                    
                    for pred_label in pred_labels:
                        pred_count[int(pred_label)] += 1
                    
                    for pred_label, true_label in zip(pred_labels, true_labels):
                        if int(pred_label) == int(true_label):
                            correct_per_class[int(pred_label)] += 1
                    
                    correct += jnp.sum(pred_labels == true_labels)
                    total += images.shape[0]
                
                accuracy = float(correct) / total
                
                print(f"Validation accuracy: {accuracy * 100:.2f}% ({correct}/{total} correct)")
                print(f"Current energy: {energy:.6f}")
                print(f"\nPrediction distribution:")
                for class_id in range(10):
                    count = pred_count[class_id]
                    correct_count = correct_per_class[class_id]
                    percentage = (count / total) * 100
                    correct_rate = (correct_count / count * 100) if count > 0 else 0
                    print(f"  Number {class_id}: {count:4d} predictions ({percentage:5.2f}%), "
                          f"{correct_count:3d} correct ({correct_rate:5.2f}%)")
                
                print(f"{'─'*60}\n")
        
        avg_epoch_energy = epoch_energy / batches_per_epoch
        print(f"\nEpoch {epoch + 1} completed. Average energy: {avg_epoch_energy:.6f}")
    
    print("\nTraining completed!")
    print()
    
    if save_path:
        print(f"Saving model to: {save_path}")
        print("-" * 60)
        network.save(save_path)
        print("  Model saved successfully!")
        print()
    
    print("Final evaluation on training set...")
    correct = 0
    total = 0
    
    for batch in train_data:
        images = batch["input"]
        targets = batch["output"]
        
        output_states = network.forward(
            input_states={"input": images},
            returned_vertices=["output"]
        )
        predictions = output_states["output"]
        
        pred_labels = jnp.argmax(predictions, axis=1)
        true_labels = jnp.argmax(targets, axis=1)
        correct += jnp.sum(pred_labels == true_labels)
        total += images.shape[0]
    
    train_accuracy = float(correct) / total
    print(f"Training accuracy: {train_accuracy * 100:.2f}% ({correct}/{total} correct)")
    print()
    
    print("Final evaluation on test set...")
    correct = 0
    total = 0
    
    for batch in test_data:
        images = batch["input"]
        targets = batch["output"]
        
        output_states = network.forward(
            input_states={"input": images},
            returned_vertices=["output"]
        )
        predictions = output_states["output"]
        
        pred_labels = jnp.argmax(predictions, axis=1)
        true_labels = jnp.argmax(targets, axis=1)
        correct += jnp.sum(pred_labels == true_labels)
        total += images.shape[0]
    
    test_accuracy = float(correct) / total
    print(f"Test accuracy: {test_accuracy * 100:.2f}% ({correct}/{total} correct)")
    print()
    
    print("=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Network: Single-Edge PC (Input -> [ViT] -> Output)")
    print(f"Configuration: {config['num_layers']} ViT layers, {config['embed_dim']} dim, {config['num_heads']} heads")
    if config["overfit_test"]:
        print(f"⚠️  OVERFIT TEST MODE: {config['overfit_samples']} samples")
        print(f"Training samples: {config['overfit_samples']}")
        print(f"Test samples: {config['overfit_samples']} (same as training)")
    else:
        print(f"Training samples: 60000")
        print(f"Test samples: 10000")
    print(f"Training epochs: {config['epochs']}")
    print(f"Final training energy: {energy_history[-1]:.6f}")
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    if config["overfit_test"] and train_accuracy > 0.95:
        print(f"✓ Model successfully overfitted on tiny dataset!")
    elif config["overfit_test"] and train_accuracy < 0.95:
        print(f"✗ Model failed to overfit on tiny dataset. This suggests a problem with PC training.")
    print("=" * 60)
    print()
    
    return network, energy_history, train_accuracy, test_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PC with single-edge ViT network on MNIST")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save the trained model (e.g. './checkpoints/single_edge_vit.eqx')")
    parser.add_argument("--load_path", type=str, default=None,
                        help="Path to load a pre-trained model (e.g. './checkpoints/single_edge_vit.eqx')")
    args = parser.parse_args()
    
    main(save_path=args.save_path, load_path=args.load_path)

