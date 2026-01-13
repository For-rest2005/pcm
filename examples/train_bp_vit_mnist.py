"""
Baseline: Standard Backpropagation-trained Vision Transformer on MNIST
Uses the same blocks as PC-ViT but trained with standard gradient descent
Configuration: 2 transformer layers, 4 attention heads, 4x4 patch size
"""
import argparse
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
from typing import Optional
from pcm.datasets import get_mnist_dataloaders
from pcm.blocks import (
    MultiHeadAttention,
    TransformerBlock,
    PatchEmbedding,
    InputTransformerBlock,
)

class StandardViT(eqx.Module):
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
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        key: jax.Array = jr.PRNGKey(42),
    ):
        num_patches = (img_size // patch_size) ** 2
        total_layers = num_layers
        
        keys = jr.split(key, num_layers + 3)
        
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            key=keys[0],
        )
        
        self.input_transformer = InputTransformerBlock(
            num_patches=num_patches,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            total_layers=total_layers,
            key=keys[1],
        )
        
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
        
        self.norm = eqx.nn.LayerNorm(embed_dim)
        self.head = eqx.nn.Linear(embed_dim, num_classes, use_bias=True, key=keys[-1])
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self.patch_embed(x)
        
        x = self.input_transformer(x)
        
        for layer in self.transformer_layers:
            x = layer(x)
        
        cls_token = x[:, 0, :]  # (B, embed_dim)
        cls_token = jax.vmap(self.norm)(cls_token)
        logits = jax.vmap(self.head)(cls_token)
        
        return logits

def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(labels * log_probs) / labels.shape[0]
    return loss

def mse_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = jnp.mean((log_probs - labels) ** 2)
    return loss

def compute_accuracy(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    pred_labels = jnp.argmax(logits, axis=-1)
    true_labels = jnp.argmax(labels, axis=-1)
    accuracy = jnp.mean(pred_labels == true_labels)
    return accuracy


@eqx.filter_jit
def train_step(model, opt_state, optimizer, images, labels):
    def loss_fn(model):
        logits = model(images)
        loss = cross_entropy_loss(logits, labels)
        # loss = mse_loss(logits, labels)
        return loss, logits
    
    (loss, logits), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    
    accuracy = compute_accuracy(logits, labels)
    
    return model, opt_state, loss, accuracy


@eqx.filter_jit
def eval_step(model, images, labels):
    logits = model(images)
    loss = cross_entropy_loss(logits, labels)
    # loss = mse_loss(logits, labels)
    accuracy = compute_accuracy(logits, labels)
    return loss, accuracy


def main(save_path=None, load_path=None):
    print("=" * 60)
    print("Standard BP-ViT Training - MNIST")
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
        "learning_rate": 1e-3,
        "weight_decay": 0.01,
        
        "seed": 42,
        "print_every": 50,
        "eval_every": 50,
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
    )
    print()
    
    print("Initializing Standard ViT model...")
    key = jr.PRNGKey(config["seed"])
    model = StandardViT(
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
    print(f"  Number of patches: {num_patches} (each {config['patch_size']}x{config['patch_size']})")
    print(f"  Embedding dimension: {config['embed_dim']}")
    print(f"  Transformer layers: {config['num_layers']}")
    print(f"  Attention heads: {config['num_heads']}")
    print(f"  Output classes: {config['num_classes']}")
    print()
    
    if load_path:
        print(f"Loading pre-trained model from: {load_path}")
        print("-" * 60)
        model = eqx.tree_deserialise_leaves(load_path, model)
        print("  Model loaded successfully!")
        print()
    
    optimizer = optax.adamw(
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    print("Starting training...")
    print(f"Training with standard backpropagation")
    print(f"Evaluate accuracy every {config['eval_every']} batches")
    print()
    
    total_batches = 0
    batches_per_epoch = len(train_loader)
    
    for epoch in range(config["epochs"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print(f"{'='*60}")
        
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch["input"]
            labels = batch["output"]
            
            model, opt_state, loss, accuracy = train_step(
                model, opt_state, optimizer, images, labels
            )
            
            epoch_loss += float(loss)
            epoch_acc += float(accuracy)
            total_batches += 1
            
            if (batch_idx + 1) % config["print_every"] == 0:
                print(f"  Batch {batch_idx + 1}/{batches_per_epoch}, "
                      f"Loss: {float(loss):.6f}, Accuracy: {float(accuracy) * 100:.2f}%")
            
            if total_batches % config["eval_every"] == 0:
                print(f"\n{'─'*60}")
                print(f"Evaluation - {total_batches} batches (Epoch {epoch+1}, Batch {batch_idx+1})")
                print(f"{'─'*60}")
                
                val_subset = []
                for i, val_batch in enumerate(test_loader):
                    val_subset.append(val_batch)
                    if i >= 19:
                        break
                
                val_loss = 0.0
                val_acc = 0.0
                pred_count = [0] * 10
                correct_per_class = [0] * 10
                
                for val_batch in val_subset:
                    val_images = val_batch["input"]
                    val_labels = val_batch["output"]
                    
                    loss, accuracy = eval_step(model, val_images, val_labels)
                    val_loss += float(loss)
                    val_acc += float(accuracy)
                    
                    logits = model(val_images)
                    pred_labels = jnp.argmax(logits, axis=-1)
                    true_labels = jnp.argmax(val_labels, axis=-1)
                    
                    for pred_label in pred_labels:
                        pred_count[int(pred_label)] += 1
                    
                    for pred_label, true_label in zip(pred_labels, true_labels):
                        if int(pred_label) == int(true_label):
                            correct_per_class[int(pred_label)] += 1
                
                val_loss /= len(val_subset)
                val_acc /= len(val_subset)
                
                print(f"Validation Loss: {val_loss:.6f}")
                print(f"Validation Accuracy: {val_acc * 100:.2f}%")
                print(f"\nPrediction distribution:")
                
                total_preds = sum(pred_count)
                for class_id in range(10):
                    count = pred_count[class_id]
                    correct_count = correct_per_class[class_id]
                    percentage = (count / total_preds * 100) if total_preds > 0 else 0
                    correct_rate = (correct_count / count * 100) if count > 0 else 0
                    print(f"  Number {class_id}: {count:4d} predictions ({percentage:5.2f}%), "
                          f"{correct_count:3d} correct ({correct_rate:5.2f}%)")
                
                print(f"{'─'*60}\n")
        
        print(f"\nEpoch {epoch + 1} completed.")
    
    print("\nTraining completed!")
    print()
    
    if save_path:
        print(f"Saving model to: {save_path}")
        print("-" * 60)
        eqx.tree_serialise_leaves(save_path, model)
        print("  Model saved successfully!")
        print()
    
    print("Evaluating on the full test set...")
    test_loss = 0.0
    test_acc = 0.0
    
    for batch in test_loader:
        images = batch["input"]
        labels = batch["output"]
        loss, accuracy = eval_step(model, images, labels)
        test_loss += float(loss)
        test_acc += float(accuracy)
    
    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    
    print(f"  Test Loss: {test_loss:.6f}")
    print(f"  Test Accuracy: {test_acc * 100:.2f}%")
    print()
    
    print("=" * 60)
    print("Training Summary")
    print("=" * 60)
    print(f"Configuration: {config['num_layers']} layers, {config['embed_dim']} dim, {config['num_heads']} heads")
    print(f"Training samples: 60000")
    print(f"Test samples: 10000")
    print(f"Training epochs: {config['epochs']}")
    print(f"Final test loss: {test_loss:.6f}")
    print(f"Final test accuracy: {test_acc * 100:.2f}%")
    print("=" * 60)
    
    return model, test_loss, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train standard BP-ViT on MNIST")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save the trained model (e.g. './checkpoints/bp_vit_mnist.eqx')")
    parser.add_argument("--load_path", type=str, default=None,
                        help="Path to load a pre-trained model (e.g. './checkpoints/bp_vit_mnist.eqx')")
    args = parser.parse_args()
    
    main(save_path=args.save_path, load_path=args.load_path)

