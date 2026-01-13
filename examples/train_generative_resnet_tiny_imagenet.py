import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

from pcm.network import ChainNetwork, Vertex, Edge
from pcm.blocks import (
    LinearBlock, ResidualBlock, ResidualUpBlock, 
    ConvBlock, ConvTransposeBlock
)
from pcm.datasets import get_tiny_imagenet_dataloaders


class ReshapeBlock(eqx.Module):
    linear: eqx.nn.Linear
    out_channels: int
    spatial_size: int
    
    def __init__(
        self,
        in_features: int,
        out_channels: int,
        spatial_size: int,
        key: jax.Array = None,
    ):
        if key is None:
            key = jr.PRNGKey(0)
        
        self.out_channels = out_channels
        self.spatial_size = spatial_size

        out_features = out_channels * spatial_size * spatial_size
        self.linear = eqx.nn.Linear(
            in_features=in_features,
            out_features=out_features,
            use_bias=True,
            key=key
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out = jax.vmap(self.linear)(x)
        batch_size = out.shape[0]
        out = out.reshape(batch_size, self.out_channels, self.spatial_size, self.spatial_size)
        out = jax.nn.relu(out)
        return out


class FinalConvBlock(eqx.Module):
    conv: eqx.nn.Conv2d
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        key: jax.Array = None,
    ):
        if key is None:
            key = jr.PRNGKey(0)
        
        self.conv = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=True,
            key=key,
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out = jax.vmap(self.conv)(x)
        # Use sigmoid to ensure output is in [0, 1]
        out = jax.nn.sigmoid(out)
        return out


def create_generative_resnet(
    key,
    latent_dim=128,
    initial_channels=512,
    initial_size=4,
    img_size=64,
):
    """
    Create a generative ResNet using ChainNetwork.
    
    Architecture:
    - Input: (1,) fixed to 0
    - Embedding: Linear to latent_dim
    - Reshape: latent_dim -> (initial_channels, initial_size, initial_size)
    - ResidualBlock at initial_size
    - ResidualUpBlock: initial_size -> initial_size*2  (4->8)
    - ResidualBlock at initial_size*2
    - ResidualUpBlock: initial_size*2 -> initial_size*4  (8->16)
    - ResidualBlock at initial_size*4
    - ResidualUpBlock: initial_size*4 -> initial_size*8  (16->32)
    - ResidualBlock at initial_size*8
    - ResidualUpBlock: initial_size*8 -> img_size  (32->64)
    - ResidualBlock at img_size
    - FinalConv: -> (3, 64, 64)
    """
    
    num_upsamples = int(jnp.log2(img_size / initial_size))
    keys = jr.split(key, 20)  # Enough keys for all layers
    key_idx = 0
    
    vertices = []
    edges = []
    
    vertices.append(Vertex(name="input", shape=(1,), fixed=True))
    vertices.append(Vertex(name="latent", shape=(latent_dim,), fixed=False))
    edges.append(Edge(
        from_v=vertices[-2],
        to_v=vertices[-1],
        forward_fn=LinearBlock(in_features=1, out_features=latent_dim, key=keys[key_idx])
    ))
    key_idx += 1
    
    current_channels = initial_channels
    current_size = initial_size
    vertices.append(Vertex(
        name="spatial_0",
        shape=(current_channels, current_size, current_size),
        fixed=False
    ))
    edges.append(Edge(
        from_v=vertices[-2],
        to_v=vertices[-1],
        forward_fn=ReshapeBlock(
            in_features=latent_dim,
            out_channels=current_channels,
            spatial_size=current_size,
            key=keys[key_idx]
        )
    ))
    key_idx += 1
    
    for i in range(num_upsamples):
        vertices.append(Vertex(
            name=f"res_block_{i}_a",
            shape=(current_channels, current_size, current_size),
            fixed=False
        ))
        edges.append(Edge(
            from_v=vertices[-2],
            to_v=vertices[-1],
            forward_fn=ResidualBlock(
                in_channels=current_channels,
                out_channels=current_channels,
                stride=1,
                key=keys[key_idx]
            )
        ))
        key_idx += 1
        
        next_size = current_size * 2
        next_channels = max(current_channels // 2, 64)  # Decrease channels as we upsample
        
        vertices.append(Vertex(
            name=f"upsample_{i}",
            shape=(next_channels, next_size, next_size),
            fixed=False
        ))
        edges.append(Edge(
            from_v=vertices[-2],
            to_v=vertices[-1],
            forward_fn=ResidualUpBlock(
                in_channels=current_channels,
                out_channels=next_channels,
                key=keys[key_idx]
            )
        ))
        key_idx += 1
        
        current_channels = next_channels
        current_size = next_size
    
    # Final ResidualBlock
    vertices.append(Vertex(
        name="res_block_final",
        shape=(current_channels, current_size, current_size),
        fixed=False
    ))
    edges.append(Edge(
        from_v=vertices[-2],
        to_v=vertices[-1],
        forward_fn=ResidualBlock(
            in_channels=current_channels,
            out_channels=current_channels,
            stride=1,
            key=keys[key_idx]
        )
    ))
    key_idx += 1
    
    # Output vertex (RGB image, fixed during training)
    vertices.append(Vertex(
        name="output",
        shape=(3, img_size, img_size),
        fixed=True
    ))
    edges.append(Edge(
        from_v=vertices[-2],
        to_v=vertices[-1],
        forward_fn=FinalConvBlock(
            in_channels=current_channels,
            out_channels=3,
            key=keys[key_idx]
        )
    ))
    
    network = ChainNetwork(edges=edges)
    return network


def plot_images_grid(images, rows=4, cols=4, title="Generated Images", save_path=None):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx < len(images):
            # images shape: (3, 64, 64), need to transpose to (64, 64, 3)
            img = images[idx].transpose(1, 2, 0)
            img = jnp.clip(img, 0, 1)
            ax.imshow(img)
        ax.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
    plt.close()


def generate_images(network, key, num_samples=16):
    batch_size = num_samples
    
    input_states = {
        "input": jnp.zeros((batch_size, 1))
    }
    
    # Forward pass with generative=True
    result = network.forward(
        input_states=input_states,
        returned_vertices=["output"],
        generative=True,
        key=key
    )
    
    return result["output"]


def train_generative_model(
    network,
    train_loader,
    key,
    epochs=10,
    train_lr=1e-4,
    inf_lr=0.05,
    inf_epoch=50,
    generate_every=50,
    num_generate=16,
):
    """Train the generative ResNet model."""
    
    train_opt = optax.adam(train_lr)
    weights = [edge.forward_fn for edge in network.edges]
    train_opt_state = train_opt.init(eqx.filter(weights, eqx.is_array))
    
    print("=" * 70)
    print("Training Generative ResNet on Tiny-ImageNet")
    print("=" * 70)
    print(f"Epochs: {epochs}")
    print(f"Training LR: {train_lr}")
    print(f"Inference LR: {inf_lr}")
    print(f"Inference epochs: {inf_epoch}")
    print("=" * 70)
    
    for epoch in range(epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"{'='*70}")
        
        batch_count = 0
        epoch_energy = 0.0
        
        for batch in train_loader:
            batch_count += 1
            key, subkey = jr.split(key)
            
            input_states = {
                "input": jnp.zeros((batch["input"].shape[0], 1)),
                "output": batch["input"]
            }
            
            train_opt_state, energy, _ = network.train_step(
                input_states=input_states,
                key=subkey,
                train_opt=train_opt,
                train_opt_state=train_opt_state,
                inf_lr=inf_lr,
                inf_epoch=inf_epoch,
            )
            
            epoch_energy += energy
            
            if batch_count % 10 == 0:
                avg_energy = epoch_energy / batch_count
                print(f"  Batch {batch_count:3d} | Avg Energy: {avg_energy:.6f}")
            
            if batch_count > 0 and batch_count % generate_every == 0:
                print(f"\n  Generating samples at Epoch {epoch + 1}, Batch {batch_count}...")
                key, gen_key = jr.split(key)
                generated = generate_images(network, gen_key, num_samples=num_generate)
                generated_np = np.array(generated)
                
                plot_images_grid(
                    generated_np,
                    rows=4,
                    cols=4,
                    title=f"Generated (Epoch {epoch + 1}, Batch {batch_count})"
                )
        
        print(f"\n  Final generation for Epoch {epoch + 1}...")
        key, gen_key = jr.split(key)
        generated = generate_images(network, gen_key, num_samples=num_generate)
        generated_np = np.array(generated)
        
        plot_images_grid(
            generated_np,
            rows=4,
            cols=4,
            title=f"Generated (Epoch {epoch + 1} - Final)"
        )
        
        avg_epoch_energy = epoch_energy / batch_count
        print(f"\nEpoch {epoch + 1} Average Energy: {avg_epoch_energy:.6f}")
    
    return network


def main():
    """Main function."""
    
    latent_dim = 128
    initial_channels = 512
    initial_size = 4
    img_size = 64
    batch_size = 64
    epochs = 10
    train_lr = 2e-3
    inf_lr = 0.05
    inf_epoch = 50
    num_train_samples = 10000  # Use subset for faster training
    
    key = jr.PRNGKey(42)
    key, model_key = jr.split(key)
    
    print("\nCreating Generative ResNet ChainNetwork...")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Initial: {initial_channels} channels at {initial_size}x{initial_size}")
    print(f"  Output: 3 channels at {img_size}x{img_size}")
    
    network = create_generative_resnet(
        key=model_key,
        latent_dim=latent_dim,
        initial_channels=initial_channels,
        initial_size=initial_size,
        img_size=img_size,
    )
    
    print(f"  Network created with {len(network.edges)} edges")
    print(f"  Input vertex: {network.input_vertex_name}")
    
    print("\n  Architecture:")
    for i, edge in enumerate(network.edges):
        from_shape = edge.from_v.shape
        to_shape = edge.to_v.shape
        print(f"    Edge {i}: {edge.from_v.name} {from_shape} -> {edge.to_v.name} {to_shape}")
    
    print("\nLoading Tiny-ImageNet dataset...")
    train_loader, val_loader = get_tiny_imagenet_dataloaders(
        batch_size=batch_size,
        img_size=img_size,
        num_train_samples=num_train_samples,
        shuffle_train=True
    )
    print(f"  Train batches: {len(train_loader)}")
    
    print("\n" + "=" * 70)
    print("Generating samples BEFORE training...")
    print("=" * 70)
    key, gen_key = jr.split(key)
    generated = generate_images(network, gen_key, num_samples=16)
    generated_np = np.array(generated)
    plot_images_grid(generated_np, rows=4, cols=4, title="Before Training")
    
    key, train_key = jr.split(key)
    network = train_generative_model(
        network=network,
        train_loader=train_loader,
        key=train_key,
        epochs=epochs,
        train_lr=train_lr,
        inf_lr=inf_lr,
        inf_epoch=inf_epoch,
        generate_every=50,
        num_generate=16
    )
    
    print("\n" + "=" * 70)
    print("Generating samples AFTER training...")
    print("=" * 70)
    key, gen_key = jr.split(key)
    generated = generate_images(network, gen_key, num_samples=16)
    generated_np = np.array(generated)
    plot_images_grid(generated_np, rows=4, cols=4, title="After Training - Final Results")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    return network


if __name__ == "__main__":
    network = main()

