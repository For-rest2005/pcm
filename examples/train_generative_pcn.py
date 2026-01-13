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
from pcm.blocks import AffineActivation, LinearBlock
from pcm.datasets import get_mnist_dataloaders


def create_generative_network(key, latent_dim=20, hidden_dims=[400, 400]):
    keys = jr.split(key, len(hidden_dims) + 2)
    
    vertices = []
    vertices.append(Vertex(name="input", shape=(1,), fixed=True))
    vertices.append(Vertex(name="latent", shape=(latent_dim,), fixed=False))
    
    for i, hidden_dim in enumerate(hidden_dims):
        vertices.append(Vertex(name=f"hidden_{i}", shape=(hidden_dim,), fixed=False))
    
    vertices.append(Vertex(name="output", shape=(784,), fixed=True))
    
    edges = []
    
    edges.append(Edge(
        from_v=vertices[0],
        to_v=vertices[1],
        forward_fn=LinearBlock(in_features=1, out_features=latent_dim, key=keys[0])
    ))
    
    edges.append(Edge(
        from_v=vertices[1],
        to_v=vertices[2],
        forward_fn=AffineActivation(
            in_features=latent_dim,
            out_features=hidden_dims[0],
            activation=jax.nn.relu,
            key=keys[1]
        )
    ))
    
    for i in range(len(hidden_dims) - 1):
        edges.append(Edge(
            from_v=vertices[i + 2],
            to_v=vertices[i + 3],
            forward_fn=AffineActivation(
                in_features=hidden_dims[i],
                out_features=hidden_dims[i + 1],
                activation=jax.nn.relu,
                key=keys[i + 2]
            )
        ))
    
    edges.append(Edge(
        from_v=vertices[-2],
        to_v=vertices[-1],
        forward_fn=AffineActivation(
            in_features=hidden_dims[-1],
            out_features=784,
            activation=jax.nn.sigmoid,
            key=keys[-1]
        )
    ))
    
    return ChainNetwork(edges=edges)


def plot_mnist_grid(images, rows=2, cols=4, title="Generated Images", save_path=None):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()
    
    for idx, ax in enumerate(axes):
        if idx < len(images):
            img = images[idx].reshape(28, 28)
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()
    plt.close()


def generate_images(network, key, num_samples=8):
    batch_size = num_samples
    
    input_states = {
        "input": jnp.zeros((batch_size, 1)),
        "output": jnp.zeros((batch_size, 784))
    }
    
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
    epochs=3,
    train_lr=1e-3,
    inf_lr=0.1,
    inf_epoch=100,
    generate_every=20,
    num_generate=8
):
    train_opt = optax.adam(train_lr)
    weights = [edge.forward_fn for edge in network.edges]
    train_opt_state = train_opt.init(eqx.filter(weights, eqx.is_array))
    
    print("=" * 70)
    print("Training Generative PCN")
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
                
                plot_mnist_grid(
                    generated_np,
                    rows=2,
                    cols=4,
                    title=f"Generated (Epoch {epoch + 1}, Batch {batch_count})"
                )
        
        print(f"\n  Final generation for Epoch {epoch + 1}...")
        key, gen_key = jr.split(key)
        generated = generate_images(network, gen_key, num_samples=num_generate)
        generated_np = np.array(generated)
        
        plot_mnist_grid(
            generated_np,
            rows=2,
            cols=4,
            title=f"Generated (Epoch {epoch + 1} - Final)"
        )
        
        avg_epoch_energy = epoch_energy / batch_count
        print(f"\nEpoch {epoch + 1} Average Energy: {avg_epoch_energy:.6f}")
    
    return network


def main():
    """Main function to train and generate."""
    
    latent_dim = 20
    hidden_dims = [400, 400]
    batch_size = 256
    epochs = 3
    train_lr = 0.001
    inf_lr = 0.05
    inf_epoch = 100
    
    key = jr.PRNGKey(42)
    key, model_key = jr.split(key)
    
    print(f"  Architecture: [1] -> [{latent_dim}] -> {hidden_dims} -> [784]")
    network = create_generative_network(
        key=model_key,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims
    )
    print(f"  Input vertex: {network.input_vertex_name}")
    
    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=batch_size,
        flatten=True,
        one_hot=False,
        shuffle_train=True
    )
    print(f"  Train batches: {len(train_loader)}")
    
    # Generate before training
    print("\n" + "=" * 70)
    print("Generating samples BEFORE training...")
    print("=" * 70)
    key, gen_key = jr.split(key)
    generated = generate_images(network, gen_key, num_samples=8)
    generated_np = np.array(generated)
    plot_mnist_grid(generated_np, rows=2, cols=4, title="Before Training")
    
    # Train
    key, train_key = jr.split(key)
    network = train_generative_model(
        network=network,
        train_loader=train_loader,
        key=train_key,
        epochs=epochs,
        train_lr=train_lr,
        inf_lr=inf_lr,
        inf_epoch=inf_epoch,
        generate_every=20,
        num_generate=8
    )
    
    # Generate after training
    print("\n" + "=" * 70)
    print("Generating samples AFTER training...")
    print("=" * 70)
    key, gen_key = jr.split(key)
    generated = generate_images(network, gen_key, num_samples=16)
    generated_np = np.array(generated)
    plot_mnist_grid(generated_np, rows=4, cols=4, title="After Training - Final Results")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    return network


if __name__ == "__main__":
    network = main()

