import sys
from pathlib import Path

import jax.numpy as jnp
import jax.random as jr

from pcm.network import ChainNetwork
import train_generative_resnet_tiny_imagenet as gen_resnet

def test_architecture():
    """Test if the architecture is built correctly."""
    print("Testing Generative ResNet Architecture...")
    print("=" * 70)
    
    key = jr.PRNGKey(0)
    
    # Create network
    network = gen_resnet.create_generative_resnet(
        key=key,
        latent_dim=128,
        initial_channels=512,
        initial_size=4,
        img_size=64,
    )
    
    print(f"✓ Network created successfully")
    print(f"  Number of edges: {len(network.edges)}")
    print(f"  Number of vertices: {len(network.vertices)}")
    print(f"  Input vertex: {network.input_vertex_name}")
    
    # Print architecture details
    print("\nArchitecture:")
    for i, edge in enumerate(network.edges):
        from_v = edge.from_v
        to_v = edge.to_v
        forward_fn_type = type(edge.forward_fn).__name__
        print(f"  {i:2d}. {from_v.name:20s} {str(from_v.shape):20s} -> "
              f"{to_v.name:20s} {str(to_v.shape):20s} [{forward_fn_type}]")
    
    print("\n" + "=" * 70)
    print("Testing forward pass...")
    batch_size = 4
    
    input_states = {
        "input": jnp.zeros((batch_size, 1)),
    }
    
    key, subkey = jr.split(key)
    result = network.forward(
        input_states=input_states,
        returned_vertices=["output"],
        generative=True,
        key=subkey
    )
    
    output = result["output"]
    print(f"✓ Forward pass successful")
    print(f"  Output shape: {output.shape}")
    print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test generation
    print("\n" + "=" * 70)
    print("Testing generation...")
    key, subkey = jr.split(key)
    generated = gen_resnet.generate_images(network, subkey, num_samples=8)
    print(f"✓ Generation successful")
    print(f"  Generated shape: {generated.shape}")
    print(f"  Generated range: [{generated.min():.3f}, {generated.max():.3f}]")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)

if __name__ == "__main__":
    test_architecture()

