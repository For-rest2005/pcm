"""
Predictive Coding Vision Transformer (PC-ViT) implementation.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from typing import Optional, List, Tuple
import numpy as np

from .network import Vertex, Edge, ChainNetwork
from .blocks import TransformerBlock, PatchEmbedding, InputTransformerBlock, OutputTransformerBlock


class PCViT:
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        num_classes: int = 200,
        embed_dim: int = 384,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        key: jax.Array = jr.PRNGKey(42),
    ):
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.num_patches = (img_size // patch_size) ** 2
        
        keys = jr.split(key, num_layers + 2)
        
        vertices = []
        
        input_vertex = Vertex(name="input", shape=(in_channels, img_size, img_size), fixed=True)
        vertices.append(input_vertex)
        
        patch_vertex = Vertex(name="patches", shape=(self.num_patches, embed_dim), fixed=False)
        vertices.append(patch_vertex)
        
        for i in range(num_layers):
            vertex = Vertex(name=f"layer_{i}", shape=(self.num_patches + 1, embed_dim), fixed=False)
            vertices.append(vertex)
        
        output_vertex = Vertex(name="output", shape=(num_classes,), fixed=True)
        vertices.append(output_vertex)
        
        edges = []
        
        patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            key=keys[0],
        )
        
        edge = Edge(
            from_v=vertices[0],
            to_v=vertices[1],
            forward_fn=patch_embed,
            energy_ratio=1.0,
        )
        edges.append(edge)
        
        total_edges = num_layers + 2
        input_transformer = InputTransformerBlock(
            num_patches=self.num_patches,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            total_layers=total_edges,
            key=keys[1],
        )
        
        edge = Edge(
            from_v=vertices[1],
            to_v=vertices[2],
            forward_fn=input_transformer,
            energy_ratio=1.0,
        )
        edges.append(edge)
        
        for i in range(1, num_layers):
            transformer = TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                total_layers=total_edges,
                key=keys[i + 1],
            )
            
            edge = Edge(
                from_v=vertices[i + 1],  # layer_{i-1}
                to_v=vertices[i + 2],    # layer_{i}
                forward_fn=transformer,
                energy_ratio=1.0,
            )
            edges.append(edge)
        
        output_transformer = OutputTransformerBlock(
            embed_dim=embed_dim,
            num_classes=num_classes,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            total_layers=total_edges,
            key=keys[-1],
        )
        
        edge = Edge(
            from_v=vertices[-2],
            to_v=vertices[-1],
            forward_fn=output_transformer,
            energy_ratio=1.0,
        )
        edges.append(edge)
        
        self.network = ChainNetwork(edges=edges)
        self.vertices = vertices
        self.edges = edges
    
    
    def train(
        self,
        train_data: List[dict],
        key: jax.Array = jr.PRNGKey(0),
        epochs: int = 10,
        train_lr: float = 1e-3,
        inf_lr: float = 0.1,
        inf_epoch: int = 50,
        verbose: bool = True,
        print_times_every_epoch: int = 10,
    ):
        train_opt = optax.adamw(train_lr)
        weights = [edge.forward_fn for edge in self.network.edges]
        train_opt_state = train_opt.init(eqx.filter(weights, eqx.is_array))
        
        energy_history = []
        
        for epoch in range(epochs):
            epoch_energy = 0.0
            num_batches = len(train_data)
            
            if verbose:
                print_interval = max(1, num_batches // print_times_every_epoch)
            
            for batch_idx, input_states in enumerate(train_data):
                key, subkey = jr.split(key)
                
                train_opt_state, energy, _ = self.network.train_step(
                    input_states=input_states,
                    key=subkey,
                    returned_vertices=None,
                    init_fun=jr.normal,
                    train_opt=train_opt,
                    train_opt_state=train_opt_state,
                    inf_lr=inf_lr,
                    inf_epoch=inf_epoch,
                )
                
                if verbose and (batch_idx + 1) % print_interval == 0:
                    print(f'  Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{num_batches}, Energy: {energy:.6f}')

        return energy_history
    
    def predict(
        self,
        images: jnp.ndarray,
    ) -> jnp.ndarray:
        input_states = {"input": images}
        output_states = self.network.forward(
            input_states=input_states,
            returned_vertices=["output"]
        )
        
        return output_states["output"]
    
    def evaluate_accuracy(
        self,
        test_data: List[dict],
        verbose: bool = True,
    ) -> float:
        correct = 0
        total = 0
        
        if verbose:
            print(f"Evaluating on {len(test_data)} batches using forward pass...")
        
        for batch_idx, batch in enumerate(test_data):
            images = batch["input"]
            targets = batch["output"]
            
            predictions = self.predict(images)
            
            pred_labels = jnp.argmax(predictions, axis=1)
            true_labels = jnp.argmax(targets, axis=1)
            batch_correct = jnp.sum(pred_labels == true_labels)
            correct += batch_correct
            total += images.shape[0]
            
            if verbose and (batch_idx + 1) % 10 == 0:
                current_acc = float(correct) / total
                print(f"  Batch {batch_idx + 1}/{len(test_data)}: Current accuracy = {current_acc * 100:.2f}%")
        
        accuracy = float(correct) / total
        
        if verbose:
            print(f"  Final accuracy: {accuracy * 100:.2f}% ({correct}/{total} correct)")
        
        return accuracy

