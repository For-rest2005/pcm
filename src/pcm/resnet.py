"""
Predictive Coding ResNet implementation.
ResNet56 architecture for Tiny-ImageNet classification.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from typing import Optional, List, Tuple
import numpy as np

from .network import Vertex, Edge, ChainNetwork


class ConvBlock(eqx.Module):
    conv: eqx.nn.Conv2d
    total_layers: int
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        total_layers: int = 1,
        key: Optional[jax.Array] = None,
    ):
        if key is None:
            key = jr.PRNGKey(0)
        
        self.conv = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_bias=False,
            key=key,
        )
        self.total_layers = total_layers
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out = jax.vmap(self.conv)(x)
        out = jax.nn.relu(out)
        return out


class ResidualBlock(eqx.Module):
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    shortcut: Optional[eqx.nn.Conv2d]
    gamma: jnp.ndarray
    stride: int
    total_layers: int
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        total_layers: int = 1,
        key: Optional[jax.Array] = None,
    ):
        if key is None:
            key = jr.PRNGKey(0)
        
        keys = jr.split(key, 3)
        self.stride = stride
        self.total_layers = total_layers
        self.conv1 = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            use_bias=False,
            key=keys[0],
        )
        
        self.conv2 = eqx.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            key=keys[1],
        )
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = eqx.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=0,
                use_bias=False,
                key=keys[2],
            )
        else:
            self.shortcut = None
        
        init_value = 0.0
        self.gamma = jnp.full((out_channels, 1, 1), init_value)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        identity = x
        
        out = jax.vmap(self.conv1)(x)
        out = jax.nn.relu(out)
        
        out = jax.vmap(self.conv2)(out)
        
        if self.shortcut is not None:
            identity = jax.vmap(self.shortcut)(identity)
    
        out = identity + self.gamma * out
        out = jax.nn.relu(out)
        
        return out


class ResidualStage(eqx.Module):
    """
    A stage consisting of multiple residual blocks.
    """
    blocks: List[ResidualBlock]
    
    def __init__(
        self,
        num_blocks: int,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        total_layers: int = 1,
        key: Optional[jax.Array] = None,
    ):
        if key is None:
            key = jr.PRNGKey(0)
        
        keys = jr.split(key, num_blocks)
        
        blocks = []
        blocks.append(ResidualBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            total_layers=total_layers,
            key=keys[0],
        ))
        
        for i in range(1, num_blocks):
            blocks.append(ResidualBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                stride=1,
                total_layers=total_layers,
                key=keys[i],
            ))
        
        self.blocks = blocks
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for block in self.blocks:
            x = block(x)
        return x


class GlobalAvgPoolAndFC(eqx.Module):
    fc: eqx.nn.Linear
    total_layers: int
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        total_layers: int = 1,
        key: Optional[jax.Array] = None,
    ):
        if key is None:
            key = jr.PRNGKey(0)
        
        self.fc = eqx.nn.Linear(
            in_features=in_channels,
            out_features=num_classes,
            use_bias=True,
            key=key,
        )
        self.total_layers = total_layers
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        pooled = jnp.mean(x, axis=(2, 3))
        
        logits = jax.vmap(self.fc)(pooled)
        
        probs = jax.nn.softmax(logits, axis=-1)
        
        return probs


class PCResNet56:
    def __init__(
        self,
        img_size: int = 64,
        in_channels: int = 3,
        num_classes: int = 200,
        key: jax.Array = jr.PRNGKey(42),
    ):
        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        total_layers = 56
        
        keys = jr.split(key, 5)
        
        vertices = []
        
        # Input vertex (fixed during training)
        input_vertex = Vertex(name="input", shape=(in_channels, img_size, img_size), fixed=True)
        vertices.append(input_vertex)
        
        conv1_vertex = Vertex(name="conv1", shape=(16, img_size, img_size), fixed=False)
        vertices.append(conv1_vertex)
        
        stage1_vertex = Vertex(name="stage1", shape=(16, img_size, img_size), fixed=False)
        vertices.append(stage1_vertex)
        
        stage2_vertex = Vertex(name="stage2", shape=(32, img_size // 2, img_size // 2), fixed=False)
        vertices.append(stage2_vertex)
        
        stage3_vertex = Vertex(name="stage3", shape=(64, img_size // 4, img_size // 4), fixed=False)
        vertices.append(stage3_vertex)
        
        # Output vertex (fixed during training as target labels)
        output_vertex = Vertex(name="output", shape=(num_classes,), fixed=True)
        vertices.append(output_vertex)
        
        edges = []
        
        initial_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
            total_layers=total_layers,
            key=keys[0],
        )
        
        edge = Edge(
            from_v=vertices[0],  # input
            to_v=vertices[1],    # conv1
            forward_fn=initial_conv,
            energy_ratio=1.0,
        )
        edges.append(edge)
        
        stage1 = ResidualStage(
            num_blocks=9,
            in_channels=16,
            out_channels=16,
            stride=1,
            total_layers=total_layers,
            key=keys[1],
        )
        
        edge = Edge(
            from_v=vertices[1],  # conv1
            to_v=vertices[2],    # stage1
            forward_fn=stage1,
            energy_ratio=1.0,
        )
        edges.append(edge)
        
        stage2 = ResidualStage(
            num_blocks=9,
            in_channels=16,
            out_channels=32,
            stride=2,
            total_layers=total_layers,
            key=keys[2],
        )
        
        edge = Edge(
            from_v=vertices[2],  # stage1
            to_v=vertices[3],    # stage2
            forward_fn=stage2,
            energy_ratio=1.0,
        )
        edges.append(edge)
        
        stage3 = ResidualStage(
            num_blocks=9,
            in_channels=32,
            out_channels=64,
            stride=2,
            total_layers=total_layers,
            key=keys[3],
        )
        
        edge = Edge(
            from_v=vertices[3],  # stage2
            to_v=vertices[4],    # stage3
            forward_fn=stage3,
            energy_ratio=1.0,
        )
        edges.append(edge)
        
        classifier = GlobalAvgPoolAndFC(
            in_channels=64,
            num_classes=num_classes,
            total_layers=total_layers,
            key=keys[4],
        )
        
        edge = Edge(
            from_v=vertices[4],  # stage3
            to_v=vertices[5],    # output
            forward_fn=classifier,
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
        train_opt = optax.adam(train_lr)
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
                
                epoch_energy += energy
                
                if verbose and (batch_idx + 1) % print_interval == 0:
                    avg_energy_so_far = epoch_energy / (batch_idx + 1)
                    print(f'  Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{num_batches}, Energy: {avg_energy_so_far:.6f}')
            
            avg_energy = epoch_energy / num_batches
            energy_history.append(float(avg_energy))
            
            if verbose:
                print(f'Epoch {epoch + 1}/{epochs} completed, Average Energy: {avg_energy:.6f}')
        
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


