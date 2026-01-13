"""
Predictive Coding MLP implementation using the PC network framework.
"""
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from typing import Optional, List, Tuple
import numpy as np

from .network import Vertex, Edge, ChainNetwork
from .blocks import AffineActivation, LinearBlock

class PCMLP:
    def __init__(
        self,
        input_dim: int = 784,  # MNIST: 28x28 = 784
        hidden_dims: List[int] = [512, 256, 128],
        output_dim: int = 10,  # MNIST: 10 classes
        activation: callable = jax.nn.relu,
        key: jax.Array = jr.PRNGKey(42),
    ):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation = activation
        
        vertices = []
        dims = [input_dim] + hidden_dims + [output_dim]
        
        self.vertex_names = []
        for i, dim in enumerate(dims):
            if i == 0:
                name = "input"
                fixed = True  # Input is fixed during training
            elif i == len(dims) - 1:
                name = "output"
                fixed = True  # Output (labels) is fixed during training
            else:
                name = f"hidden_{i}"
                fixed = False  # Hidden layers are trainable
            self.vertex_names.append(name)
            vertex = Vertex(name=name, shape=(dim,), fixed=fixed)
            vertices.append(vertex)
        
        edges = []
        keys = jr.split(key, len(dims) - 1)
        
        for i in range(len(dims) - 1):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            
            if i < len(dims) - 2:
                forward_fn = AffineActivation(
                    in_features=in_dim,
                    out_features=out_dim,
                    activation=activation,
                    key=keys[i],
                    use_bias=True,
                )
            else:
                forward_fn = AffineActivation(
                    in_features=in_dim,
                    out_features=out_dim,
                    activation=jax.nn.softmax,
                    key=keys[i],
                    use_bias=True,
                )
            
            edge = Edge(
                from_v=vertices[i],
                to_v=vertices[i + 1],
                forward_fn=forward_fn,
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
        input_data: jnp.ndarray
    ) -> jnp.ndarray:
        input_states = {"input": input_data}
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
            inputs = batch["input"]
            targets = batch["output"]
            
            # Get predictions using forward pass (no inference)
            input_states = {"input": inputs}
            output_states = self.network.forward(
                input_states=input_states,
                returned_vertices=["output"]
            )
            predictions = output_states["output"]
            
            # Calculate accuracy
            pred_labels = jnp.argmax(predictions, axis=1)
            true_labels = jnp.argmax(targets, axis=1)
            batch_correct = jnp.sum(pred_labels == true_labels)
            correct += batch_correct
            total += inputs.shape[0]
            
            if verbose and (batch_idx + 1) % 50 == 0:
                current_acc = float(correct) / total
                print(f"  Batch {batch_idx + 1}/{len(test_data)}: Current accuracy = {current_acc * 100:.2f}%")
        
        accuracy = float(correct) / total
        
        if verbose:
            print(f"  Final accuracy: {accuracy * 100:.2f}% ({correct}/{total} correct)")
        
        return accuracy
