from math import nan
from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

class Vertex():
    def __init__(
        self,
        name: str,
        shape: tuple = None,
        fixed: bool = False,
        fixed_state: jnp.array = None,
    ) -> None:
        self.name = name
        self.shape = shape
        self.fixed = fixed

        self.grad = None
        self.state = None

    def init_state(
        self, 
        key: Optional[jax.Array] = jr.PRNGKey(0), 
        # NOTICE: I highly recommend users to pass the parameter by hand.
        batch_size: int = 0,
        init_fun: callable = jr.normal,
        state: Optional[jnp.array] = None,
        fixed: Optional[bool] = None,
    ) -> None:
        if fixed is not None:
            self.fixed = fixed

        if self.fixed:
            assert state is not None
            self.state = state
            self.grad = None
        else:
            shape = (batch_size, *self.shape)
            self.state = state if state is not None else init_fun(key, shape)
            self.grad = jnp.zeros(shape)

    def zero_grad(self):
        if self.grad is not None:
            self.grad = jnp.zeros_like(self.state)

class Edge():
    def __init__(
        self, 
        from_v: Vertex, 
        to_v: Vertex, 
        forward_fn: eqx.Module,
        energy_ratio: float = 1.
    ) -> None:
        self.from_v = from_v
        self.to_v = to_v
        self.forward_fn = forward_fn
        self.energy_ratio = energy_ratio
        self.grad = None

    def update_grad(
        self,
        update_weight_grad: bool = False,
        update_state_grad: bool = False
    ) -> float:
        from_state = self.from_v.state
        to_state = self.to_v.state

        def energy_fn(inputs_pack):
            model, f_s, t_s = inputs_pack
            pred = model(f_s)
            return self.energy_ratio * jnp.sum((pred - t_s) ** 2)

        inputs = (self.forward_fn, from_state, to_state)
        energy, grads = eqx.filter_value_and_grad(energy_fn)(inputs)

        weight_grads, from_grad, to_grad = grads

        if update_weight_grad:
            self.grad = weight_grads
        
        if update_state_grad:
            if not self.from_v.fixed:
                self.from_v.grad = self.from_v.grad + from_grad
            
            if not self.to_v.fixed:
                self.to_v.grad = self.to_v.grad + to_grad

        return energy

class Network():
    def __init__(
        self,
        edges: list[Edge]
    ) -> None:
        self.edges = edges
        self.vertices: dict[str, Vertex] = {}
        
        for edge in self.edges:
            self.vertices[edge.from_v.name] = edge.from_v
            self.vertices[edge.to_v.name] = edge.to_v
        
    def compute_energy(
        self, 
        update_state_grad: bool = False,
        update_weight_grad: bool = False,
    ):
        states_grads = {}
        weights_grads = {}
        energy = 0.0
        for edge in self.edges:
            energy += edge.update_grad(
                update_state_grad=update_state_grad, 
                update_weight_grad=update_weight_grad
            )
        if update_state_grad:
            for name, vertex in self.vertices.items():
                if not vertex.fixed:
                    states_grads[name] = vertex.grad
                vertex.zero_grad()
        
        if update_weight_grad:
            for edge in self.edges:
                weights_grads[edge] = edge.grad  

        return energy, states_grads, weights_grads

    def inference_step(
        self,
        opt: optax.GradientTransformation,
        opt_state: optax.OptState,
    ):
        states = {}
        for name, vertex in self.vertices.items():
            if not vertex.fixed:
                states[name] = vertex.state
        
        energy, grads, _ = self.compute_energy(update_state_grad=True)
        updates, new_opt_state = opt.update(grads, opt_state, params=states)
        states = eqx.apply_updates(states, updates)
        
        for name, state in states.items():
            self.vertices[name].state = state
        
        return new_opt_state, energy

    def inference(
        self,
        inf_lr: float = 0.01,
        inf_tolerance: float = 1e-2,
        inf_epoch: int = 100,
        verbose: bool = False,
        print_times: int = 10,
    ): 
        states = {}
        for name, vertex in self.vertices.items():
            if not vertex.fixed:
                states[name] = vertex.state
        
        inf_opt = optax.adam(inf_lr)
        opt_state = inf_opt.init(eqx.filter(states, eqx.is_array))

        last_energy = None
        for idx in range(inf_epoch):
            opt_state, energy = self.inference_step(inf_opt, opt_state)
            
            if verbose and idx % (inf_epoch // print_times) == 0:
                print(f'inf {idx} iter, energy {energy}')
            
            if (last_energy is not None) and (abs(last_energy - energy) < inf_tolerance):
                break
            
            last_energy = energy
    
    def train_step(
        self,
        input_states: dict[str, jnp.array],
        key: jax.Array, 
        returned_vertices: list[str] = None,
        init_fun: callable = jr.normal,
        train_opt = None,
        train_opt_state = None,
        inf_lr: float = 1e-2,
        inf_tolerance: float = 1e-2,
        inf_epoch: int = 100,
        verbose: bool = False,
    ):  
        batch_size = next(iter(input_states.values())).shape[0]
        for name, vertex in self.vertices.items():
            if name in input_states:
                vertex.init_state(state=input_states[name], fixed=True)
            else:
                key, subkey = jr.split(key)
                vertex.init_state(key=subkey, batch_size=batch_size, init_fun=init_fun, fixed=False)
        
        self.inference(inf_lr, inf_tolerance, inf_epoch)

        weights = [edge.forward_fn for edge in self.edges]
        
        energy, _, weights_grads = self.compute_energy(update_weight_grad=True)
        grads_list = [weights_grads.get(edge, None) for edge in self.edges]
        updates, new_opt_state = train_opt.update(grads_list, train_opt_state, params=weights)
        weights = eqx.apply_updates(weights, updates)
        
        for idx, edge in enumerate(self.edges):
            edge.forward_fn = weights[idx]
                
        if verbose:
            print(f'Training energy: {energy}')
        
        if returned_vertices is not None:
            returned_states = {name: self.vertices[name].state for name in returned_vertices if name in self.vertices}
        else:
            returned_states = None
        
        return new_opt_state, energy, returned_states

    def train(
        self,
        train_data: list[dict[str, jnp.array]],
        key: jax.Array,
        epochs: int = 10,
        train_lr: float = 1e-3,
        train_opt: Optional[optax.GradientTransformation] = None,
        returned_vertices: list[str] = None,
        init_fun: callable = jr.normal,
        inf_lr: float = 1e-2,
        inf_tolerance: float = 1e-2,
        inf_epoch: int = 100,
        verbose: bool = False,
        print_every: int = 1,
    ):
        """
        Complete training loop over multiple epochs and batches.
        
        Args:
            train_data: List of dicts containing input states for each batch
            key: Random key for initialization
            epochs: Number of training epochs
            train_lr: Learning rate for weight updates
            train_opt: Optional optimizer (defaults to Adam)
            returned_vertices: List of vertex names to return after training
            init_fun: Initialization function for non-fixed vertices
            inf_lr: Learning rate for inference phase
            inf_tolerance: Tolerance for inference convergence
            inf_epoch: Max iterations for inference
            verbose: Whether to print progress
            print_every: Print progress every N epochs
        
        Returns:
            energy_history: List of energies per epoch
            final_states: Optional dict of returned vertex states
        """
        if train_opt is None:
            train_opt = optax.adam(train_lr)
        
        weights = [edge.forward_fn for edge in self.edges]
        train_opt_state = train_opt.init(eqx.filter(weights, eqx.is_array))
        
        energy_history = []
        
        for epoch in range(epochs):
            epoch_energy = 0.0
            
            for batch_idx, input_states in enumerate(train_data):
                key, subkey = jr.split(key)
                
                train_opt_state, energy, _ = self.train_step(
                    input_states=input_states,
                    key=subkey,
                    returned_vertices=None,
                    init_fun=init_fun,
                    train_opt=train_opt,
                    train_opt_state=train_opt_state,
                    inf_lr=inf_lr,
                    inf_tolerance=inf_tolerance,
                    inf_epoch=inf_epoch,
                    verbose=False,
                )
                
                epoch_energy += energy
            
            avg_energy = epoch_energy / len(train_data)
            energy_history.append(float(avg_energy))
            
            if verbose and epoch % print_every == 0:
                print(f'Epoch {epoch}/{epochs}, Average Energy: {avg_energy:.6f}')
        
        final_states = None
        if returned_vertices is not None:
            final_states = {name: self.vertices[name].state for name in returned_vertices if name in self.vertices}
        
        return energy_history, final_states



    def forward(
        self,
        input_states: dict[str, jnp.array]
    ):

        pass
        