from typing import Optional

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

import math
from pathlib import Path

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

        self.state = None

    def init_state(
        self, 
        key: Optional[jax.Array] = jr.PRNGKey(0), 
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
        else:
            shape = (batch_size, *self.shape)
            self.state = state if state is not None else init_fun(key, shape)


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

class ChainNetwork():
    def __init__(
        self,
        edges: list[Edge]
    ) -> None:
        self.edges = edges
        self.vertices: dict[str, Vertex] = {}
        
        for edge in self.edges:
            self.vertices[edge.from_v.name] = edge.from_v
            self.vertices[edge.to_v.name] = edge.to_v
    
        self._build_chain()
    
    def _build_chain(self):
        """Build the chain structure from edges."""
        out_degree = {name: 0 for name in self.vertices.keys()}
        in_degree = {name: 0 for name in self.vertices.keys()}
        self.edge_from_vertex = {}
        
        for edge in self.edges:
            from_name = edge.from_v.name
            to_name = edge.to_v.name
            out_degree[from_name] += 1
            in_degree[to_name] += 1
            self.edge_from_vertex[from_name] = edge
        
        for name in self.vertices.keys():
            if out_degree[name] > 1 or in_degree[name] > 1:
                raise ValueError(f"Network is not a chain: vertex {name} has "
                               f"{in_degree[name]} incoming and {out_degree[name]} outgoing edges")
    
        start_vertices = [name for name, deg in in_degree.items() if deg == 0]
        if len(start_vertices) == 0:
            raise ValueError("Chain network has no starting vertex (all vertices have incoming edges)")
        if len(start_vertices) > 1:
            raise ValueError(f"Chain network has multiple starting vertices: {start_vertices}")
        
        self.input_vertex_name = start_vertices[0]
        self.non_input_element_count = 0
        for name, vertex in self.vertices.items():
            if name != self.input_vertex_name:
                element_count = math.prod(vertex.shape) if vertex.shape else 0
                self.non_input_element_count += element_count
        
    def _compute_total_energy(self, weights, states_dict):
        total_energy = 0.0
        for edge in self.edges:
            f_s = states_dict[edge.from_v.name]
            t_s = states_dict[edge.to_v.name]
            forward_fn = weights[self.edges.index(edge)]
            pred = forward_fn(f_s)
            energy = edge.energy_ratio * jnp.sum((pred - t_s) ** 2)
            total_energy += energy
        return total_energy

    def inference(
        self,
        weights,
        states_dict: dict[str, jnp.ndarray],
        inf_lr: float,
        inf_epoch: int,
        noised: bool,
        noised_key: jax.Array,
    ):
        inf_opt = optax.adamw(inf_lr)
        trainable_states = {k: v for k, v in states_dict.items() if not self.vertices[k].fixed}
        fixed_states = {k: v for k, v in states_dict.items() if self.vertices[k].fixed}
        
        opt_state = inf_opt.init(trainable_states)

        def scan_fn(carry, k):
            curr_states, opt_state = carry
            
            def energy_for_grad(t_states):
                full_states = {**t_states, **fixed_states}
                return self._compute_total_energy(weights, full_states)

            energy, grads = jax.value_and_grad(energy_for_grad)(curr_states)
            updates, next_opt_state = inf_opt.update(grads, opt_state, curr_states)
            next_states = eqx.apply_updates(curr_states, updates)

            if noised:
                subkeys = jr.split(k, len(next_states))
                noise_scale = jnp.sqrt(2 * inf_lr)
                next_states = {
                    name: next_states[name] + jr.normal(subkeys[i], s.shape) * noise_scale
                    for i, (name, s) in enumerate(next_states.items())
                }
            
            return (next_states, next_opt_state), energy

        keys = jr.split(noised_key, inf_epoch)
        (final_trainable_states, _), energy_history = jax.lax.scan(
            scan_fn, (trainable_states, opt_state), keys
        )
        
        return {**final_trainable_states, **fixed_states}, energy_history[-1]

    @eqx.filter_jit
    def _full_train_step(
        self, 
        weights, 
        states_dict, 
        train_opt_state, 
        train_opt, 
        inf_params,
        key
    ):
        inf_key, _ = jr.split(key)
        updated_states, _ = self.inference(
            weights, states_dict, **inf_params, noised_key=inf_key
        )

        def weight_loss_fn(w):
            return self._compute_total_energy(w, jax.lax.stop_gradient(updated_states))

        energy, weight_grads = eqx.filter_value_and_grad(weight_loss_fn)(weights)
        
        updates, next_train_opt_state = train_opt.update(weight_grads, train_opt_state, weights)
        new_weights = eqx.apply_updates(weights, updates)
        
        return new_weights, next_train_opt_state, energy, updated_states

    def train_step(
        self,
        input_states: dict[str, jnp.array],
        key: jax.Array, 
        returned_vertices: list[str] = None,
        init_fun: callable = jr.normal,
        train_opt = None,
        train_opt_state = None,
        inf_lr: float = 1e-2,
        inf_epoch: int = 100,
    ):
        batch_size = next(iter(input_states.values())).shape[0]
        initial_states = {}
        for name, vertex in self.vertices.items():
            if name in input_states:
                initial_states[name] = input_states[name]
            else:
                key, subkey = jr.split(key)
                initial_states[name] = init_fun(subkey, (batch_size, *vertex.shape))

        weights = [edge.forward_fn for edge in self.edges]
        inf_params = {
            "inf_lr": inf_lr,
            "inf_epoch": inf_epoch,
            "noised": False,
        }

        new_weights, new_opt_state, energy, final_states = self._full_train_step(
            weights, initial_states, train_opt_state, train_opt, inf_params, key
        )

        for i, edge in enumerate(self.edges):
            edge.forward_fn = new_weights[i]
        for name, state in final_states.items():
            self.vertices[name].state = state

        returned_states = {n: final_states[n] for n in returned_vertices} if returned_vertices else None
        average_energy = energy / (batch_size * self.non_input_element_count)
        return new_opt_state, average_energy, returned_states

    def forward(
        self,
        input_states: dict[str, jnp.array],
        returned_vertices: list[str] = None,
        generative: bool = False,
        # We reuse the forward function to generate: 
        #   Duplicate a new vertex as input,
        #   let input_states to be all zeros,
        #   then the bias become the mean of prior Gaussion distribution
        key: Optional[jax.Array] = None,
    ):
        computed_states = {}
        computed_states.update(input_states)
        
        current_name = self.input_vertex_name
        current_state = input_states[current_name]
        first_forward = True

        if generative and key is None:
            key = jr.PRNGKey(0)

        while current_name in self.edge_from_vertex:
            edge = self.edge_from_vertex[current_name]
            current_state = edge.forward_fn(current_state)
            if first_forward and generative:
                current_state = current_state + jr.normal(key, current_state.shape) / jnp.sqrt(current_state.shape[0])

            current_name = edge.to_v.name
            computed_states[current_name] = current_state
            first_forward = False
        
        return {name: computed_states[name] for name in returned_vertices if name in computed_states}
    
    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        weights = [edge.forward_fn for edge in self.edges]

        with open(path, 'wb') as f:
            eqx.tree_serialise_leaves(f, weights)
    
    def load(self, path: str):
        path = Path(path)
        weights_template = [edge.forward_fn for edge in self.edges]

        with open(path, 'rb') as f:
            loaded_weights = eqx.tree_deserialise_leaves(f, weights_template)
        for i, edge in enumerate(self.edges):
            edge.forward_fn = loaded_weights[i]
        
class Relation():
    def __init__(self) -> None:
        # TODO: implement Relation class to represent several-to-one edge.
        pass

class GraphNetwork():
    def __init__(self) -> None:
        # TODO: implement GraphNetwork class to represent a graph PCN.
        pass