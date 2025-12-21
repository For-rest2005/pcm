import unittest

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from PCModel.network import Edge, Vertex, Network

class SimpleLinear(eqx.Module):
    weight: jnp.ndarray

    def __call__(self, x):
        # x has shape (batch, in_features)
        # weight has shape (out_features, in_features)
        # output has shape (batch, out_features)
        return jnp.dot(x, self.weight.T)


class EdgeGradTest(unittest.TestCase):
    def test_update_grad_computes_state_and_weight_grads(self):
        # set up vertices and states
        from_v = Vertex(name="from", shape=(2,), fixed=False)
        to_v = Vertex(name="to", shape=(2,), fixed=False)

        # states include batch dimension to match init_state convention
        batch_size = 1
        from_v.init_state(state = jnp.array([[1.0, 0.5]]), batch_size=batch_size)
        to_v.init_state(state = jnp.array([[2.0, 1.0]]), batch_size=batch_size)

        # simple linear forward function: y = x @ w.T where w is (2, 2)
        # Using identity-like weights for simplicity
        forward = SimpleLinear(weight=jnp.array([[2.0, 1.0], [1.0, 2.0]]))
        edge = Edge(from_v=from_v, to_v=to_v, forward_fn=forward, energy_ratio=1.0)

        edge.update_grad(update_weight_grad=True, update_state_grad=True)

        # Just check that gradients are computed (shapes match)
        self.assertEqual(from_v.grad.shape, (1, 2))
        self.assertEqual(to_v.grad.shape, (1, 2))
        self.assertEqual(edge.grad.weight.shape, (2, 2))


class NetworkTest(unittest.TestCase):
    def setUp(self):
        """Create a simple two-layer network for testing."""
        # Create vertices: input -> hidden -> output
        self.input_v = Vertex(name="input", shape=(2,), fixed=False)
        self.hidden_v = Vertex(name="hidden", shape=(3,), fixed=False)
        self.output_v = Vertex(name="output", shape=(2,), fixed=False)
        
        # Create simple linear layers
        key = jr.PRNGKey(42)
        key1, key2 = jr.split(key)
        
        # Simple linear modules
        self.layer1 = SimpleLinear(weight=jr.normal(key1, (3, 2)))
        self.layer2 = SimpleLinear(weight=jr.normal(key2, (2, 3)))
        
        # Create edges
        self.edge1 = Edge(self.input_v, self.hidden_v, self.layer1)
        self.edge2 = Edge(self.hidden_v, self.output_v, self.layer2)
        
        # Create network
        self.network = Network(edges=[self.edge1, self.edge2])
    
    def test_network_initialization(self):
        """Test that network properly initializes vertices."""
        self.assertEqual(len(self.network.vertices), 3)
        self.assertIn("input", self.network.vertices)
        self.assertIn("hidden", self.network.vertices)
        self.assertIn("output", self.network.vertices)
        self.assertEqual(len(self.network.edges), 2)
    
    def test_compute_energy(self):
        """Test energy computation."""
        # Initialize states
        batch_size = 4
        key = jr.PRNGKey(0)
        self.input_v.init_state(key=key, batch_size=batch_size)
        self.hidden_v.init_state(key=key, batch_size=batch_size)
        self.output_v.init_state(key=key, batch_size=batch_size)
        
        # Compute energy
        energy, states_grads, weights_grads = self.network.compute_energy(
            update_state_grad=True,
            update_weight_grad=True
        )
        
        # Check that energy is a scalar
        self.assertTrue(jnp.isscalar(energy) or energy.shape == ())
        # Check that we got gradients for states
        self.assertGreater(len(states_grads), 0)
        # Check that we got gradients for weights
        self.assertEqual(len(weights_grads), 2)
    
    def test_train_step(self):
        """Test single training step."""
        batch_size = 4
        key = jr.PRNGKey(123)
        
        # Create input data
        input_data = {
            "input": jr.normal(key, (batch_size, 2)),
            "output": jr.normal(key, (batch_size, 2)),
        }
        
        # Create optimizer
        import optax
        train_opt = optax.adam(1e-2)
        weights = [edge.forward_fn for edge in self.network.edges]
        train_opt_state = train_opt.init(eqx.filter(weights, eqx.is_array))
        
        # Run train step
        new_opt_state, energy, returned_states = self.network.train_step(
            input_states=input_data,
            key=key,
            train_opt=train_opt,
            train_opt_state=train_opt_state,
            inf_epoch=5,  # Small number for testing
            verbose=False,
        )
        
        # Check that energy is computed
        self.assertTrue(jnp.isscalar(energy) or energy.shape == ())
        self.assertIsNotNone(new_opt_state)
    
    def test_train_with_returned_vertices(self):
        """Test training step with returned vertices."""
        batch_size = 4
        key = jr.PRNGKey(456)
        
        input_data = {
            "input": jr.normal(key, (batch_size, 2)),
            "output": jr.normal(key, (batch_size, 2)),
        }
        
        import optax
        train_opt = optax.adam(1e-2)
        weights = [edge.forward_fn for edge in self.network.edges]
        train_opt_state = train_opt.init(eqx.filter(weights, eqx.is_array))
        
        # Run with returned vertices
        new_opt_state, energy, returned_states = self.network.train_step(
            input_states=input_data,
            key=key,
            returned_vertices=["hidden"],
            train_opt=train_opt,
            train_opt_state=train_opt_state,
            inf_epoch=5,
            verbose=False,
        )
        
        # Check returned states
        self.assertIsNotNone(returned_states)
        self.assertIn("hidden", returned_states)
        self.assertEqual(returned_states["hidden"].shape, (batch_size, 3))
    
    def test_train_full(self):
        """Test full training loop."""
        batch_size = 4
        n_batches = 3
        key = jr.PRNGKey(789)
        
        # Create training data
        train_data = []
        for i in range(n_batches):
            key, subkey = jr.split(key)
            train_data.append({
                "input": jr.normal(subkey, (batch_size, 2)),
                "output": jr.normal(subkey, (batch_size, 2)),
            })
        
        # Train
        key, train_key = jr.split(key)
        energy_history, final_states = self.network.train(
            train_data=train_data,
            key=train_key,
            epochs=3,
            train_lr=1e-2,
            inf_epoch=5,
            verbose=False,
        )
        
        # Check energy history
        self.assertEqual(len(energy_history), 3)
        # Energy should be decreasing (or at least exist)
        for energy in energy_history:
            self.assertIsInstance(energy, float)
    
    def test_train_with_verbose(self):
        """Test training with verbose output."""
        batch_size = 2
        key = jr.PRNGKey(999)
        
        train_data = [{
            "input": jr.normal(key, (batch_size, 2)),
            "output": jr.normal(key, (batch_size, 2)),
        }]
        
        # This should print progress
        energy_history, _ = self.network.train(
            train_data=train_data,
            key=key,
            epochs=2,
            inf_epoch=3,
            verbose=True,
            print_every=1,
        )
        
        self.assertEqual(len(energy_history), 2)


if __name__ == "__main__":
    unittest.main()

