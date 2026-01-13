import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from typing import Optional

class AffineActivation(eqx.Module):
    linear: eqx.nn.Linear
    activation: callable
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        activation: callable = jax.nn.relu,
        key: jax.Array = None,
        use_bias: bool = True,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)
        
        self.linear = eqx.nn.Linear(
            in_features, 
            out_features, 
            use_bias=use_bias,
            key=key
        )
        self.activation = activation
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim == 1:
            return self.activation(self.linear(x))
        else:
            return jax.vmap(lambda xi: self.activation(self.linear(xi)))(x)


class Identity(eqx.Module):
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return x


class LinearBlock(eqx.Module):
    linear: eqx.nn.Linear
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: jax.Array = None,
        use_bias: bool = True,
    ):
        if key is None:
            key = jr.PRNGKey(0)
        
        self.linear = eqx.nn.Linear(
            in_features,
            out_features,
            use_bias=use_bias,
            key=key
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim == 1:
            return self.linear(x)
        else:
            return jax.vmap(self.linear)(x)


class MultiHeadAttention(eqx.Module):
    num_heads: int
    head_dim: int
    embed_dim: int
    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        key: Optional[jax.Array] = None,
    ):
        if key is None:
            key = jr.PRNGKey(0)
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim
        
        keys = jr.split(key, 4)
        self.q_proj = eqx.nn.Linear(embed_dim, embed_dim, use_bias=True, key=keys[0])
        self.k_proj = eqx.nn.Linear(embed_dim, embed_dim, use_bias=True, key=keys[1])
        self.v_proj = eqx.nn.Linear(embed_dim, embed_dim, use_bias=True, key=keys[2])
        self.out_proj = eqx.nn.Linear(embed_dim, embed_dim, use_bias=True, key=keys[3])
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size, num_patches, _ = x.shape
        
        Q = jax.vmap(jax.vmap(self.q_proj))(x)
        K = jax.vmap(jax.vmap(self.k_proj))(x)
        V = jax.vmap(jax.vmap(self.v_proj))(x)

        Q = Q.reshape(batch_size, num_patches, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, num_patches, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, num_patches, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = jnp.sqrt(self.head_dim)
        attn_scores = jnp.einsum('bhqd,bhkd->bhqk', Q, K) / scale
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        attn_output = jnp.einsum('bhqk,bhvd->bhqd', attn_weights, V)
        
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, num_patches, self.embed_dim)
        
        output = jax.vmap(jax.vmap(self.out_proj))(attn_output)
        
        return output

class TransformerBlock(eqx.Module):
    attention: MultiHeadAttention
    mlp_linear1: eqx.nn.Linear
    mlp_linear2: eqx.nn.Linear
    gamma1: jnp.ndarray 
    gamma2: jnp.ndarray
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        total_layers: int = 1,
        key: Optional[jax.Array] = None,
    ):
        if key is None:
            key = jr.PRNGKey(0)
        
        keys = jr.split(key, 3)
        
        self.attention = MultiHeadAttention(embed_dim, num_heads, key=keys[0])
        
        mlp_dim = embed_dim * mlp_ratio
        self.mlp_linear1 = eqx.nn.Linear(embed_dim, mlp_dim, use_bias=True, key=keys[1])
        self.mlp_linear2 = eqx.nn.Linear(mlp_dim, embed_dim, use_bias=True, key=keys[2])
        
        init_value = 0.0
        self.gamma1 = jnp.full((embed_dim,), init_value)
        self.gamma2 = jnp.full((embed_dim,), init_value)
        
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        attn_out = self.attention(x)    
        x = x + self.gamma1 * attn_out
        mlp_out = jax.vmap(jax.vmap(self.mlp_linear1))(x)
        mlp_out = jax.nn.gelu(mlp_out)
        mlp_out = jax.vmap(jax.vmap(self.mlp_linear2))(mlp_out)
        
        x = x + self.gamma2 * mlp_out
        
        return x

class PatchEmbedding(eqx.Module):
    patch_size: int
    embed_dim: int
    projection: eqx.nn.Conv2d
    
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 384,
        key: Optional[jax.Array] = None,
    ):
        if key is None:
            key = jr.PRNGKey(0)
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        self.projection = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            use_bias=True,
            key=key,
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        patches = jax.vmap(self.projection)(x)
        
        batch_size = patches.shape[0]
        embed_dim = patches.shape[1]
        
        patches = patches.reshape(batch_size, embed_dim, -1).transpose(0, 2, 1)
        
        return patches


class InputTransformerBlock(eqx.Module):
    transformer: TransformerBlock
    cls_token: jnp.ndarray
    pos_embed: jnp.ndarray
    
    def __init__(
        self,
        num_patches: int,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        total_layers: int = 1,
        key: Optional[jax.Array] = None,
    ):
        if key is None:
            key = jr.PRNGKey(0)
        
        keys = jr.split(key, 3)
        
        self.transformer = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            total_layers=total_layers,
            key=keys[0],
        )
        
        self.cls_token = jr.normal(keys[1], (1, 1, embed_dim)) * 0.02
        self.pos_embed = jr.normal(keys[2], (1, num_patches + 1, embed_dim)) * 0.1
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        batch_size = x.shape[0]
        
        cls_tokens = jnp.tile(self.cls_token, (batch_size, 1, 1))
        x_with_cls = jnp.concatenate([cls_tokens, x], axis=1)
        
        x_with_pos = x_with_cls + self.pos_embed
        
        output = self.transformer(x_with_pos)
        return output


class OutputTransformerBlock(eqx.Module):
    """
    Output transformer block that combines transformer processing with classification.
    This processes all tokens through a transformer layer, then extracts the CLS token
    for classification, avoiding unnecessary fixing of non-CLS tokens.
    """
    transformer: TransformerBlock
    norm: eqx.nn.LayerNorm
    classifier: eqx.nn.Linear
    
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        total_layers: int = 1,
        key: Optional[jax.Array] = None,
    ):
        """
        Args:
            embed_dim: Embedding dimension
            num_classes: Number of output classes
            num_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            total_layers: Total number of layers (for energy scaling)
            key: Random key for initialization
        """
        if key is None:
            key = jr.PRNGKey(0)
        
        keys = jr.split(key, 2)
        
        # Transformer block to process all tokens
        self.transformer = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            total_layers=total_layers,
            key=keys[0],
        )
        
        # Classification head: norm + linear
        self.norm = eqx.nn.LayerNorm(embed_dim)
        self.classifier = eqx.nn.Linear(embed_dim, num_classes, use_bias=True, key=keys[1])
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Forward pass: transformer processing + classification.
        
        Args:
            x: Input tensor of shape (batch_size, num_patches+1, embed_dim)
        
        Returns:
            Class probabilities of shape (batch_size, num_classes)
        """
        # Process all tokens through transformer
        x = self.transformer(x)
        
        # Extract CLS token (first token)
        cls_token = x[:, 0, :]  # (batch_size, embed_dim)
        
        # Apply normalization and classification
        cls_normed = jax.vmap(self.norm)(cls_token)
        logits = jax.vmap(self.classifier)(cls_normed)
        
        # Apply softmax to get probabilities
        probs = jax.nn.softmax(logits, axis=-1)
        
        return probs

class ConvBlock(eqx.Module):
    conv: eqx.nn.Conv2d
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
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
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        key: Optional[jax.Array] = None,
    ):
        if key is None:
            key = jr.PRNGKey(0)
        
        keys = jr.split(key, 3)
        self.stride = stride
        
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


class ConvTransposeBlock(eqx.Module):
    conv_transpose: eqx.nn.ConvTranspose2d
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1,
        key: Optional[jax.Array] = None,
    ):
        if key is None:
            key = jr.PRNGKey(0)
        
        self.conv_transpose = eqx.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_bias=False,
            key=key,
        )
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        out = jax.vmap(self.conv_transpose)(x)
        out = jax.nn.relu(out)
        return out


class ResidualUpBlock(eqx.Module):
    upsample: eqx.nn.ConvTranspose2d
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    shortcut: eqx.nn.ConvTranspose2d
    gamma: jnp.ndarray
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        key: Optional[jax.Array] = None,
    ):
        if key is None:
            key = jr.PRNGKey(0)
        
        keys = jr.split(key, 4)
        
        # Upsample via transposed convolution
        self.upsample = eqx.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            use_bias=False,
            key=keys[0],
        )
        
        # Two convolutions
        self.conv1 = eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            key=keys[1],
        )
        
        self.conv2 = eqx.nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_bias=False,
            key=keys[2],
        )
        
        # Shortcut for upsampling
        self.shortcut = eqx.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            use_bias=False,
            key=keys[3],
        )
        
        # Learnable residual scaling
        init_value = 0.0
        self.gamma = jnp.full((out_channels, 1, 1), init_value)
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # Upsample input
        out = jax.vmap(self.upsample)(x)
        out = jax.nn.relu(out)
        
        # First conv + ReLU
        out = jax.vmap(self.conv1)(out)
        out = jax.nn.relu(out)
        
        # Second conv
        out = jax.vmap(self.conv2)(out)
        
        # Shortcut with upsampling
        identity = jax.vmap(self.shortcut)(x)
        
        # Residual connection
        out = identity + self.gamma * out
        out = jax.nn.relu(out)
        
        return out

