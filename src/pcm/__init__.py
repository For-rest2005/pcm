"""
pcm: Predictive Coding Network Framework

A JAX-based implementation of predictive coding networks with support for:
- Multi-Layer Perceptrons (MLP)
- Vision Transformers (ViT)
- ResNets (ResNet56)
"""

from .network import Vertex, Edge, ChainNetwork
from .blocks import (
    AffineActivation,
    Identity,
    LinearBlock,
    MultiHeadAttention,
    TransformerBlock,
    PatchEmbedding,
    InputTransformerBlock,
    OutputTransformerBlock,
    ConvBlock,
    ResidualBlock,
    ConvTransposeBlock,
    ResidualUpBlock,
)   
from .mlp import PCMLP
from .vit import PCViT
from .resnet import PCResNet56


from .datasets import get_mnist_dataloaders, get_tiny_imagenet_dataloaders

__all__ = [
    # Core network components
    "Vertex",
    "Edge",
    "ChainNetwork",
    # Building blocks
    "AffineActivation",
    "Identity",
    "LinearBlock",
    "MultiHeadAttention",
    "TransformerBlock",
    "PatchEmbedding",
    "InputTransformerBlock",
    "OutputTransformerBlock",
    "ConvBlock",
    "ResidualBlock",
    "ConvTransposeBlock",
    "ResidualUpBlock",
    "PCMLP",
    "PCViT",
    "PCResNet56",
    "get_mnist_dataloaders", 
    "get_tiny_imagenet_dataloaders"
]

__version__ = "0.1.0"





