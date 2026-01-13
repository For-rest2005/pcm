"""
Dataset loading utilities for pcm.
Provides memory-efficient dataloader implementations for predictive coding models.
"""
import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Iterator, Dict, Any
from abc import ABC, abstractmethod

import torch
import torchvision
import torchvision.transforms as transforms


class DataLoader:
    def __init__(
        self,
        dataset: Any,
        batch_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: Optional[int] = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self._rng = np.random.RandomState(seed)
        
    def __len__(self) -> int:
        n_samples = len(self.dataset)
        if self.drop_last:
            return n_samples // self.batch_size
        else:
            return (n_samples + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[Dict[str, jnp.ndarray]]:
        n_samples = len(self.dataset)
        indices = np.arange(n_samples)
        
        if self.shuffle:
            self._rng.shuffle(indices)
        
        for i in range(0, n_samples, self.batch_size):
            if self.drop_last and i + self.batch_size > n_samples:
                break
                
            end_idx = min(i + self.batch_size, n_samples)
            batch_indices = indices[i:end_idx]
            
            batch_data = [self.dataset[idx] for idx in batch_indices]
            
            batch = self._collate_batch(batch_data)
            yield batch
    
    @abstractmethod
    def _collate_batch(self, batch_data: list) -> Dict[str, jnp.ndarray]:
        raise NotImplementedError("Subclasses must implement _collate_batch")

class MNISTDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Any,
        batch_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = False,
        flatten: bool = True,
        one_hot: bool = True,
        num_classes: int = 10,
        seed: Optional[int] = None,
    ):
        super().__init__(dataset, batch_size, shuffle, drop_last, seed)
        self.flatten = flatten
        self.one_hot = one_hot
        self.num_classes = num_classes
    
    def _collate_batch(self, batch_data: list) -> Dict[str, jnp.ndarray]:
        """Collate MNIST batch data."""
        images_list = []
        labels_list = []
        
        for image, label in batch_data:
            # Convert to numpy
            if torch.is_tensor(image):
                image_np = image.numpy()
            else:
                image_np = np.array(image)
            
            # Flatten if requested
            if self.flatten:
                image_np = image_np.reshape(-1)
            
            images_list.append(image_np)
            labels_list.append(label)
        
        # Stack into batch arrays
        X_batch = jnp.array(np.array(images_list))
        y_batch = np.array(labels_list)
        
        # One-hot encode labels if requested
        if self.one_hot:
            y_batch = jax.nn.one_hot(jnp.array(y_batch), self.num_classes)
        else:
            y_batch = jnp.array(y_batch)
        
        return {
            "input": X_batch,
            "output": y_batch
        }


class ImageNetDataLoader(DataLoader):
    """DataLoader for ImageNet-style datasets (including Tiny-ImageNet)."""
    
    def __init__(
        self,
        dataset: Any,
        batch_size: int = 64,
        shuffle: bool = True,
        drop_last: bool = False,
        num_classes: int = 200,
        one_hot: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__(dataset, batch_size, shuffle, drop_last, seed)
        self.num_classes = num_classes
        self.one_hot = one_hot
    
    def _collate_batch(self, batch_data: list) -> Dict[str, jnp.ndarray]:
        """Collate ImageNet batch data."""
        images_list = []
        labels_list = []
        
        for image, label in batch_data:
            # Convert to numpy
            if torch.is_tensor(image):
                image_np = image.numpy()
            else:
                image_np = np.array(image)
            
            images_list.append(image_np)
            labels_list.append(label)
        
        # Stack into batch arrays
        X_batch = jnp.array(np.array(images_list))
        y_batch = np.array(labels_list)
        
        # One-hot encode labels if requested
        if self.one_hot:
            y_batch = jax.nn.one_hot(jnp.array(y_batch), self.num_classes)
        else:
            y_batch = jnp.array(y_batch)
        
        return {
            "input": X_batch,
            "output": y_batch
        }


def get_mnist_dataloaders(
    data_dir: str = './data',
    batch_size: int = 64,
    flatten: bool = True,
    one_hot: bool = True,
    shuffle_train: bool = True,
    shuffle_test: bool = False,
    seed: Optional[int] = None,
) -> tuple[MNISTDataLoader, MNISTDataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = MNISTDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        flatten=flatten,
        one_hot=one_hot,
        num_classes=10,
        seed=seed,
    )
    
    test_loader = MNISTDataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=shuffle_test,
        flatten=flatten,
        one_hot=one_hot,
        num_classes=10,
        seed=seed,
    )
    
    return train_loader, test_loader


def get_tiny_imagenet_dataloaders(
    data_dir: str = './data/tiny-imagenet-200',
    batch_size: int = 64,
    img_size: int = 64,
    num_train_samples: Optional[int] = None,
    num_val_samples: Optional[int] = None,
    shuffle_train: bool = True,
    shuffle_val: bool = False,
    seed: Optional[int] = None,
) -> tuple[ImageNetDataLoader, ImageNetDataLoader]:
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_dataset = torchvision.datasets.ImageFolder(
        root=f'{data_dir}/train',
        transform=transform
    )
    
    val_dataset = torchvision.datasets.ImageFolder(
        root=f'{data_dir}/val',
        transform=transform
    )
    
    if num_train_samples is not None and num_train_samples < len(train_dataset):
        rng = np.random.RandomState(seed)
        train_indices = np.arange(len(train_dataset))
        rng.shuffle(train_indices)
        train_indices = train_indices[:num_train_samples]
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    
    if num_val_samples is not None and num_val_samples < len(val_dataset):
        rng = np.random.RandomState(seed)
        val_indices = np.arange(len(val_dataset))
        rng.shuffle(val_indices)
        val_indices = val_indices[:num_val_samples]
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    
    train_loader = ImageNetDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_classes=200,
        seed=seed,
    )
    
    val_loader = ImageNetDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=shuffle_val,
        num_classes=200,
        seed=seed,
    )
    
    return train_loader, val_loader
