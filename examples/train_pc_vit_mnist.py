import argparse
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from pcm.vit import PCViT
from pcm.datasets import get_mnist_dataloaders


def extract_attention_weights(model, image):
    images = jnp.expand_dims(image, axis=0)
    
    patch_embed_fn = model.edges[0].forward_fn
    patches = patch_embed_fn(images)
    
    input_transformer = model.edges[1].forward_fn
    batch_size = patches.shape[0]
    cls_tokens = jnp.tile(input_transformer.cls_token, (batch_size, 1, 1))
    x = jnp.concatenate([cls_tokens, patches], axis=1)
    x = x + input_transformer.pos_embed
    
    attention_weights_list = []
    transformer_block = input_transformer.transformer
    attention_module = transformer_block.attention
    norm1 = transformer_block.norm1
    
    # Normalize input before attention
    normed = jax.vmap(jax.vmap(norm1))(x)
    attn_weights = compute_attention_weights(attention_module, normed)
    attention_weights_list.append(attn_weights[0])  # Remove batch dimension
    
    # Apply the transformer
    x = input_transformer.transformer(x)
    
    # Forward through remaining transformer layers
    for i in range(1, model.num_layers):
        transformer = model.edges[i + 1].forward_fn
        attention_module = transformer.attention
        norm1 = transformer.norm1
        
        # Normalize input
        normed = jax.vmap(jax.vmap(norm1))(x)
        
        # Extract attention weights
        attn_weights = compute_attention_weights(attention_module, normed)
        attention_weights_list.append(attn_weights[0])  # Remove batch dimension
        
        # Apply transformer
        x = transformer(x)
    
    return attention_weights_list


def compute_attention_weights(attention_module, x):
    batch_size, num_patches, _ = x.shape
    
    # Project to Q, K, V
    Q = jax.vmap(jax.vmap(attention_module.q_proj))(x)
    K = jax.vmap(jax.vmap(attention_module.k_proj))(x)
    
    # Reshape for multi-head attention
    Q = Q.reshape(batch_size, num_patches, attention_module.num_heads, attention_module.head_dim).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, num_patches, attention_module.num_heads, attention_module.head_dim).transpose(0, 2, 1, 3)
    
    # Compute attention scores and weights
    scale = jnp.sqrt(attention_module.head_dim)
    attn_scores = jnp.einsum('bhqd,bhkd->bhqk', Q, K) / scale
    attn_weights = jax.nn.softmax(attn_scores, axis=-1)
    
    return attn_weights


def visualize_attention_maps(model, image, true_label=None):
    """
    Visualize attention maps for all layers and heads.
    
    Args:
        model: Trained PCViT model
        image: Single image of shape (channels, height, width)
        true_label: Optional true label of the image
    """
    # Extract attention weights
    attention_weights_list = extract_attention_weights(model, image)
    
    num_layers = len(attention_weights_list)
    num_heads = model.num_heads
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_layers, num_heads + 1, figsize=(3 * (num_heads + 1), 3 * num_layers))
    
    if num_layers == 1:
        axes = axes.reshape(1, -1)
    
    # Display original image in first column
    for layer_idx in range(num_layers):
        ax = axes[layer_idx, 0]
        # Convert CHW to HWC for display
        img_display = jnp.transpose(image, (1, 2, 0))
        if img_display.shape[-1] == 1:
            img_display = jnp.squeeze(img_display, axis=-1)
            ax.imshow(img_display, cmap='gray')
        else:
            ax.imshow(img_display)
        
        if layer_idx == 0:
            title = f'Input Image'
            if true_label is not None:
                title += f'\n(Label: {true_label})'
            ax.set_title(title)
        else:
            ax.set_title('')
        ax.axis('off')
    
    # Display attention maps for each layer and head
    for layer_idx in range(num_layers):
        attn_weights = attention_weights_list[layer_idx]  # (num_heads, num_patches+1, num_patches+1)
        
        for head_idx in range(num_heads):
            ax = axes[layer_idx, head_idx + 1]
            
            # Get attention from CLS token to all patches
            cls_attention = attn_weights[head_idx, 0, :]  # (num_patches+1,)
            
            # Exclude CLS token attention (first element)
            patch_attention = cls_attention[1:]  # (num_patches,)
            
            # Reshape to 2D grid
            grid_size = int(jnp.sqrt(len(patch_attention)))
            attention_map = patch_attention.reshape(grid_size, grid_size)
            
            # Display attention map
            im = ax.imshow(attention_map, cmap='viridis', interpolation='nearest')
            ax.set_title(f'L{layer_idx+1}-H{head_idx+1}')
            ax.axis('off')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.suptitle('Attention Maps: CLS Token to Image Patches', fontsize=16, y=0.995)
    plt.tight_layout()
    plt.show()


def main(save_path=None, load_path=None):
    print("=" * 60)
    print("PC-ViT Training - MNIST")
    print("=" * 60)
    print()
    
    config = {
        "img_size": 28,
        "patch_size": 4,
        "in_channels": 1,
        "num_classes": 10,
        "embed_dim": 128,
        "num_layers": 1,
        "num_heads": 4,
        "mlp_ratio": 4,
        
        "batch_size": 64,
        "epochs": 10,
        "train_lr": 1e-2,
        "inf_lr": 1e-2,
        "inf_epoch": 200,
        
        "seed": 42,
        "print_times_every_epoch": 938,
        
        # Overfit test: Use a tiny dataset (16 samples) to test if model can overfit
        "overfit_test": False,  # Set to True to enable overfit testing on tiny dataset
        "overfit_samples": 64,  # Number of samples for overfit test
    }
    
    num_patches = (config["img_size"] // config["patch_size"]) ** 2
    print(f"Image will be split into {num_patches} patches (each {config['patch_size']}x{config['patch_size']})")
    print()
    
    print("Loading MNIST dataset...")
    train_loader, test_loader = get_mnist_dataloaders(
        batch_size=config["batch_size"],
        flatten=False,
        one_hot=True,
        seed = config["seed"]
    )
    
    # Overfit test mode: use only a tiny subset of data
    if config["overfit_test"]:
        print(f"OVERFIT TEST MODE ENABLED")
        print(f"Using only {config['overfit_samples']} samples for training and testing")
        print(f"This is designed to test if the model can overfit on a tiny dataset")
        print()
        
        first_batch = next(iter(train_loader))
        images = first_batch["input"][:config["overfit_samples"]]
        labels = first_batch["output"][:config["overfit_samples"]]
        
        small_dataset = [{"input": images, "output": labels}]
        train_data = small_dataset
        test_data = small_dataset  # Use same data for testing to verify overfitting
        
        print(f"Dataset size: {images.shape[0]} samples")
    else:
        # Normal mode: use the loaders directly
        train_data = train_loader
        test_data = test_loader
    
    print()
    
    print("Initializing PC-ViT model...")
    key = jr.PRNGKey(config["seed"])
    model = PCViT(
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        in_channels=config["in_channels"],
        num_classes=config["num_classes"],
        embed_dim=config["embed_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        mlp_ratio=config["mlp_ratio"],
        key=key,
    )
    
    print(f"Model architecture:")
    print(f"  Image size: {config['img_size']}x{config['img_size']}")
    print(f"  Patches number: {model.num_patches} (each {config['patch_size']}x{config['patch_size']})")
    print(f"  Embedding dimension: {config['embed_dim']}")
    print(f"  Transformer layers: {config['num_layers']}")
    print(f"  Attention heads: {config['num_heads']}")
    print(f"  Output classes: {config['num_classes']}")
    print()
    
    if load_path:
        print(f"Loading pre-trained model from: {load_path}")
        print("-" * 60)
        model.network.load(load_path)
        print("  Model loaded successfully!")
        print()
    
    print("Starting training...")
    if not config["overfit_test"]:
        print(f"Evaluate accuracy every {config['print_times_every_epoch']} batches")
    else:
        print(f"OVERFIT MODE: Training on {config['overfit_samples']} samples only")
    print()
    
    key, subkey = jr.split(key)
    
    train_opt = optax.adamw(config["train_lr"])
    weights = [edge.forward_fn for edge in model.network.edges]
    train_opt_state = train_opt.init(eqx.filter(weights, eqx.is_array))
    
    energy_history = []
    total_batches = 0
    batches_per_epoch = len(train_data)
    print(f"batches_per_epoch: {batches_per_epoch}")
    
    for epoch in range(config["epochs"]):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config['epochs']}")
        print(f"{'='*60}")
        
        epoch_energy = 0.0
        
        for batch_idx, batch in enumerate(train_data):
            key, subkey = jr.split(key)
            
            train_opt_state, energy, _ = model.network.train_step(
                input_states=batch,
                key=subkey,
                returned_vertices=None,
                init_fun=jr.normal,
                train_opt=train_opt,
                train_opt_state=train_opt_state,
                inf_lr=config["inf_lr"],
                inf_epoch=config["inf_epoch"],
            )
            
            energy_history.append(float(energy))
            epoch_energy += energy
            total_batches += 1
            
            if config["overfit_test"] or (batch_idx + 1) % max(1, batches_per_epoch // config["print_times_every_epoch"]) == 0:
                print(f"  Batch {batch_idx + 1}/{batches_per_epoch}, "
                      f"Energy: {energy:.6f}")
            
            eval_frequency = 1 if config["overfit_test"] else 50
            if total_batches % eval_frequency == 0:
                print(f"\n{'─'*60}")
                print(f"Evaluate - {total_batches} batches (Epoch {epoch+1}, Batch {batch_idx+1})")
                print(f"{'─'*60}")
                
                # In overfit mode, use all data; otherwise use first 20 batches
                if config["overfit_test"]:
                    val_subset = test_data
                else:
                    # Get first 20 batches from test loader
                    val_subset = []
                    for i, val_batch in enumerate(test_data):
                        val_subset.append(val_batch)
                        if i >= 19:  # Get 20 batches (0-19)
                            break
                
                correct = 0
                total = 0
                pred_count = [0] * 10
                correct_per_class = [0] * 10
                
                for val_batch in val_subset:
                    images = val_batch["input"]
                    targets = val_batch["output"]
                    
                    predictions = model.predict(images)
                    
                    pred_labels = jnp.argmax(predictions, axis=1)
                    true_labels = jnp.argmax(targets, axis=1)
                    
                    for pred_label in pred_labels:
                        pred_count[int(pred_label)] += 1
                    
                    for pred_label, true_label in zip(pred_labels, true_labels):
                        if int(pred_label) == int(true_label):
                            correct_per_class[int(pred_label)] += 1
                    
                    correct += jnp.sum(pred_labels == true_labels)
                    total += images.shape[0]
                
                accuracy = float(correct) / total
                
                print(f"Validation accuracy: {accuracy * 100:.2f}% ({correct}/{total} correct)")
                print(f"Current energy: {energy:.6f}")
                print(f"\nPrediction distribution:")
                for class_id in range(10):
                    count = pred_count[class_id]
                    correct_count = correct_per_class[class_id]
                    percentage = (count / total) * 100
                    correct_rate = (correct_count / count * 100) if count > 0 else 0
                    print(f"  Number {class_id}: {count:4d} predictions ({percentage:5.2f}%), "
                          f"{correct_count:3d} correct ({correct_rate:5.2f}%)")
                
                print(f"{'─'*60}\n")
        
        avg_epoch_energy = epoch_energy / batches_per_epoch
        print(f"\nEpoch {epoch + 1} completed. Average energy: {avg_epoch_energy:.6f}")
    
    print("\nTraining completed!")
    print()
    
    if save_path:
        print(f"Saving model to: {save_path}")
        print("-" * 60)
        model.network.save(save_path)
        print("  Model saved successfully!")
        print()
    
    print("Evaluating on the training set...")
    train_accuracy = model.evaluate_accuracy(
        test_data=train_data,
        verbose=True,
    )
    print()
    
    print("Evaluating on the test set...")
    test_accuracy = model.evaluate_accuracy(
        test_data=test_data,
        verbose=True,
    )
    print()

    print("=" * 60)
    print("Training summary")
    print("=" * 60)
    print(f"Configuration: {config['num_layers']} layers, {config['embed_dim']} dim, {config['num_heads']} heads")
    if config["overfit_test"]:
        print(f"⚠️  OVERFIT TEST MODE: {config['overfit_samples']} samples")
        print(f"Training samples: {config['overfit_samples']}")
        print(f"Test samples: {config['overfit_samples']} (same as training)")
    else:
        print(f"Training samples: 60000")
        print(f"Test samples: 10000")
    print(f"Training epochs: {config['epochs']}")
    print(f"Final training energy: {energy_history[-1]:.6f}")
    print(f"Training accuracy: {train_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    if config["overfit_test"] and train_accuracy > 0.95:
        print(f"✓ Model successfully overfitted on tiny dataset!")
    print("=" * 60)
    print()
    
    # Visualize attention maps
    print("Generating attention visualization...")
    print("Selecting a random sample from test set for visualization...")
    
    # Get a sample from test data
    if config["overfit_test"]:
        first_batch = test_data[0]
    else:
        first_batch = next(iter(test_data))
    sample_image = first_batch["input"][0]
    sample_label = jnp.argmax(first_batch["output"][0])
    
    print(f"Visualizing attention maps for a test image (label: {sample_label})")
    visualize_attention_maps(model, sample_image, true_label=int(sample_label))
    print("Attention visualization completed!")
    print()
    
    return model, energy_history, train_accuracy, test_accuracy


if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Train PC-ViT on MNIST")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save the trained model (e.g. './checkpoints/vit_mnist.eqx')")
    parser.add_argument("--load_path", type=str, default=None,
                        help="Path to load the pre-trained model (e.g. './checkpoints/vit_mnist.eqx')")
    args = parser.parse_args()
    
    main(save_path=args.save_path, load_path=args.load_path)

