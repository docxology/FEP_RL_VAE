"""Comprehensive demonstration of all visualization capabilities."""

import torch
import numpy as np
from fep_rl_vae.data.loader import MNISTLoader
from fep_rl_vae.visualization import (
    # Image visualizations
    plot_image_grid,
    plot_image_sequence,
    plot_image_comparison,
    plot_image_distribution,
    # Number visualizations
    plot_number_distribution,
    plot_prediction_probabilities,
    plot_confusion_matrix,
    plot_sequence_predictions,
    # Description visualizations
    plot_similarity_matrix,
    plot_vector_components,
    # Training visualizations
    save_training_visualizations,
    # Model visualizations
    plot_feature_maps,
    plot_latent_space,
)


def main():
    """Generate all types of visualizations."""
    print("🎨 Generating comprehensive visualizations...")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading MNIST data...")
    loader = MNISTLoader()
    images, labels = loader.get_batch(batch_size=16)
    print(f"   ✓ Loaded {len(images)} images")
    
    # Image visualizations
    print("\n2. Generating image visualizations...")
    plot_image_grid(images[:9], title="MNIST Sample Images", filename="mnist_grid.png")
    print("   ✓ Image grid saved")
    
    plot_image_sequence(images[:5], title="Image Sequence", filename="mnist_sequence.png")
    print("   ✓ Image sequence saved")
    
    # Create fake reconstructions for comparison
    reconstructed = images[:3] + torch.randn_like(images[:3]) * 0.1
    plot_image_comparison(images[:3], reconstructed, title="Original vs Reconstructed",
                         filename="mnist_comparison.png")
    print("   ✓ Image comparison saved")
    
    plot_image_distribution(images, title="Pixel Value Distribution", filename="mnist_distribution.png")
    print("   ✓ Image distribution saved")
    
    # Number visualizations
    print("\n3. Generating number visualizations...")
    number_labels = labels.argmax(dim=-1)
    plot_number_distribution(number_labels, title="Digit Class Distribution", filename="digit_distribution.png")
    print("   ✓ Number distribution saved")
    
    # Create fake predictions
    predictions = torch.rand(10, 10)
    predictions = torch.softmax(predictions, dim=-1)
    true_labels = torch.randint(0, 10, (10,))
    plot_prediction_probabilities(predictions, true_labels, title="Prediction Probabilities",
                                  filename="prediction_probabilities.png")
    print("   ✓ Prediction probabilities saved")
    
    # Confusion matrix
    pred_labels = predictions.argmax(dim=-1)
    plot_confusion_matrix(pred_labels, true_labels, title="Confusion Matrix", filename="confusion_matrix.png")
    print("   ✓ Confusion matrix saved")
    
    # Sequence predictions
    sequences = torch.randn(5, 10)
    seq_predictions = torch.randint(0, 10, (5, 10))
    seq_labels = torch.randint(0, 10, (5, 10))
    plot_sequence_predictions(sequences, seq_predictions, seq_labels, title="Sequence Predictions",
                             filename="sequence_predictions.png")
    print("   ✓ Sequence predictions saved")
    
    # Description visualizations
    print("\n4. Generating description visualizations...")
    embeddings = torch.randn(50, 64)
    plot_similarity_matrix(embeddings, title="Embedding Similarity Matrix", filename="similarity_matrix.png")
    print("   ✓ Similarity matrix saved")
    
    plot_vector_components(embeddings[:5], title="Vector Components", filename="vector_components.png")
    print("   ✓ Vector components saved")
    
    # Training visualizations
    print("\n5. Generating training visualizations...")
    epoch_history = {
        "accuracy_losses": {
            "image": [0.5, 0.3, 0.2, 0.15, 0.1],
            "number": [0.8, 0.6, 0.4, 0.3, 0.2],
        },
        "complexity_losses": {
            "image": [0.01, 0.008, 0.006, 0.005, 0.004],
        },
        "total_reward": [10.5, 12.3, 15.1, 17.2, 19.5],
        "reward": [10.0, 12.0, 15.0, 17.0, 19.0],
        "curiosities": {
            "obs1": [0.2, 0.3, 0.4, 0.5, 0.6],
        },
        "actor_loss": [0.9, 0.7, 0.5, 0.4, 0.3],
        "critic_losses": [
            [0.8, 0.6, 0.4, 0.3, 0.2],
            [0.7, 0.5, 0.3, 0.2, 0.1],
        ],
        "alpha_entropies": {
            "action1": [0.9, 0.8, 0.7, 0.6, 0.5],
        },
        "alpha_normal_entropies": {
            "action1": [0.1, 0.2, 0.3, 0.4, 0.5],
        },
        "total_entropies": {
            "action1": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
    }
    saved_files = save_training_visualizations(epoch_history, prefix="demo_training")
    print(f"   ✓ Training visualizations saved ({len(saved_files)} files)")
    
    # Model visualizations
    print("\n6. Generating model visualizations...")
    feature_maps = torch.randn(16, 14, 14)  # (channels, height, width)
    plot_feature_maps(feature_maps, title="Feature Maps", filename="feature_maps.png")
    print("   ✓ Feature maps saved")
    
    latent_vectors = torch.randn(50, 32)
    latent_labels = torch.randint(0, 10, (50,))
    plot_latent_space(latent_vectors, latent_labels, title="Latent Space Visualization",
                     filename="latent_space.png")
    print("   ✓ Latent space saved")
    
    print("\n" + "=" * 60)
    print("✅ All visualizations generated successfully!")
    print(f"📁 Check the 'output/' directory for all saved files")
    print("=" * 60)


if __name__ == "__main__":
    main()
