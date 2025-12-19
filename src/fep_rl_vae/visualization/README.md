# Visualization Module

Comprehensive visualization capabilities for all FEP-RL-VAE data formats.

## Features

- **Image Visualizations**: Grids, sequences, comparisons, distributions
- **Number Visualizations**: Distributions, predictions, confusion matrices, sequences
- **Description Visualizations**: Embedding spaces, similarity matrices, vector components
- **Training Visualizations**: Loss curves, rewards, entropies, learning curves
- **Model Visualizations**: Feature maps, latent spaces, architectures

## Usage

```python
from fep_rl_vae.visualization import (
    plot_image_grid,
    plot_number_distribution,
    plot_training_history,
    save_training_visualizations,
)

# Image visualization
images = torch.randn(9, 28, 28, 1)
plot_image_grid(images, title="Sample Images", save=True)

# Number visualization
numbers = torch.randint(0, 10, (100,))
plot_number_distribution(numbers, save=True)

# Training visualization
epoch_history = {...}
save_training_visualizations(epoch_history)
```

## Output Organization

All visualizations are saved to the `output/` directory:

```
output/
├── images/          # Image visualizations
├── numbers/         # Number/categorical visualizations
├── descriptions/    # Description/embedding visualizations
├── training/        # Training history visualizations
├── models/         # Model-specific visualizations
└── comparisons/    # Comparison visualizations
```

## Quick Reference

### Image Functions
- `plot_image_grid()` - Grid of images
- `plot_image_sequence()` - Horizontal sequence
- `plot_image_comparison()` - Before/after comparison
- `plot_image_distribution()` - Pixel value distribution

### Number Functions
- `plot_number_distribution()` - Class distribution
- `plot_prediction_probabilities()` - Prediction bars
- `plot_confusion_matrix()` - Classification matrix
- `plot_sequence_predictions()` - Sequence predictions

### Description Functions
- `plot_embedding_space()` - 2D/3D embedding visualization
- `plot_similarity_matrix()` - Cosine similarity heatmap
- `plot_vector_components()` - Component analysis

### Training Functions
- `plot_loss_curves()` - Accuracy/complexity losses
- `plot_reward_curves()` - Reward tracking
- `plot_entropy_curves()` - Entropy metrics
- `plot_learning_curves()` - Actor/critic losses
- `save_training_visualizations()` - Save all training plots

### Model Functions
- `plot_feature_maps()` - Convolutional feature maps
- `plot_latent_space()` - Latent space visualization
- `plot_model_architecture()` - Model structure
