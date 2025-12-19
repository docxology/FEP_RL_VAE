# AGENTS.md - Visualization Module

## Overview

The visualization module provides comprehensive visualization capabilities for all data formats in FEP-RL-VAE. All outputs are automatically saved to the `output/` directory with organized subdirectories.

## Architecture

### Module Structure

```
visualization/
├── __init__.py          # Module exports
├── utils.py             # Shared utilities (paths, styles)
├── images.py            # Image visualizations
├── numbers.py           # Number/categorical visualizations
├── descriptions.py      # Description/embedding visualizations
├── training.py          # Training history visualizations
└── models.py           # Model-specific visualizations
```

### Output Organization

All visualizations are saved to `output/` with the following structure:

- `output/images/` - Image grids, sequences, comparisons
- `output/numbers/` - Distributions, predictions, confusion matrices
- `output/descriptions/` - Embedding spaces, similarity matrices
- `output/training/` - Loss curves, rewards, entropies
- `output/models/` - Feature maps, latent spaces, architectures
- `output/comparisons/` - Before/after comparisons

## Design Patterns

### Consistent Interface

All visualization functions follow a consistent pattern:

```python
def plot_something(
    data,
    title: str = "Default Title",
    show: bool = False,
    save: bool = True,
    filename: Optional[str] = None,
    subdir: str = "category",
) -> plt.Figure:
    """Plot something."""
    setup_plot_style()
    # ... plotting code ...
    if save:
        save_figure(fig, filename or "default.png", subdir)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig
```

### Automatic File Management

- Output directories are created automatically
- Filenames are auto-generated if not provided
- All figures are saved with high DPI (300) by default
- Consistent styling across all visualizations

### Data Format Handling

Functions handle multiple input formats:
- PyTorch tensors (automatically converted to numpy)
- NumPy arrays
- Python lists
- One-hot encoded labels (automatically converted to indices)

## Key Functions

### Image Visualizations (`images.py`)

**plot_image_grid()**: Grid layout for multiple images
- Handles variable numbers of images
- Auto-calculates optimal grid dimensions
- Supports grayscale and RGB images

**plot_image_sequence()**: Horizontal sequence layout
- Useful for temporal sequences
- Optional labels for each image
- Configurable figure size

**plot_image_comparison()**: Side-by-side comparison
- Original vs reconstructed
- Multiple samples in one figure
- Clear visual comparison

**plot_image_distribution()**: Pixel value histogram
- Statistical analysis of images
- Configurable bin count
- Useful for data validation

### Number Visualizations (`numbers.py`)

**plot_number_distribution()**: Class distribution bar chart
- Counts occurrences per class
- Value labels on bars
- Configurable number of classes

**plot_prediction_probabilities()**: Prediction probability bars
- Multiple samples in grid layout
- Highlights true and predicted labels
- Color-coded for easy interpretation

**plot_confusion_matrix()**: Classification confusion matrix
- Normalized or raw counts
- Color-coded heatmap
- Text annotations for values

**plot_sequence_predictions()**: Temporal prediction visualization
- Line plots for sequences
- True vs predicted comparison
- Multiple sequences in one figure

### Description Visualizations (`descriptions.py`)

**plot_embedding_space()**: Dimensionality reduction visualization
- t-SNE or PCA reduction
- 2D or 3D visualization
- Optional label coloring

**plot_similarity_matrix()**: Cosine similarity heatmap
- Shows relationships between embeddings
- Color-coded similarity values
- Configurable sample size

**plot_vector_components()**: Component analysis
- Bar charts for vector components
- Multiple samples comparison
- Useful for understanding embeddings

### Training Visualizations (`training.py`)

**plot_loss_curves()**: Accuracy and complexity losses
- Side-by-side subplots
- Multiple modalities support
- Clear legend and labels

**plot_reward_curves()**: Reward tracking
- Total and component rewards
- Curiosity metrics
- Single comprehensive plot

**plot_entropy_curves()**: Entropy metrics
- Alpha entropies
- Total entropies
- Two-panel layout

**plot_learning_curves()**: Actor and critic losses
- Separate subplots
- Multiple critic support
- Learning progress visualization

**save_training_visualizations()**: Comprehensive training plots
- Generates all training visualizations
- Returns dictionary of saved paths
- One-call convenience function

### Model Visualizations (`models.py`)

**plot_feature_maps()**: Convolutional feature maps
- Grid layout for channels
- Color-coded activation maps
- Configurable number of maps

**plot_latent_space()**: Latent space visualization
- Dimensionality reduction
- Optional label coloring
- 2D projection

**plot_model_architecture()**: Model structure
- Uses torchinfo for detailed summary
- Fallback to string representation
- Text-based visualization

## Utilities (`utils.py`)

**ensure_output_dir()**: Create output directory
- Handles nested directories
- Returns Path object
- Idempotent operation

**get_output_path()**: Generate full file path
- Combines directory and filename
- Ensures directory exists
- Returns Path object

**save_figure()**: Save matplotlib figure
- High DPI (300) by default
- Tight bounding box
- Configurable format

**setup_plot_style()**: Configure matplotlib style
- Consistent styling
- Grid and font settings
- Fallback handling

## Integration Patterns

### With Training Loop

```python
from fep_rl_vae.visualization import save_training_visualizations

epoch_history = {}
for epoch in range(num_epochs):
    # ... training ...
    add_to_epoch_history(epoch_history, epoch_data)
    
    if epoch % 10 == 0:
        save_training_visualizations(epoch_history)
```

### With Data Loading

```python
from fep_rl_vae.visualization import plot_image_grid, plot_number_distribution
from fep_rl_vae.data.loader import MNISTLoader

loader = MNISTLoader()
images, labels = loader.get_batch(16)

plot_image_grid(images, title="Training Batch")
plot_number_distribution(labels.argmax(dim=-1))
```

### With Model Outputs

```python
from fep_rl_vae.visualization import plot_image_comparison, plot_latent_space

# After forward pass
reconstructed, log_prob = decoder(latent)
plot_image_comparison(original, reconstructed)

plot_latent_space(latent, labels=ground_truth_labels)
```

## Performance Considerations

- **Memory Efficiency**: Figures are closed after saving (unless show=True)
- **Batch Processing**: Functions handle batched data efficiently
- **Lazy Loading**: sklearn imports only when needed
- **File Management**: Automatic cleanup and organization

## Dependencies

- **matplotlib**: Core plotting library
- **numpy**: Array operations
- **torch**: Tensor handling
- **sklearn** (optional): Dimensionality reduction for embeddings
- **torchinfo** (optional): Model architecture visualization

## Extension Points

### Adding New Visualizations

1. Create function in appropriate module
2. Follow consistent interface pattern
3. Use `setup_plot_style()` and `save_figure()`
4. Add to `__init__.py` exports
5. Add tests in `test_visualization.py`

### Custom Output Locations

```python
from fep_rl_vae.visualization.utils import DEFAULT_OUTPUT_DIR

# Use custom output directory
DEFAULT_OUTPUT_DIR = Path("custom_output")
```

### Custom Styles

```python
from fep_rl_vae.visualization.utils import setup_plot_style

# Use custom matplotlib style
setup_plot_style("seaborn-v0_8")
```
