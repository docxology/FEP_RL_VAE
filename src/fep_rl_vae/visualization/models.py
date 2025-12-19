"""Model visualization functions for FEP-RL-VAE."""

from typing import List, Optional, Union, Tuple, Dict
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .utils import ensure_output_dir, get_output_path, save_figure, setup_plot_style


def plot_feature_maps(
    feature_maps: Union[torch.Tensor, np.ndarray],
    title: str = "Feature Maps",
    n_maps: int = 16,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "viridis",
    show: bool = False,
    save: bool = True,
    filename: Optional[str] = None,
    subdir: str = "models",
) -> plt.Figure:
    """Plot feature maps from convolutional layers.
    
    Args:
        feature_maps: Feature maps tensor (B, C, H, W) or (C, H, W)
        title: Title for the plot
        n_maps: Number of feature maps to display
        figsize: Figure size
        cmap: Colormap to use
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Output filename
        subdir: Subdirectory within output folder
        
    Returns:
        Matplotlib figure object
    """
    setup_plot_style()
    
    # Convert to numpy if needed
    if isinstance(feature_maps, torch.Tensor):
        feature_maps = feature_maps.detach().cpu().numpy()
    
    # Handle batch dimension
    if feature_maps.ndim == 4:
        feature_maps = feature_maps[0]  # Take first sample
    
    n_channels = feature_maps.shape[0]
    n_maps = min(n_maps, n_channels)
    
    # Calculate grid dimensions
    n_cols = int(np.ceil(np.sqrt(n_maps)))
    n_rows = int(np.ceil(n_maps / n_cols))
    
    if figsize is None:
        figsize = (n_cols * 2, n_rows * 2)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    
    if n_maps == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(n_maps):
        ax = axes[i]
        feature_map = feature_maps[i]
        im = ax.imshow(feature_map, cmap=cmap)
        ax.set_title(f"Channel {i}", fontsize=9)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # Hide unused subplots
    for i in range(n_maps, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    
    if save:
        filename = filename or f"feature_maps_{n_maps}channels.png"
        save_figure(fig, filename, subdir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_latent_space(
    latent_vectors: Union[torch.Tensor, np.ndarray],
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: str = "Latent Space",
    method: str = "pca",
    show: bool = False,
    save: bool = True,
    filename: Optional[str] = None,
    subdir: str = "models",
) -> plt.Figure:
    """Plot latent space visualization.
    
    Args:
        latent_vectors: Latent vectors (N, dim)
        labels: Optional labels for coloring
        title: Title for the plot
        method: Dimensionality reduction method ('pca' or 'tsne')
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Output filename
        subdir: Subdirectory within output folder
        
    Returns:
        Matplotlib figure object
    """
    setup_plot_style()
    
    # Convert to numpy if needed
    if isinstance(latent_vectors, torch.Tensor):
        latent_vectors = latent_vectors.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Handle one-hot labels
    if labels is not None and labels.ndim > 1:
        labels = np.argmax(labels, axis=-1)
    
    # Dimensionality reduction
    try:
        if method.lower() == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
        
        latent_reduced = reducer.fit_transform(latent_vectors)
    except ImportError:
        # Fallback: use first two dimensions if sklearn not available
        if latent_vectors.shape[1] >= 2:
            latent_reduced = latent_vectors[:, :2]
        else:
            raise ImportError("scikit-learn is required for latent space visualization. Install with: pip install scikit-learn")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        scatter = ax.scatter(latent_reduced[:, 0], latent_reduced[:, 1],
                           c=labels, cmap="tab10", alpha=0.6, s=50)
        plt.colorbar(scatter, ax=ax, label="Class")
    else:
        ax.scatter(latent_reduced[:, 0], latent_reduced[:, 1], alpha=0.6, s=50)
    
    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filename = filename or f"latent_space_{method}.png"
        save_figure(fig, filename, subdir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_model_architecture(
    model,
    input_shape: Tuple[int, ...],
    title: str = "Model Architecture",
    show: bool = False,
    save: bool = True,
    filename: Optional[str] = None,
    subdir: str = "models",
) -> plt.Figure:
    """Plot model architecture diagram.
    
    Args:
        model: PyTorch model
        input_shape: Input shape tuple
        title: Title for the plot
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Output filename
        subdir: Subdirectory within output folder
        
    Returns:
        Matplotlib figure object
    """
    setup_plot_style()
    
    try:
        from torchinfo import summary
        from io import StringIO
        import sys
        
        # Capture model summary
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        summary(model, input_shape, verbose=0)
        summary_str = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        # Create figure with text
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.text(0.1, 0.5, summary_str, fontfamily="monospace", fontsize=8,
               verticalalignment="center", transform=ax.transAxes)
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        
    except ImportError:
        # Fallback: simple text representation
        fig, ax = plt.subplots(figsize=(10, 6))
        model_str = str(model)
        ax.text(0.1, 0.5, model_str, fontfamily="monospace", fontsize=8,
               verticalalignment="center", transform=ax.transAxes)
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    
    plt.tight_layout()
    
    if save:
        filename = filename or "model_architecture.png"
        save_figure(fig, filename, subdir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def save_model_visualizations(
    feature_maps: Optional[Union[torch.Tensor, np.ndarray]] = None,
    latent_vectors: Optional[Union[torch.Tensor, np.ndarray]] = None,
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    model: Optional[torch.nn.Module] = None,
    input_shape: Optional[Tuple[int, ...]] = None,
    prefix: str = "model",
    subdir: str = "models",
) -> dict:
    """Save all model visualizations.
    
    Args:
        feature_maps: Feature maps to visualize
        latent_vectors: Latent vectors to visualize
        labels: Optional labels
        model: Model to visualize architecture
        input_shape: Input shape for architecture visualization
        prefix: Prefix for filenames
        subdir: Subdirectory within output folder
        
    Returns:
        Dictionary of saved file paths
    """
    saved_files = {}
    
    if feature_maps is not None:
        fig = plot_feature_maps(feature_maps, save=True, show=False,
                               filename=f"{prefix}_feature_maps.png", subdir=subdir)
        saved_files["feature_maps"] = str(get_output_path(f"{prefix}_feature_maps.png", subdir))
    
    if latent_vectors is not None:
        fig = plot_latent_space(latent_vectors, labels, save=True, show=False,
                               filename=f"{prefix}_latent_space.png", subdir=subdir)
        saved_files["latent_space"] = str(get_output_path(f"{prefix}_latent_space.png", subdir))
    
    if model is not None and input_shape is not None:
        fig = plot_model_architecture(model, input_shape, save=True, show=False,
                                     filename=f"{prefix}_architecture.png", subdir=subdir)
        saved_files["architecture"] = str(get_output_path(f"{prefix}_architecture.png", subdir))
    
    return saved_files
