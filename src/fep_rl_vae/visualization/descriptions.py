"""Description and embedding visualization functions for FEP-RL-VAE."""

from typing import List, Optional, Union, Tuple, Dict
import numpy as np
import torch
import matplotlib.pyplot as plt

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .utils import ensure_output_dir, get_output_path, save_figure, setup_plot_style


def plot_embedding_space(
    embeddings: Union[torch.Tensor, np.ndarray],
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: str = "Embedding Space",
    method: str = "tsne",
    n_components: int = 2,
    show: bool = False,
    save: bool = True,
    filename: Optional[str] = None,
    subdir: str = "descriptions",
) -> plt.Figure:
    """Plot embedding space using dimensionality reduction.
    
    Args:
        embeddings: Embedding vectors (N, dim)
        labels: Optional labels for coloring points
        title: Title for the plot
        method: Dimensionality reduction method ('tsne' or 'pca')
        n_components: Number of dimensions for reduction (2 or 3)
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Output filename
        subdir: Subdirectory within output folder
        
    Returns:
        Matplotlib figure object
    """
    setup_plot_style()
    
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Handle one-hot labels
    if labels is not None and labels.ndim > 1:
        labels = np.argmax(labels, axis=-1)
    
    # Dimensionality reduction
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for embedding visualization. Install with: pip install scikit-learn")
    
    if method.lower() == "tsne":
        reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
    else:
        reducer = PCA(n_components=n_components)
    
    embeddings_reduced = reducer.fit_transform(embeddings)
    
    # Create figure
    if n_components == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if labels is not None:
            scatter = ax.scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1],
                               c=labels, cmap="tab10", alpha=0.6, s=50)
            plt.colorbar(scatter, ax=ax, label="Class")
        else:
            ax.scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1],
                      alpha=0.6, s=50)
        
        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
    else:  # 3D
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")
        
        if labels is not None:
            scatter = ax.scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1],
                               embeddings_reduced[:, 2], c=labels, cmap="tab10", alpha=0.6, s=50)
            plt.colorbar(scatter, ax=ax, label="Class")
        else:
            ax.scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1],
                      embeddings_reduced[:, 2], alpha=0.6, s=50)
        
        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        ax.set_zlabel(f"{method.upper()} Component 3")
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filename = filename or f"embedding_space_{method}_{n_components}d.png"
        save_figure(fig, filename, subdir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_similarity_matrix(
    embeddings: Union[torch.Tensor, np.ndarray],
    title: str = "Similarity Matrix",
    n_samples: Optional[int] = None,
    show: bool = False,
    save: bool = True,
    filename: Optional[str] = None,
    subdir: str = "descriptions",
) -> plt.Figure:
    """Plot cosine similarity matrix between embeddings.
    
    Args:
        embeddings: Embedding vectors (N, dim)
        title: Title for the plot
        n_samples: Number of samples to include (all if None)
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Output filename
        subdir: Subdirectory within output folder
        
    Returns:
        Matplotlib figure object
    """
    setup_plot_style()
    
    # Convert to numpy if needed
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    
    if n_samples is not None:
        embeddings = embeddings[:n_samples]
    
    # Compute cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / (norms + 1e-8)
    similarity = np.dot(embeddings_norm, embeddings_norm.T)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(similarity, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.figure.colorbar(im, ax=ax, label="Cosine Similarity")
    
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Sample Index")
    
    plt.tight_layout()
    
    if save:
        filename = filename or "similarity_matrix.png"
        save_figure(fig, filename, subdir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_vector_components(
    vectors: Union[torch.Tensor, np.ndarray],
    title: str = "Vector Components",
    n_samples: int = 10,
    show: bool = False,
    save: bool = True,
    filename: Optional[str] = None,
    subdir: str = "descriptions",
) -> plt.Figure:
    """Plot individual components of vectors.
    
    Args:
        vectors: Vector data (N, dim)
        title: Title for the plot
        n_samples: Number of samples to visualize
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Output filename
        subdir: Subdirectory within output folder
        
    Returns:
        Matplotlib figure object
    """
    setup_plot_style()
    
    # Convert to numpy if needed
    if isinstance(vectors, torch.Tensor):
        vectors = vectors.detach().cpu().numpy()
    
    n_samples = min(n_samples, len(vectors))
    dim = vectors.shape[-1]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(dim)
    width = 0.8 / n_samples
    
    for i in range(n_samples):
        offset = (i - n_samples/2) * width
        ax.bar(x + offset, vectors[i], width, label=f"Sample {i+1}", alpha=0.7)
    
    ax.set_xlabel("Component Index")
    ax.set_ylabel("Value")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    if save:
        filename = filename or f"vector_components_{n_samples}samples.png"
        save_figure(fig, filename, subdir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def save_description_visualizations(
    embeddings: Union[torch.Tensor, np.ndarray],
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    prefix: str = "description",
    subdir: str = "descriptions",
) -> Dict[str, str]:
    """Save all description visualizations.
    
    Args:
        embeddings: Embedding vectors
        labels: Optional labels
        prefix: Prefix for filenames
        subdir: Subdirectory within output folder
        
    Returns:
        Dictionary of saved file paths
    """
    from typing import Dict
    
    saved_files = {}
    
    # Embedding space (2D)
    fig = plot_embedding_space(embeddings, labels, save=True, show=False,
                              filename=f"{prefix}_embedding_2d.png", subdir=subdir)
    saved_files["embedding_2d"] = str(get_output_path(f"{prefix}_embedding_2d.png", subdir))
    
    # Similarity matrix
    fig = plot_similarity_matrix(embeddings, save=True, show=False,
                                filename=f"{prefix}_similarity.png", subdir=subdir)
    saved_files["similarity"] = str(get_output_path(f"{prefix}_similarity.png", subdir))
    
    # Vector components
    fig = plot_vector_components(embeddings, save=True, show=False,
                               filename=f"{prefix}_components.png", subdir=subdir)
    saved_files["components"] = str(get_output_path(f"{prefix}_components.png", subdir))
    
    return saved_files
