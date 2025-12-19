"""Image visualization functions for FEP-RL-VAE."""

import math
from typing import List, Optional, Union, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .utils import ensure_output_dir, get_output_path, save_figure, setup_plot_style


def plot_image_grid(
    images: Union[List, torch.Tensor, np.ndarray],
    title: str = "Image Grid",
    n_cols: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "gray",
    show: bool = False,
    save: bool = True,
    filename: Optional[str] = None,
    subdir: str = "images",
) -> plt.Figure:
    """Plot a grid of images.
    
    Args:
        images: List or tensor of images (N, H, W) or (N, H, W, C)
        title: Title for the plot
        n_cols: Number of columns (auto-calculated if None)
        figsize: Figure size (auto-calculated if None)
        cmap: Colormap to use
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Output filename (auto-generated if None)
        subdir: Subdirectory within output folder
        
    Returns:
        Matplotlib figure object
    """
    setup_plot_style()
    
    # Convert to numpy if needed
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    elif isinstance(images, list):
        images = np.array([img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img 
                          for img in images])
    
    n_images = len(images)
    if n_images == 0:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.text(0.5, 0.5, "No images to display", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.suptitle(title)
        if save:
            filename = filename or "empty_grid.png"
            save_figure(fig, filename, subdir)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
    
    # Handle different image shapes
    if images.ndim == 4:
        # (N, H, W, C) -> (N, H, W)
        if images.shape[-1] == 1:
            images = images.squeeze(-1)
        elif images.shape[-1] == 3:
            cmap = None  # RGB images don't use colormap
    
    # Calculate grid dimensions
    if n_cols is None:
        n_cols = math.ceil(math.sqrt(n_images))
    n_rows = math.ceil(n_images / n_cols)
    
    # Set figure size
    if figsize is None:
        figsize = (n_cols * 2, n_rows * 2)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    
    # Handle single subplot case
    if n_images == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for i in range(n_rows * n_cols):
        ax = axes[i]
        if i < n_images:
            img = images[i]
            if cmap is None:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap)
        ax.axis("off")
    
    plt.tight_layout()
    
    if save:
        filename = filename or f"image_grid_{n_images}images.png"
        save_figure(fig, filename, subdir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_image_sequence(
    images: Union[List, torch.Tensor, np.ndarray],
    title: str = "Image Sequence",
    labels: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 3),
    cmap: str = "gray",
    show: bool = False,
    save: bool = True,
    filename: Optional[str] = None,
    subdir: str = "images",
) -> plt.Figure:
    """Plot a sequence of images in a horizontal row.
    
    Args:
        images: List or tensor of images
        title: Title for the plot
        labels: Optional labels for each image
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
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    elif isinstance(images, list):
        images = np.array([img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img 
                          for img in images])
    
    n_images = len(images)
    if n_images == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No images to display", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        fig.suptitle(title)
        if save:
            filename = filename or "empty_sequence.png"
            save_figure(fig, filename, subdir)
        if show:
            plt.show()
        else:
            plt.close(fig)
        return fig
    
    # Handle different image shapes
    if images.ndim == 4:
        if images.shape[-1] == 1:
            images = images.squeeze(-1)
        elif images.shape[-1] == 3:
            cmap = None
    
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    
    if n_images == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        if i < len(images):
            img = images[i]
            if cmap is None:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap)
            if labels and i < len(labels):
                ax.set_title(labels[i], fontsize=10)
        ax.axis("off")
    
    plt.tight_layout()
    
    if save:
        filename = filename or f"image_sequence_{n_images}images.png"
        save_figure(fig, filename, subdir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_image_comparison(
    original: Union[List, torch.Tensor, np.ndarray],
    reconstructed: Union[List, torch.Tensor, np.ndarray],
    title: str = "Original vs Reconstructed",
    n_samples: Optional[int] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "gray",
    show: bool = False,
    save: bool = True,
    filename: Optional[str] = None,
    subdir: str = "comparisons",
) -> plt.Figure:
    """Plot side-by-side comparison of original and reconstructed images.
    
    Args:
        original: Original images
        reconstructed: Reconstructed images
        title: Title for the plot
        n_samples: Number of samples to show (all if None)
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
    if isinstance(original, torch.Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, torch.Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()
    
    if isinstance(original, list):
        original = np.array([img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img 
                            for img in original])
    if isinstance(reconstructed, list):
        reconstructed = np.array([img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img 
                                 for img in reconstructed])
    
    n_samples = n_samples or min(len(original), len(reconstructed))
    n_samples = min(n_samples, len(original), len(reconstructed))
    
    # Handle different image shapes
    if original.ndim == 4:
        if original.shape[-1] == 1:
            original = original.squeeze(-1)
    if reconstructed.ndim == 4:
        if reconstructed.shape[-1] == 1:
            reconstructed = reconstructed.squeeze(-1)
    
    if figsize is None:
        figsize = (10, n_samples * 2)
    
    fig, axes = plt.subplots(n_samples, 2, figsize=figsize)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_samples):
        # Original
        ax_orig = axes[i, 0]
        img_orig = original[i]
        ax_orig.imshow(img_orig, cmap=cmap)
        ax_orig.set_title("Original", fontsize=10)
        ax_orig.axis("off")
        
        # Reconstructed
        ax_recon = axes[i, 1]
        img_recon = reconstructed[i]
        ax_recon.imshow(img_recon, cmap=cmap)
        ax_recon.set_title("Reconstructed", fontsize=10)
        ax_recon.axis("off")
    
    plt.tight_layout()
    
    if save:
        filename = filename or f"comparison_{n_samples}samples.png"
        save_figure(fig, filename, subdir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_image_distribution(
    images: Union[List, torch.Tensor, np.ndarray],
    title: str = "Image Distribution",
    bins: int = 50,
    show: bool = False,
    save: bool = True,
    filename: Optional[str] = None,
    subdir: str = "images",
) -> plt.Figure:
    """Plot pixel value distribution of images.
    
    Args:
        images: List or tensor of images
        title: Title for the plot
        bins: Number of bins for histogram
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Output filename
        subdir: Subdirectory within output folder
        
    Returns:
        Matplotlib figure object
    """
    setup_plot_style()
    
    # Convert to numpy if needed
    if isinstance(images, torch.Tensor):
        images = images.detach().cpu().numpy()
    elif isinstance(images, list):
        images = np.array([img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img 
                          for img in images])
    
    # Flatten all pixels
    pixels = images.flatten()
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pixels, bins=bins, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Frequency")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filename = filename or "image_distribution.png"
        save_figure(fig, filename, subdir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


# Convenience functions that always save
def save_image_grid(images, title="Image Grid", **kwargs):
    """Save image grid (always saves, never shows)."""
    return plot_image_grid(images, title=title, show=False, save=True, **kwargs)


def save_image_sequence(images, title="Image Sequence", **kwargs):
    """Save image sequence (always saves, never shows)."""
    return plot_image_sequence(images, title=title, show=False, save=True, **kwargs)


def save_image_comparison(original, reconstructed, title="Original vs Reconstructed", **kwargs):
    """Save image comparison (always saves, never shows)."""
    return plot_image_comparison(original, reconstructed, title=title, show=False, save=True, **kwargs)
