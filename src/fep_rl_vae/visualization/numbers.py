"""Number and categorical visualization functions for FEP-RL-VAE."""

import math
from typing import List, Optional, Union, Tuple, Dict
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .utils import ensure_output_dir, get_output_path, save_figure, setup_plot_style


def plot_number_distribution(
    numbers: Union[List[int], torch.Tensor, np.ndarray],
    title: str = "Number Distribution",
    n_classes: int = 10,
    show: bool = False,
    save: bool = True,
    filename: Optional[str] = None,
    subdir: str = "numbers",
) -> plt.Figure:
    """Plot distribution of number classes.
    
    Args:
        numbers: List or tensor of number labels (can be one-hot or indices)
        title: Title for the plot
        n_classes: Number of classes
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Output filename
        subdir: Subdirectory within output folder
        
    Returns:
        Matplotlib figure object
    """
    setup_plot_style()
    
    # Convert to numpy if needed
    if isinstance(numbers, torch.Tensor):
        numbers = numbers.detach().cpu().numpy()
    elif isinstance(numbers, list):
        numbers = np.array(numbers)
    
    # Handle one-hot encoding
    if numbers.ndim > 1:
        numbers = np.argmax(numbers, axis=-1)
    
    # Count occurrences
    counts = np.bincount(numbers, minlength=n_classes)
    classes = np.arange(n_classes)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(classes, counts, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Digit Class")
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(classes)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save:
        filename = filename or "number_distribution.png"
        save_figure(fig, filename, subdir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_prediction_probabilities(
    probabilities: Union[torch.Tensor, np.ndarray],
    true_labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    title: str = "Prediction Probabilities",
    n_samples: int = 10,
    show: bool = False,
    save: bool = True,
    filename: Optional[str] = None,
    subdir: str = "numbers",
) -> plt.Figure:
    """Plot prediction probabilities as bar charts.
    
    Args:
        probabilities: Prediction probabilities (N, n_classes)
        true_labels: True labels for comparison (optional)
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
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.detach().cpu().numpy()
    
    n_samples = min(n_samples, len(probabilities))
    n_classes = probabilities.shape[-1]
    
    # Handle one-hot true labels
    if true_labels is not None and true_labels.ndim > 1:
        true_labels = np.argmax(true_labels, axis=-1)
    
    n_cols = min(5, n_samples)
    n_rows = math.ceil(n_samples / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2.5))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    
    if n_samples == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    classes = np.arange(n_classes)
    
    for i in range(n_samples):
        ax = axes[i]
        probs = probabilities[i]
        bars = ax.bar(classes, probs, alpha=0.7, edgecolor="black")
        
        # Highlight true label if provided
        if true_labels is not None:
            true_label = true_labels[i]
            bars[true_label].set_color("green")
            bars[true_label].set_alpha(0.9)
        
        # Highlight predicted label
        pred_label = np.argmax(probs)
        bars[pred_label].set_edgecolor("red")
        bars[pred_label].set_linewidth(2)
        
        ax.set_xlabel("Class")
        ax.set_ylabel("Probability")
        ax.set_title(f"Sample {i+1}", fontsize=10)
        ax.set_xticks(classes)
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis="y")
    
    # Hide unused subplots
    for i in range(n_samples, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    
    if save:
        filename = filename or f"prediction_probabilities_{n_samples}samples.png"
        save_figure(fig, filename, subdir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_confusion_matrix(
    predictions: Union[torch.Tensor, np.ndarray],
    true_labels: Union[torch.Tensor, np.ndarray],
    title: str = "Confusion Matrix",
    n_classes: int = 10,
    normalize: bool = True,
    show: bool = False,
    save: bool = True,
    filename: Optional[str] = None,
    subdir: str = "numbers",
) -> plt.Figure:
    """Plot confusion matrix.
    
    Args:
        predictions: Predicted labels (can be one-hot or indices)
        true_labels: True labels (can be one-hot or indices)
        title: Title for the plot
        n_classes: Number of classes
        normalize: Whether to normalize the matrix
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Output filename
        subdir: Subdirectory within output folder
        
    Returns:
        Matplotlib figure object
    """
    setup_plot_style()
    
    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.detach().cpu().numpy()
    
    # Handle one-hot encoding
    if predictions.ndim > 1:
        predictions = np.argmax(predictions, axis=-1)
    if true_labels.ndim > 1:
        true_labels = np.argmax(true_labels, axis=-1)
    
    # Compute confusion matrix
    cm = np.zeros((n_classes, n_classes), dtype=np.int32)
    for true, pred in zip(true_labels, predictions):
        cm[int(true), int(pred)] += 1
    
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=np.arange(n_classes),
           yticklabels=np.arange(n_classes),
           title=title,
           ylabel="True Label",
           xlabel="Predicted Label")
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            text = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text, ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black", fontsize=9)
    
    plt.tight_layout()
    
    if save:
        filename = filename or "confusion_matrix.png"
        save_figure(fig, filename, subdir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_sequence_predictions(
    sequences: Union[List, torch.Tensor, np.ndarray],
    predictions: Union[List, torch.Tensor, np.ndarray],
    true_labels: Optional[Union[List, torch.Tensor, np.ndarray]] = None,
    title: str = "Sequence Predictions",
    n_sequences: int = 5,
    show: bool = False,
    save: bool = True,
    filename: Optional[str] = None,
    subdir: str = "numbers",
) -> plt.Figure:
    """Plot predictions for sequences of numbers.
    
    Args:
        sequences: Input sequences
        predictions: Predicted sequences
        true_labels: True labels for comparison (optional)
        title: Title for the plot
        n_sequences: Number of sequences to visualize
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Output filename
        subdir: Subdirectory within output folder
        
    Returns:
        Matplotlib figure object
    """
    setup_plot_style()
    
    # Convert to numpy if needed
    if isinstance(sequences, torch.Tensor):
        sequences = sequences.detach().cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(true_labels, torch.Tensor):
        true_labels = true_labels.detach().cpu().numpy()
    
    # Handle one-hot encoding
    if predictions.ndim > 2:
        predictions = np.argmax(predictions, axis=-1)
    if true_labels is not None and true_labels.ndim > 2:
        true_labels = np.argmax(true_labels, axis=-1)
    
    n_sequences = min(n_sequences, len(sequences))
    seq_length = predictions.shape[-1]
    
    fig, axes = plt.subplots(n_sequences, 1, figsize=(12, n_sequences * 2))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    
    if n_sequences == 1:
        axes = [axes]
    
    for i in range(n_sequences):
        ax = axes[i]
        pred_seq = predictions[i]
        x = np.arange(seq_length)
        
        ax.plot(x, pred_seq, marker="o", label="Predicted", linewidth=2, markersize=8)
        
        if true_labels is not None:
            true_seq = true_labels[i]
            ax.plot(x, true_seq, marker="s", label="True", linewidth=2, markersize=8, alpha=0.7)
        
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Class")
        ax.set_title(f"Sequence {i+1}", fontsize=10)
        ax.set_xticks(x)
        ax.set_yticks(np.arange(10))
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filename = filename or f"sequence_predictions_{n_sequences}seqs.png"
        save_figure(fig, filename, subdir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def save_number_visualizations(
    numbers: Union[List[int], torch.Tensor, np.ndarray],
    predictions: Optional[Union[torch.Tensor, np.ndarray]] = None,
    true_labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    prefix: str = "number",
    subdir: str = "numbers",
) -> Dict[str, str]:
    """Save all number visualizations.
    
    Args:
        numbers: Number labels
        predictions: Predictions (optional)
        true_labels: True labels (optional)
        prefix: Prefix for filenames
        subdir: Subdirectory within output folder
        
    Returns:
        Dictionary of saved file paths
    """
    saved_files = {}
    
    # Distribution
    fig = plot_number_distribution(numbers, save=True, show=False, 
                                   filename=f"{prefix}_distribution.png", subdir=subdir)
    saved_files["distribution"] = str(get_output_path(f"{prefix}_distribution.png", subdir))
    
    if predictions is not None:
        # Prediction probabilities
        fig = plot_prediction_probabilities(predictions, true_labels, save=True, show=False,
                                           filename=f"{prefix}_probabilities.png", subdir=subdir)
        saved_files["probabilities"] = str(get_output_path(f"{prefix}_probabilities.png", subdir))
        
        if true_labels is not None:
            # Confusion matrix
            fig = plot_confusion_matrix(predictions, true_labels, save=True, show=False,
                                       filename=f"{prefix}_confusion.png", subdir=subdir)
            saved_files["confusion"] = str(get_output_path(f"{prefix}_confusion.png", subdir))
    
    return saved_files
