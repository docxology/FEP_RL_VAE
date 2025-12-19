"""Training history visualization functions for FEP-RL-VAE."""

from typing import Dict, List, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from .utils import ensure_output_dir, get_output_path, save_figure, setup_plot_style


def plot_loss_curves(
    epoch_history: Dict,
    title: str = "Training Loss Curves",
    show: bool = False,
    save: bool = True,
    filename: Optional[str] = None,
    subdir: str = "training",
) -> plt.Figure:
    """Plot accuracy and complexity loss curves.
    
    Args:
        epoch_history: Dictionary containing training history
        title: Title for the plot
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Output filename
        subdir: Subdirectory within output folder
        
    Returns:
        Matplotlib figure object
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    
    # Accuracy losses
    ax1 = axes[0]
    has_accuracy = False
    if "accuracy_losses" in epoch_history:
        for key, values in epoch_history["accuracy_losses"].items():
            ax1.plot(values, label=f"Accuracy Loss ({key})", linewidth=2)
            has_accuracy = True
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Accuracy Losses", fontsize=12)
    if has_accuracy:
        ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Complexity losses
    ax2 = axes[1]
    has_complexity = False
    if "complexity_losses" in epoch_history:
        for key, values in epoch_history["complexity_losses"].items():
            ax2.plot(values, label=f"Complexity Loss ({key})", linewidth=2)
            has_complexity = True
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Complexity Losses", fontsize=12)
    if has_complexity:
        ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filename = filename or "loss_curves.png"
        save_figure(fig, filename, subdir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_reward_curves(
    epoch_history: Dict,
    title: str = "Reward Curves",
    show: bool = False,
    save: bool = True,
    filename: Optional[str] = None,
    subdir: str = "training",
) -> plt.Figure:
    """Plot reward curves.
    
    Args:
        epoch_history: Dictionary containing training history
        title: Title for the plot
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Output filename
        subdir: Subdirectory within output folder
        
    Returns:
        Matplotlib figure object
    """
    setup_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if "total_reward" in epoch_history:
        ax.plot(epoch_history["total_reward"], label="Total Reward", linewidth=2)
    if "reward" in epoch_history:
        ax.plot(epoch_history["reward"], label="Reward", linewidth=2, alpha=0.7)
    if "curiosities" in epoch_history:
        for key, values in epoch_history["curiosities"].items():
            ax.plot(values, label=f"Curiosity ({key})", linewidth=2, alpha=0.7)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filename = filename or "reward_curves.png"
        save_figure(fig, filename, subdir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_entropy_curves(
    epoch_history: Dict,
    title: str = "Entropy Curves",
    show: bool = False,
    save: bool = True,
    filename: Optional[str] = None,
    subdir: str = "training",
) -> plt.Figure:
    """Plot entropy curves.
    
    Args:
        epoch_history: Dictionary containing training history
        title: Title for the plot
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Output filename
        subdir: Subdirectory within output folder
        
    Returns:
        Matplotlib figure object
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    
    # Alpha entropies
    ax1 = axes[0]
    if "alpha_entropies" in epoch_history:
        for key, values in epoch_history["alpha_entropies"].items():
            ax1.plot(values, label=f"Alpha Entropy ({key})", linewidth=2)
    if "alpha_normal_entropies" in epoch_history:
        for key, values in epoch_history["alpha_normal_entropies"].items():
            ax1.plot(values, label=f"Alpha Normal Entropy ({key})", linewidth=2, linestyle="--")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Entropy")
    ax1.set_title("Alpha Entropies", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Total entropies
    ax2 = axes[1]
    if "total_entropies" in epoch_history:
        for key, values in epoch_history["total_entropies"].items():
            ax2.plot(values, label=f"Total Entropy ({key})", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Entropy")
    ax2.set_title("Total Entropies", fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filename = filename or "entropy_curves.png"
        save_figure(fig, filename, subdir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_learning_curves(
    epoch_history: Dict,
    title: str = "Learning Curves",
    show: bool = False,
    save: bool = True,
    filename: Optional[str] = None,
    subdir: str = "training",
) -> plt.Figure:
    """Plot comprehensive learning curves including actor and critic losses.
    
    Args:
        epoch_history: Dictionary containing training history
        title: Title for the plot
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Output filename
        subdir: Subdirectory within output folder
        
    Returns:
        Matplotlib figure object
    """
    setup_plot_style()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    
    # Actor loss
    ax1 = axes[0]
    has_actor = False
    if "actor_loss" in epoch_history:
        ax1.plot(epoch_history["actor_loss"], label="Actor Loss", linewidth=2, color="blue")
        has_actor = True
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Actor Loss", fontsize=12)
    if has_actor:
        ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Critic losses
    ax2 = axes[1]
    has_critic = False
    if "critic_losses" in epoch_history:
        for i, critic_loss in enumerate(epoch_history["critic_losses"]):
            ax2.plot(critic_loss, label=f"Critic {i} Loss", linewidth=2)
            has_critic = True
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Critic Losses", fontsize=12)
    if has_critic:
        ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filename = filename or "learning_curves.png"
        save_figure(fig, filename, subdir)
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_training_history(
    epoch_history: Dict,
    show: bool = False,
    save: bool = True,
    prefix: str = "training",
    subdir: str = "training",
) -> Dict[str, str]:
    """Plot comprehensive training history with all metrics.
    
    Args:
        epoch_history: Dictionary containing training history
        show: Whether to display the plots
        save: Whether to save the plots
        prefix: Prefix for filenames
        subdir: Subdirectory within output folder
        
    Returns:
        Dictionary of saved file paths
    """
    saved_files = {}
    
    # Loss curves
    fig = plot_loss_curves(epoch_history, save=save, show=show,
                          filename=f"{prefix}_losses.png", subdir=subdir)
    saved_files["losses"] = str(get_output_path(f"{prefix}_losses.png", subdir))
    
    # Reward curves
    fig = plot_reward_curves(epoch_history, save=save, show=show,
                            filename=f"{prefix}_rewards.png", subdir=subdir)
    saved_files["rewards"] = str(get_output_path(f"{prefix}_rewards.png", subdir))
    
    # Entropy curves
    fig = plot_entropy_curves(epoch_history, save=save, show=show,
                             filename=f"{prefix}_entropies.png", subdir=subdir)
    saved_files["entropies"] = str(get_output_path(f"{prefix}_entropies.png", subdir))
    
    # Learning curves
    fig = plot_learning_curves(epoch_history, save=save, show=show,
                              filename=f"{prefix}_learning.png", subdir=subdir)
    saved_files["learning"] = str(get_output_path(f"{prefix}_learning.png", subdir))
    
    return saved_files


def save_training_visualizations(
    epoch_history: Dict,
    prefix: str = "training",
    subdir: str = "training",
) -> Dict[str, str]:
    """Save all training visualizations (always saves, never shows).
    
    Args:
        epoch_history: Dictionary containing training history
        prefix: Prefix for filenames
        subdir: Subdirectory within output folder
        
    Returns:
        Dictionary of saved file paths
    """
    return plot_training_history(epoch_history, show=False, save=True, prefix=prefix, subdir=subdir)
