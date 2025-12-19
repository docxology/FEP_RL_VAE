"""Utility functions for visualization module."""

import os
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import matplotlib.style as style


# Default output directory
DEFAULT_OUTPUT_DIR = Path("output")


def ensure_output_dir(subdir: str = "") -> Path:
    """Ensure output directory exists.
    
    Args:
        subdir: Subdirectory within output folder
        
    Returns:
        Path to the output directory
    """
    output_path = DEFAULT_OUTPUT_DIR / subdir if subdir else DEFAULT_OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def get_output_path(filename: str, subdir: str = "") -> Path:
    """Get full path for output file.
    
    Args:
        filename: Name of the output file
        subdir: Subdirectory within output folder
        
    Returns:
        Full path to output file
    """
    output_dir = ensure_output_dir(subdir)
    return output_dir / filename


def save_figure(fig, filename: str, subdir: str = "", dpi: int = 300, 
                bbox_inches: str = "tight", **kwargs) -> Path:
    """Save matplotlib figure to output directory.
    
    Args:
        fig: Matplotlib figure object
        filename: Name of the output file
        subdir: Subdirectory within output folder
        dpi: Resolution for saved figure
        bbox_inches: Bounding box for saved figure
        **kwargs: Additional arguments for savefig
        
    Returns:
        Path to saved file
    """
    output_path = get_output_path(filename, subdir)
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
    return output_path


def setup_plot_style(style_name: str = "default"):
    """Setup matplotlib plotting style.
    
    Args:
        style_name: Name of matplotlib style to use
    """
    try:
        plt.style.use(style_name)
    except OSError:
        # Fallback to default if style not found
        plt.style.use("default")
    
    # Set default parameters
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3


def create_timestamped_filename(base_name: str, extension: str = "png") -> str:
    """Create timestamped filename.
    
    Args:
        base_name: Base name for the file
        extension: File extension
        
    Returns:
        Timestamped filename
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{base_name}_{timestamp}.{extension}"
