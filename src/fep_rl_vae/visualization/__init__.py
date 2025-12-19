"""Comprehensive visualization module for FEP-RL-VAE.

This module provides visualization capabilities for all data formats:
- Images: grids, sequences, comparisons
- Numbers: distributions, predictions, confusion matrices
- Descriptions: embeddings, similarities, vectors
- Training: loss curves, rewards, metrics
- Models: architectures, feature maps, latent spaces
"""

from .images import (
    plot_image_grid,
    plot_image_sequence,
    plot_image_comparison,
    plot_image_distribution,
    save_image_grid,
    save_image_sequence,
    save_image_comparison,
)
from .numbers import (
    plot_number_distribution,
    plot_prediction_probabilities,
    plot_confusion_matrix,
    plot_sequence_predictions,
    save_number_visualizations,
)
from .descriptions import (
    plot_embedding_space,
    plot_similarity_matrix,
    plot_vector_components,
    save_description_visualizations,
)
from .training import (
    plot_training_history,
    plot_loss_curves,
    plot_reward_curves,
    plot_entropy_curves,
    plot_learning_curves,
    save_training_visualizations,
)
from .models import (
    plot_feature_maps,
    plot_latent_space,
    plot_model_architecture,
    save_model_visualizations,
)
from .utils import (
    ensure_output_dir,
    get_output_path,
    save_figure,
    setup_plot_style,
)

__all__ = [
    # Image visualizations
    "plot_image_grid",
    "plot_image_sequence",
    "plot_image_comparison",
    "plot_image_distribution",
    "save_image_grid",
    "save_image_sequence",
    "save_image_comparison",
    # Number visualizations
    "plot_number_distribution",
    "plot_prediction_probabilities",
    "plot_confusion_matrix",
    "plot_sequence_predictions",
    "save_number_visualizations",
    # Description visualizations
    "plot_embedding_space",
    "plot_similarity_matrix",
    "plot_vector_components",
    "save_description_visualizations",
    # Training visualizations
    "plot_training_history",
    "plot_loss_curves",
    "plot_reward_curves",
    "plot_entropy_curves",
    "plot_learning_curves",
    "save_training_visualizations",
    # Model visualizations
    "plot_feature_maps",
    "plot_latent_space",
    "plot_model_architecture",
    "save_model_visualizations",
    # Utilities
    "ensure_output_dir",
    "get_output_path",
    "save_figure",
    "setup_plot_style",
]
