"""Tests for visualization module."""

import os
import pytest
import numpy as np
import torch
from pathlib import Path

from fep_rl_vae.visualization import (
    # Image visualizations
    plot_image_grid,
    plot_image_sequence,
    plot_image_comparison,
    plot_image_distribution,
    save_image_grid,
    # Number visualizations
    plot_number_distribution,
    plot_prediction_probabilities,
    plot_confusion_matrix,
    plot_sequence_predictions,
    # Description visualizations
    plot_similarity_matrix,
    plot_vector_components,
    # Training visualizations
    plot_loss_curves,
    plot_reward_curves,
    plot_entropy_curves,
    plot_learning_curves,
    save_training_visualizations,
    # Model visualizations
    plot_feature_maps,
    plot_latent_space,
    # Utilities
    ensure_output_dir,
    get_output_path,
    save_figure,
    setup_plot_style,
)


class TestImageVisualizations:
    """Test image visualization functions."""

    @pytest.fixture
    def sample_images(self):
        """Create sample images."""
        return torch.randn(9, 28, 28, 1)

    def test_plot_image_grid(self, sample_images):
        """Test image grid plotting."""
        fig = plot_image_grid(sample_images, save=False, show=False)
        assert fig is not None

    def test_plot_image_grid_empty(self):
        """Test image grid with empty list."""
        fig = plot_image_grid([], save=False, show=False)
        assert fig is not None

    def test_plot_image_sequence(self, sample_images):
        """Test image sequence plotting."""
        fig = plot_image_sequence(sample_images[:5], save=False, show=False)
        assert fig is not None

    def test_plot_image_comparison(self, sample_images):
        """Test image comparison plotting."""
        original = sample_images[:3]
        reconstructed = torch.randn(3, 28, 28, 1)
        fig = plot_image_comparison(original, reconstructed, save=False, show=False)
        assert fig is not None

    def test_plot_image_distribution(self, sample_images):
        """Test image distribution plotting."""
        fig = plot_image_distribution(sample_images, save=False, show=False)
        assert fig is not None

    def test_save_image_grid(self, sample_images):
        """Test saving image grid."""
        fig = save_image_grid(sample_images, filename="test_grid.png")
        assert Path("output/images/test_grid.png").exists()
        Path("output/images/test_grid.png").unlink()  # Cleanup


class TestNumberVisualizations:
    """Test number visualization functions."""

    @pytest.fixture
    def sample_numbers(self):
        """Create sample number labels."""
        return torch.randint(0, 10, (100,))

    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions."""
        return torch.rand(10, 10)  # (n_samples, n_classes)

    def test_plot_number_distribution(self, sample_numbers):
        """Test number distribution plotting."""
        fig = plot_number_distribution(sample_numbers, save=False, show=False)
        assert fig is not None

    def test_plot_prediction_probabilities(self, sample_predictions):
        """Test prediction probabilities plotting."""
        true_labels = torch.randint(0, 10, (10,))
        fig = plot_prediction_probabilities(sample_predictions, true_labels, save=False, show=False)
        assert fig is not None

    def test_plot_confusion_matrix(self, sample_numbers):
        """Test confusion matrix plotting."""
        predictions = torch.randint(0, 10, (100,))
        fig = plot_confusion_matrix(predictions, sample_numbers, save=False, show=False)
        assert fig is not None

    def test_plot_sequence_predictions(self):
        """Test sequence predictions plotting."""
        sequences = torch.randn(5, 10)  # (n_sequences, seq_length)
        predictions = torch.randint(0, 10, (5, 10))
        true_labels = torch.randint(0, 10, (5, 10))
        fig = plot_sequence_predictions(sequences, predictions, true_labels, save=False, show=False)
        assert fig is not None


class TestDescriptionVisualizations:
    """Test description visualization functions."""

    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings."""
        return torch.randn(50, 64)  # (n_samples, embedding_dim)

    def test_plot_similarity_matrix(self, sample_embeddings):
        """Test similarity matrix plotting."""
        fig = plot_similarity_matrix(sample_embeddings, save=False, show=False)
        assert fig is not None

    def test_plot_vector_components(self, sample_embeddings):
        """Test vector components plotting."""
        fig = plot_vector_components(sample_embeddings, save=False, show=False)
        assert fig is not None

    def test_plot_embedding_space(self, sample_embeddings):
        """Test embedding space plotting (requires sklearn)."""
        try:
            import sklearn
            from fep_rl_vae.visualization.descriptions import plot_embedding_space
            labels = torch.randint(0, 10, (50,))
            fig = plot_embedding_space(sample_embeddings, labels, save=False, show=False)
            assert fig is not None
        except ImportError:
            pytest.skip("scikit-learn not available")


class TestTrainingVisualizations:
    """Test training visualization functions."""

    @pytest.fixture
    def sample_epoch_history(self):
        """Create sample epoch history."""
        return {
            "accuracy_losses": {"image": [0.5, 0.3, 0.1], "number": [0.8, 0.6, 0.2]},
            "complexity_losses": {"image": [0.01, 0.008, 0.005]},
            "total_reward": [10.5, 12.3, 15.1],
            "reward": [10.0, 12.0, 15.0],
            "curiosities": {"obs1": [0.2, 0.3, 0.4]},
            "actor_loss": [0.9, 0.7, 0.5],
            "critic_losses": [[0.8, 0.6, 0.4], [0.7, 0.5, 0.3]],
            "alpha_entropies": {"action1": [0.9, 0.8, 0.7]},
            "alpha_normal_entropies": {"action1": [0.1, 0.2, 0.3]},
            "total_entropies": {"action1": [1.0, 1.0, 1.0]},
        }

    def test_plot_loss_curves(self, sample_epoch_history):
        """Test loss curves plotting."""
        fig = plot_loss_curves(sample_epoch_history, save=False, show=False)
        assert fig is not None

    def test_plot_reward_curves(self, sample_epoch_history):
        """Test reward curves plotting."""
        fig = plot_reward_curves(sample_epoch_history, save=False, show=False)
        assert fig is not None

    def test_plot_entropy_curves(self, sample_epoch_history):
        """Test entropy curves plotting."""
        fig = plot_entropy_curves(sample_epoch_history, save=False, show=False)
        assert fig is not None

    def test_plot_learning_curves(self, sample_epoch_history):
        """Test learning curves plotting."""
        fig = plot_learning_curves(sample_epoch_history, save=False, show=False)
        assert fig is not None

    def test_save_training_visualizations(self, sample_epoch_history):
        """Test saving all training visualizations."""
        saved_files = save_training_visualizations(sample_epoch_history)
        assert len(saved_files) == 4
        assert "losses" in saved_files
        assert "rewards" in saved_files
        assert "entropies" in saved_files
        assert "learning" in saved_files
        
        # Verify files exist
        for filepath in saved_files.values():
            assert Path(filepath).exists()


class TestModelVisualizations:
    """Test model visualization functions."""

    @pytest.fixture
    def sample_feature_maps(self):
        """Create sample feature maps."""
        return torch.randn(16, 28, 28)  # (channels, height, width)

    def test_plot_feature_maps(self, sample_feature_maps):
        """Test feature maps plotting."""
        fig = plot_feature_maps(sample_feature_maps, save=False, show=False)
        assert fig is not None

    def test_plot_latent_space(self):
        """Test latent space plotting."""
        latent_vectors = torch.randn(50, 32)
        labels = torch.randint(0, 10, (50,))
        fig = plot_latent_space(latent_vectors, labels, save=False, show=False)
        assert fig is not None


class TestVisualizationUtilities:
    """Test visualization utility functions."""

    def test_ensure_output_dir(self):
        """Test output directory creation."""
        output_dir = ensure_output_dir("test_subdir")
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_get_output_path(self):
        """Test output path generation."""
        path = get_output_path("test.png", "test_subdir")
        assert "test.png" in str(path)
        assert "test_subdir" in str(path)

    def test_setup_plot_style(self):
        """Test plot style setup."""
        setup_plot_style()
        # Should not raise any errors
        assert True

    def test_save_figure(self):
        """Test figure saving."""
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        path = save_figure(fig, "test_figure.png", "test_subdir")
        assert Path(path).exists()
        Path(path).unlink()  # Cleanup


class TestIntegration:
    """Integration tests for visualization module."""

    def test_complete_visualization_workflow(self):
        """Test complete visualization workflow."""
        # Create sample data
        images = torch.randn(9, 28, 28, 1)
        numbers = torch.randint(0, 10, (100,))
        predictions = torch.rand(10, 10)
        epoch_history = {
            "accuracy_losses": {"image": [0.5, 0.3, 0.1]},
            "total_reward": [10.5, 12.3, 15.1],
            "actor_loss": [0.9, 0.7, 0.5],
            "critic_losses": [[0.8, 0.6, 0.4]],
        }
        
        # Generate visualizations
        fig1 = plot_image_grid(images, save=False, show=False)
        fig2 = plot_number_distribution(numbers, save=False, show=False)
        fig3 = plot_loss_curves(epoch_history, save=False, show=False)
        
        assert fig1 is not None
        assert fig2 is not None
        assert fig3 is not None
