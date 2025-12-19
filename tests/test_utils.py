"""Tests for utility functions."""

import torch
import pytest
from fep_rl_vae.utils.logging import add_to_epoch_history, print_epoch_summary, print_epoch_dict
from fep_rl_vae.utils.plotting import plot_images


class TestLogging:
    """Test logging utilities."""

    @pytest.fixture
    def sample_epoch_dict(self):
        """Create sample epoch data."""
        return {
            "accuracy_losses": {"image": 0.5, "number": 0.3},
            "complexity_losses": {"image": 0.01},
            "total_reward": 10.5,
            "actor_loss": 0.8,
            "critic_losses": [0.7, 0.6],
            "alpha_entropies": {"action1": 0.9, "action2": 0.8},
            "curiosities": {"obs1": 0.2, "obs2": 0.1}
        }

    def test_add_to_epoch_history_empty(self):
        """Test adding to empty history."""
        history = {}
        epoch_dict = {"loss": 0.5, "reward": 10.0}

        add_to_epoch_history(history, epoch_dict)

        assert "loss" in history
        assert "reward" in history
        assert history["loss"] == [0.5]
        assert history["reward"] == [10.0]

    def test_add_to_epoch_history_accumulate(self):
        """Test accumulating multiple epochs."""
        history = {}
        epoch_dict1 = {"loss": 0.5, "reward": 10.0}
        epoch_dict2 = {"loss": 0.3, "reward": 12.0}

        add_to_epoch_history(history, epoch_dict1)
        add_to_epoch_history(history, epoch_dict2)

        assert history["loss"] == [0.5, 0.3]
        assert history["reward"] == [10.0, 12.0]

    def test_add_to_epoch_history_nested_dict(self):
        """Test handling nested dictionary structures."""
        history = {}
        epoch_dict = {
            "accuracy_losses": {"image": 0.5, "number": 0.3},
            "complexity_losses": {"image": 0.01}
        }

        add_to_epoch_history(history, epoch_dict)

        assert "accuracy_losses" in history
        assert "image" in history["accuracy_losses"]
        assert "number" in history["accuracy_losses"]
        assert history["accuracy_losses"]["image"] == [0.5]
        assert history["accuracy_losses"]["number"] == [0.3]

    def test_add_to_epoch_history_nested_list(self):
        """Test handling nested list structures."""
        history = {}
        epoch_dict = {
            "critic_losses": [0.7, 0.6, 0.5]
        }

        add_to_epoch_history(history, epoch_dict)

        assert "critic_losses" in history
        assert history["critic_losses"] == [[0.7, 0.6, 0.5]]

    def test_add_to_epoch_history_multiple_lists(self):
        """Test accumulating multiple list entries."""
        history = {}
        epoch_dict1 = {"critic_losses": [0.7, 0.6]}
        epoch_dict2 = {"critic_losses": [0.5, 0.4]}

        add_to_epoch_history(history, epoch_dict1)
        add_to_epoch_history(history, epoch_dict2)

        assert history["critic_losses"] == [[0.7, 0.6], [0.5, 0.4]]

    def test_print_epoch_summary(self, sample_epoch_dict, capsys):
        """Test epoch summary printing."""
        history = {}
        add_to_epoch_history(history, sample_epoch_dict)

        print_epoch_summary(history)

        captured = capsys.readouterr()
        assert "accuracy_losses" in captured.out
        assert "dict" in captured.out

    def test_print_epoch_dict(self, sample_epoch_dict, capsys):
        """Test epoch dict printing."""
        print_epoch_dict(sample_epoch_dict)

        captured = capsys.readouterr()
        assert "Epoch data:" in captured.out
        assert "accuracy_losses" in captured.out


class TestPlotting:
    """Test plotting utilities."""

    @pytest.fixture
    def sample_images(self):
        """Create sample image batch for plotting."""
        return [torch.randn(28, 28) for _ in range(4)]

    def test_plot_images_basic(self, sample_images):
        """Test basic image plotting functionality."""
        # Should not raise exceptions
        plot_images(sample_images, "Test Images", show=False)

    def test_plot_images_single_image(self):
        """Test plotting single image."""
        single_image = torch.randn(28, 28)
        plot_images([single_image], "Single Image", show=False)

    def test_plot_images_empty_list(self):
        """Test plotting empty image list."""
        plot_images([], "Empty", show=False)

    def test_plot_images_various_sizes(self):
        """Test plotting various numbers of images."""
        for n_images in [1, 4, 9, 16]:
            images = [torch.randn(28, 28) for _ in range(n_images)]
            plot_images(images, f"{n_images} Images", show=False)


class TestIntegration:
    """Test integration between logging and plotting."""

    def test_complete_workflow(self):
        """Test complete logging and plotting workflow."""
        history = {}

        # Simulate training epochs
        for epoch in range(3):
            epoch_data = {
                "total_reward": float(epoch + 10),
                "actor_loss": 1.0 / (epoch + 1),
                "accuracy_losses": {"image": 0.5 / (epoch + 1)},
                "critic_losses": [0.8 / (epoch + 1), 0.7 / (epoch + 1)]
            }
            add_to_epoch_history(history, epoch_data)

        # Verify history structure
        assert len(history["total_reward"]) == 3
        assert len(history["actor_loss"]) == 3
        assert len(history["accuracy_losses"]["image"]) == 3
        assert len(history["critic_losses"]) == 3

        # Test summary printing (should not crash)
        print_epoch_summary(history)

        # Note: plot_training_history requires matplotlib backend
        # and may not work in headless environments
        # from fep_rl_vae.utils.plotting import plot_training_history
        # plot_training_history(history)  # Commented out for CI compatibility
