"""Tests for data loading functionality."""

import torch
import pytest
from fep_rl_vae.data.loader import MNISTLoader, get_batch, get_labeled_digits, get_repeating_digit_sequence


class TestMNISTLoader:
    """Test MNISTLoader class."""

    @pytest.fixture
    def loader(self):
        return MNISTLoader()

    def test_instantiation(self, loader):
        """Test loader can be instantiated."""
        assert isinstance(loader, MNISTLoader)
        assert hasattr(loader, 'train_x')
        assert hasattr(loader, 'train_y')
        assert hasattr(loader, 'test_x')
        assert hasattr(loader, 'test_y')

    def test_data_shapes(self, loader):
        """Test data has correct shapes."""
        assert loader.train_x.shape[1:] == (28, 28, 1)  # (N, H, W, C)
        assert loader.train_y.shape[1] == 10  # One-hot encoded
        assert loader.test_x.shape[1:] == (28, 28, 1)
        assert loader.test_y.shape[1] == 10

    def test_data_normalization(self, loader):
        """Test data is properly normalized."""
        assert 0 <= loader.train_x.min() <= loader.train_x.max() <= 1
        assert 0 <= loader.test_x.min() <= loader.test_x.max() <= 1

    def test_get_batch(self, loader):
        """Test batch generation."""
        batch_size = 16
        images, labels = loader.get_batch(batch_size)

        assert images.shape == (batch_size, 28, 28, 1)
        assert labels.shape == (batch_size, 10)
        assert torch.all((images >= 0) & (images <= 1))
        assert torch.all(labels.sum(dim=-1) == 1)  # One-hot property

    def test_get_batch_test_split(self, loader):
        """Test test split batch generation."""
        batch_size = 8
        images, labels = loader.get_batch(batch_size, test=True)

        assert images.shape == (batch_size, 28, 28, 1)
        assert labels.shape == (batch_size, 10)

    def test_get_labeled_digits(self, loader):
        """Test labeled digits generation."""
        digits = loader.get_labeled_digits()

        assert isinstance(digits, dict)
        assert len(digits) == 10  # Digits 0-9
        assert all(isinstance(img, torch.Tensor) for img in digits.values())
        assert all(img.shape == (28, 28, 1) for img in digits.values())

    def test_get_labeled_digits_test_split(self, loader):
        """Test labeled digits from test set."""
        digits = loader.get_labeled_digits(test=True)

        assert isinstance(digits, dict)
        assert len(digits) == 10
        assert all(img.shape == (28, 28, 1) for img in digits.values())

    def test_get_repeating_digit_sequence(self, loader):
        """Test sequence generation."""
        batch_size, steps, n_digits = 4, 8, 3
        sequences, labels = loader.get_repeating_digit_sequence(
            batch_size=batch_size, steps=steps, n_digits=n_digits
        )

        assert sequences.shape == (batch_size, steps, 28, 28, 1)
        assert labels.shape == (batch_size, steps, 10)
        assert torch.all((sequences >= 0) & (sequences <= 1))
        assert torch.all(labels.sum(dim=-1) == 1)  # One-hot property

    def test_sequence_uniqueness(self, loader):
        """Test that sequences contain expected digit patterns."""
        batch_size, steps, n_digits = 2, 6, 3
        sequences, labels = loader.get_repeating_digit_sequence(
            batch_size=batch_size, steps=steps, n_digits=n_digits
        )

        # Check that we only have n_digits unique classes in each sequence
        for b in range(batch_size):
            unique_classes = torch.unique(labels[b].argmax(dim=-1))
            assert len(unique_classes) == n_digits


class TestBackwardCompatibility:
    """Test backward compatibility functions."""

    def test_get_batch_function(self):
        """Test global get_batch function."""
        images, labels = get_batch(batch_size=8)
        assert images.shape == (8, 28, 28, 1)
        assert labels.shape == (8, 10)

    def test_get_labeled_digits_function(self):
        """Test global get_labeled_digits function."""
        digits = get_labeled_digits()
        assert isinstance(digits, dict)
        assert len(digits) == 10

    def test_get_repeating_digit_sequence_function(self):
        """Test global get_repeating_digit_sequence function."""
        sequences, labels = get_repeating_digit_sequence(
            batch_size=2, steps=5, n_digits=3
        )
        assert sequences.shape == (2, 5, 28, 28, 1)
        assert labels.shape == (2, 5, 10)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.fixture
    def loader(self):
        return MNISTLoader()

    def test_sequence_n_digits_edge_cases(self, loader):
        """Test sequence generation with edge case n_digits."""
        # Test minimum n_digits
        sequences, _ = loader.get_repeating_digit_sequence(
            batch_size=1, steps=3, n_digits=1
        )
        assert sequences.shape == (1, 3, 28, 28, 1)

        # Test maximum n_digits
        sequences, _ = loader.get_repeating_digit_sequence(
            batch_size=1, steps=3, n_digits=10
        )
        assert sequences.shape == (1, 3, 28, 28, 1)

    def test_batch_size_edge_cases(self, loader):
        """Test batch generation with various sizes."""
        # Test small batch
        images, labels = loader.get_batch(1)
        assert images.shape == (1, 28, 28, 1)

        # Test larger batch
        images, labels = loader.get_batch(100)
        assert images.shape == (100, 28, 28, 1)
