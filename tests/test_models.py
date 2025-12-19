"""Tests for encoder and decoder models."""

import pytest

try:
    import general_FEP_RL
    GENERAL_FEP_RL_AVAILABLE = True
except ImportError:
    GENERAL_FEP_RL_AVAILABLE = False

if GENERAL_FEP_RL_AVAILABLE:
    import torch
    from fep_rl_vae.encoders import ImageEncoder, NumberEncoder, DescriptionEncoder
    from fep_rl_vae.decoders import ImageDecoder, NumberDecoder, DescriptionDecoder

pytestmark = pytest.mark.skipif(
    not GENERAL_FEP_RL_AVAILABLE,
    reason="general_FEP_RL package not available"
)


class TestImageEncoder:
    """Test ImageEncoder functionality."""

    @pytest.fixture
    def encoder(self):
        return ImageEncoder()

    @pytest.fixture
    def sample_batch(self):
        """Create sample image batch."""
        return torch.randn(4, 1, 28, 28)

    def test_instantiation(self, encoder):
        """Test encoder can be instantiated."""
        assert isinstance(encoder, ImageEncoder)
        assert hasattr(encoder, 'out_features')
        assert encoder.out_features == 256

    def test_forward_pass(self, encoder, sample_batch):
        """Test forward pass produces correct output shape."""
        output = encoder(sample_batch)
        assert output.shape == (4, 256)
        assert output.requires_grad

    def test_example_output_shape(self, encoder):
        """Test example output matches expected shape."""
        assert encoder.example_output.shape == (32, 16, 256)


class TestNumberEncoder:
    """Test NumberEncoder functionality."""

    @pytest.fixture
    def encoder(self):
        return NumberEncoder(arg_dict={"number_of_digits": 10})

    @pytest.fixture
    def sample_batch(self):
        """Create sample number batch (one-hot encoded)."""
        return torch.eye(10)[torch.randint(0, 10, (4,))].unsqueeze(1)

    def test_instantiation(self, encoder):
        """Test encoder can be instantiated."""
        assert isinstance(encoder, NumberEncoder)
        assert hasattr(encoder, 'out_features')
        assert encoder.out_features == 16

    def test_forward_pass(self, encoder, sample_batch):
        """Test forward pass produces correct output shape."""
        output = encoder(sample_batch)
        assert output.shape == (4, 16)
        assert output.requires_grad

    def test_example_output_shape(self, encoder):
        """Test example output matches expected shape."""
        assert encoder.example_output.shape == (32, 16, 16)


class TestDescriptionEncoder:
    """Test DescriptionEncoder functionality."""

    @pytest.fixture
    def encoder(self):
        return DescriptionEncoder()

    @pytest.fixture
    def sample_batch(self):
        """Create sample description batch."""
        return torch.randn(4, 16, 128)

    def test_instantiation(self, encoder):
        """Test encoder can be instantiated."""
        assert isinstance(encoder, DescriptionEncoder)
        assert hasattr(encoder, 'out_features')
        assert encoder.out_features == 64

    def test_forward_pass(self, encoder, sample_batch):
        """Test forward pass produces correct output shape."""
        output = encoder(sample_batch)
        assert output.shape == (4, 64)
        assert output.requires_grad


class TestImageDecoder:
    """Test ImageDecoder functionality."""

    @pytest.fixture
    def decoder_deterministic(self):
        return ImageDecoder(hidden_state_size=256, entropy=False)

    @pytest.fixture
    def decoder_entropy(self):
        return ImageDecoder(hidden_state_size=256, entropy=True)

    @pytest.fixture
    def sample_latent(self):
        """Create sample latent vectors."""
        return torch.randn(4, 256)

    def test_instantiation_deterministic(self, decoder_deterministic):
        """Test deterministic decoder instantiation."""
        assert isinstance(decoder_deterministic, ImageDecoder)

    def test_instantiation_entropy(self, decoder_entropy):
        """Test entropy decoder instantiation."""
        assert isinstance(decoder_entropy, ImageDecoder)

    def test_forward_pass_deterministic(self, decoder_deterministic, sample_latent):
        """Test deterministic forward pass."""
        output, log_prob = decoder_deterministic(sample_latent)
        assert output.shape == (4, 28, 28, 1)
        assert log_prob.shape == (4,)
        assert torch.all((output >= 0) & (output <= 1))  # Sigmoid output

    def test_forward_pass_entropy(self, decoder_entropy, sample_latent):
        """Test entropy forward pass."""
        output, log_prob = decoder_entropy(sample_latent)
        assert output.shape == (4, 28, 28, 1)
        assert log_prob.shape == (4,)

    def test_loss_function(self):
        """Test loss function works correctly."""
        true = torch.randn(4, 28, 28, 1)
        pred = torch.sigmoid(torch.randn(4, 28, 28, 1))
        loss = ImageDecoder.loss_func(true, pred)
        assert loss.shape == (4, 28, 28, 1)
        assert loss.requires_grad


class TestNumberDecoder:
    """Test NumberDecoder functionality."""

    @pytest.fixture
    def decoder_deterministic(self):
        return NumberDecoder(hidden_state_size=128, entropy=False, arg_dict={"number_of_digits": 10})

    @pytest.fixture
    def decoder_entropy(self):
        return NumberDecoder(hidden_state_size=128, entropy=True, arg_dict={"number_of_digits": 10})

    @pytest.fixture
    def sample_latent(self):
        """Create sample latent vectors."""
        return torch.randn(4, 128)

    def test_instantiation_deterministic(self, decoder_deterministic):
        """Test deterministic decoder instantiation."""
        assert isinstance(decoder_deterministic, NumberDecoder)

    def test_instantiation_entropy(self, decoder_entropy):
        """Test entropy decoder instantiation."""
        assert isinstance(decoder_entropy, NumberDecoder)

    def test_forward_pass_deterministic(self, decoder_deterministic, sample_latent):
        """Test deterministic forward pass."""
        output, log_prob = decoder_deterministic(sample_latent)
        assert output.shape == (4, 10)
        assert log_prob.shape == (4,)
        # Check softmax properties
        assert torch.allclose(output.sum(dim=-1), torch.ones(4))
        assert torch.all((output >= 0) & (output <= 1))

    def test_forward_pass_entropy(self, decoder_entropy, sample_latent):
        """Test entropy forward pass."""
        output, log_prob = decoder_entropy(sample_latent)
        assert output.shape == (4, 10)
        assert log_prob.shape == (4,)

    def test_loss_function(self):
        """Test loss function works correctly."""
        true = torch.randn(4, 10)
        pred = torch.softmax(torch.randn(4, 10), dim=-1)
        loss = NumberDecoder.loss_func(true, pred)
        assert loss.shape == (4, 10)
        assert loss.requires_grad


class TestDescriptionDecoder:
    """Test DescriptionDecoder functionality."""

    @pytest.fixture
    def decoder_deterministic(self):
        return DescriptionDecoder(hidden_state_size=128, entropy=False)

    @pytest.fixture
    def decoder_entropy(self):
        return DescriptionDecoder(hidden_state_size=128, entropy=True)

    @pytest.fixture
    def sample_latent(self):
        """Create sample latent vectors."""
        return torch.randn(4, 128)

    def test_instantiation_deterministic(self, decoder_deterministic):
        """Test deterministic decoder instantiation."""
        assert isinstance(decoder_deterministic, DescriptionDecoder)

    def test_instantiation_entropy(self, decoder_entropy):
        """Test entropy decoder instantiation."""
        assert isinstance(decoder_entropy, DescriptionDecoder)

    def test_forward_pass_deterministic(self, decoder_deterministic, sample_latent):
        """Test deterministic forward pass."""
        output, log_prob = decoder_deterministic(sample_latent)
        assert output.shape == (4, 128)
        assert log_prob.shape == (4,)

    def test_forward_pass_entropy(self, decoder_entropy, sample_latent):
        """Test entropy forward pass."""
        output, log_prob = decoder_entropy(sample_latent)
        assert output.shape == (4, 128)
        assert log_prob.shape == (4,)

    def test_loss_function(self):
        """Test loss function works correctly."""
        true = torch.randn(4, 128)
        pred = torch.randn(4, 128)
        loss = DescriptionDecoder.loss_func(true, pred)
        assert loss.shape == (4, 128)
        assert loss.requires_grad
