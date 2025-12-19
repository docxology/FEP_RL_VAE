# FEP-RL-VAE

Free Energy Principle - Reinforcement Learning Variational Autoencoder implementation.

## Overview

This package implements a variational autoencoder (VAE) that combines Free Energy Principle (FEP) concepts with reinforcement learning for multimodal learning tasks, particularly focused on digit recognition and generation.

## Features

- **Multimodal Encoders/Decoders**: Support for image and categorical data modalities
- **Reinforcement Learning Integration**: Uses RL agents for training VAE components
- **Comprehensive Visualization**: Full visualization module for all data formats (images, numbers, descriptions, training, models)
- **Modular Architecture**: Clean separation of encoders, decoders, data loading, utilities, and visualization
- **Comprehensive Testing**: Full pytest suite for reliability (74 tests: 50 passing, 24 skipped)
- **Modern Python Packaging**: Uses uv for dependency management
- **Organized Outputs**: All visualizations automatically saved to `output/` directory

## Installation

See [INSTALL.md](INSTALL.md) for detailed installation instructions.

### Quick Start

```bash
# Create virtual environment and install package
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Install general_FEP_RL dependency
./scripts/setup_general_fep_rl.sh

# Or use make
make install-dev
```

**Note**: The `general_FEP_RL` package must be installed from the sibling `active-inference-sim-lab` directory. See [INSTALL.md](INSTALL.md) for details.

## Quick Start

```python
from fep_rl_vae.encoders import ImageEncoder
from fep_rl_vae.decoders import ImageDecoder
from fep_rl_vae.data.loader import MNISTLoader

# Load data
loader = MNISTLoader()
images, labels = loader.get_batch(batch_size=32)

# Create models
encoder = ImageEncoder()
decoder = ImageDecoder(hidden_state_size=256)

# Use in your training loop
encoded = encoder(images)
reconstructed, log_prob = decoder(encoded)
```

## Examples

See the `examples/` directory for complete training scripts:

- `exploratory_training.py`: Full FEP-RL-VAE training with multimodal observations
- `basic_vae_training.py`: Basic VAE training on digit sequences

## Project Structure

```
FEP_RL_VAE/
├── src/fep_rl_vae/      # Main package
│   ├── encoders/        # Data encoders for different modalities
│   ├── decoders/        # Data decoders for different modalities
│   ├── data/           # Data loading utilities
│   ├── utils/          # Logging and plotting utilities
│   └── visualization/  # Comprehensive visualization module
├── examples/            # Example training scripts
├── tests/              # Test suite (74 tests)
├── scripts/            # Utility scripts
├── output/             # Generated visualizations (organized by type)
└── docs/               # Additional documentation
```

## Visualization

The package includes a comprehensive visualization module for all data formats:

```python
from fep_rl_vae.visualization import (
    plot_image_grid,
    plot_number_distribution,
    save_training_visualizations,
)

# Generate visualizations
images = torch.randn(9, 28, 28, 1)
plot_image_grid(images, title="Sample Images", save=True)

# All outputs automatically saved to output/ directory
```

**Visualization Features**:
- **Images**: Grids, sequences, comparisons, distributions
- **Numbers**: Distributions, predictions, confusion matrices, sequences
- **Descriptions**: Embedding spaces, similarity matrices, vector components
- **Training**: Loss curves, rewards, entropies, learning curves
- **Models**: Feature maps, latent spaces, architectures

See `examples/visualization_demo.py` for a complete demonstration.

## Quick Reference

### Common Commands

```bash
# Setup
make install-dev              # Install with dev dependencies
./scripts/setup_general_fep_rl.sh  # Install general_FEP_RL

# Development
make test                     # Run tests
make format                   # Format code
make lint                    # Lint code
make clean                   # Clean build artifacts

# Validation
python scripts/validate_setup.py  # Verify installation
```

### Documentation

- **[INSTALL.md](INSTALL.md)** - Detailed installation guide
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines
- **[CHANGELOG.md](CHANGELOG.md)** - Version history
- **[AGENTS.md](AGENTS.md)** - Technical documentation
- **[PROJECT_STATUS.md](PROJECT_STATUS.md)** - Current project status

## Development

### Setup

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install package with dev dependencies
uv pip install -e ".[dev]"

# Install general_FEP_RL dependency (required)
cd ../active-inference-sim-lab && uv pip install -e . && cd ../FEP_RL_VAE
```

### Testing

```bash
# Run all tests
uv run pytest
# or
make test

# Run with coverage
uv run pytest --cov=fep_rl_vae --cov-report=html
# or
make test-cov

# Run specific test file
uv run pytest tests/test_data.py -v
```

### Code Quality

```bash
# Format code
make format
# or
uv run black src/ tests/ examples/
uv run isort src/ tests/ examples/

# Lint code
make lint
# or
uv run flake8 src/ tests/ examples/
uv run mypy src/
```

### Common Tasks

```bash
# Clean build artifacts
make clean

# Reinstall package
make install-dev
```

## Dependencies

- torch: Deep learning framework
- torchvision: MNIST dataset loading
- numpy: Numerical computing
- matplotlib: Plotting and visualization
- general_FEP_RL: FEP-RL agent framework (path dependency)

### Optional Dependencies

- scikit-learn: For embedding space visualization (install with `pip install scikit-learn` or `uv pip install ".[viz]"`)

## License

MIT License
