# Installation Guide

Complete installation instructions for FEP-RL-VAE.

## Prerequisites

- Python 3.8 or higher
- [uv](https://github.com/astral-sh/uv) package manager (recommended) or pip

## Quick Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd FEP_RL_VAE

# Create virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package with dev dependencies
uv pip install -e ".[dev]"

# Install general_FEP_RL dependency
./scripts/setup_general_fep_rl.sh
# Or manually:
cd ../active-inference-sim-lab && uv pip install -e . && cd ../FEP_RL_VAE
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd FEP_RL_VAE

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e .

# Install dev dependencies
pip install -r requirements-dev.txt

# Install general_FEP_RL dependency
cd ../active-inference-sim-lab && pip install -e . && cd ../FEP_RL_VAE
```

## Verifying Installation

```bash
# Run tests (should pass 26 tests, skip 23 that require general_FEP_RL)
uv run pytest
# or
pytest

# Test imports
python -c "from fep_rl_vae.encoders import ImageEncoder; print('✓ Imports work')"
```

## Installing general_FEP_RL

The `general_FEP_RL` package is required for model functionality but must be installed separately. It should be located in a sibling directory:

```
GitHub/
├── FEP_RL_VAE/          # This repository
└── active-inference-sim-lab/  # Contains general_FEP_RL
```

### Automatic Installation

```bash
./scripts/setup_general_fep_rl.sh
```

### Manual Installation

```bash
cd ../active-inference-sim-lab
uv pip install -e .  # or: pip install -e .
cd ../FEP_RL_VAE
```

### Alternative: Add to PYTHONPATH

If `general_FEP_RL` is installed elsewhere:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/active-inference-sim-lab/src"
```

## Development Setup

```bash
# Install with all dev dependencies
uv pip install -e ".[dev]"

# Or use make
make install-dev

# Run tests
make test

# Format code
make format

# Lint code
make lint
```

## Troubleshooting

### ModuleNotFoundError: No module named 'general_FEP_RL'

**Solution**: Install general_FEP_RL from the sibling directory:
```bash
cd ../active-inference-sim-lab && uv pip install -e . && cd ../FEP_RL_VAE
```

### Import errors with fep_rl_vae

**Solution**: Ensure the package is installed in editable mode:
```bash
uv pip install -e .
```

### MNIST dataset download issues

**Solution**: The dataset will be downloaded automatically on first use. If issues occur:
- Check internet connection
- Verify disk space
- Try clearing cache: `rm -rf ~/.cache/torchvision`

### Test failures

**Solution**: 
- Ensure all dependencies are installed: `uv pip install -e ".[dev]"`
- Check that general_FEP_RL is installed if running model tests
- Run tests with verbose output: `pytest -vv`

## Next Steps

After installation:

1. **Run examples**: See `examples/` directory
2. **Read documentation**: Check `README.md` and `AGENTS.md`
3. **Explore tests**: Review `tests/` for usage examples
4. **Start developing**: Use `make help` for available commands
