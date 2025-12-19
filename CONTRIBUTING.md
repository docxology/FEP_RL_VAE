# Contributing to FEP-RL-VAE

Thank you for your interest in contributing to FEP-RL-VAE! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Fork and clone the repository**
   ```bash
   git clone <your-fork-url>
   cd FEP_RL_VAE
   ```

2. **Set up development environment**
   ```bash
   uv venv
   source .venv/bin/activate
   uv pip install -e ".[dev]"
   ```

3. **Install general_FEP_RL dependency**
   ```bash
   ./scripts/setup_general_fep_rl.sh
   ```

4. **Run tests to verify setup**
   ```bash
   make test
   ```

## Code Style

### Formatting

We use `black` for code formatting and `isort` for import sorting:

```bash
make format
# or
black src/ tests/ examples/
isort src/ tests/ examples/
```

### Linting

We use `flake8` and `mypy` for code quality:

```bash
make lint
# or
flake8 src/ tests/ examples/
mypy src/
```

### Type Hints

Please add type hints to all new functions and methods:

```python
def example_function(x: torch.Tensor, y: int) -> torch.Tensor:
    """Example function with type hints."""
    return x * y
```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest tests/test_data.py -v

# Run specific test
pytest tests/test_data.py::TestMNISTLoader::test_instantiation -v
```

### Writing Tests

- Place tests in `tests/` directory
- Follow naming convention: `test_*.py` for files, `test_*` for functions
- Use pytest fixtures for common setup
- Aim for high test coverage (>80%)

Example:

```python
import pytest
from fep_rl_vae.data.loader import MNISTLoader

def test_loader_instantiation():
    """Test that loader can be instantiated."""
    loader = MNISTLoader()
    assert loader is not None
```

## Documentation

### Code Documentation

- Add docstrings to all public functions, classes, and methods
- Use Google-style docstrings:

```python
def example_function(x: torch.Tensor, y: int) -> torch.Tensor:
    """Brief description.
    
    Longer description if needed.
    
    Args:
        x: Input tensor
        y: Multiplier value
        
    Returns:
        Multiplied tensor
    """
    return x * y
```

### Updating Documentation

- Update `README.md` for user-facing changes
- Update `AGENTS.md` for technical/architectural changes
- Add entries to `CHANGELOG.md` for significant changes

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following style guidelines
   - Add tests for new functionality
   - Update documentation as needed

3. **Run checks**
   ```bash
   make format
   make lint
   make test
   ```

4. **Commit your changes**
   ```bash
   git commit -m "Description of changes"
   ```
   Use clear, descriptive commit messages.

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a pull request on GitHub.

## Pull Request Guidelines

- **Title**: Clear, descriptive title
- **Description**: Explain what changes were made and why
- **Tests**: Ensure all tests pass
- **Documentation**: Update relevant documentation
- **Size**: Keep PRs focused and reasonably sized

## Code Review

- Be respectful and constructive
- Focus on code quality and correctness
- Ask questions if something is unclear
- Respond to feedback promptly

## Reporting Issues

When reporting issues:

1. Check if the issue already exists
2. Use a clear, descriptive title
3. Provide steps to reproduce
4. Include relevant environment information
5. Add code examples if applicable

## Questions?

Feel free to open an issue for questions or discussions about contributions.
