# Tests

Comprehensive test suite for FEP-RL-VAE package.

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fep_rl_vae --cov-report=html

# Run specific test file
pytest tests/test_models.py

# Run with verbose output
pytest -v
```

## Test Structure

- **test_models.py**: Encoder and decoder functionality tests
- **test_data.py**: Data loading and preprocessing tests
- **test_utils.py**: Utility function tests

## Test Coverage

Tests cover:
- Model instantiation and forward passes
- Data loading and batching
- Utility functions for logging and plotting
- Integration between components

## Adding Tests

When adding new functionality:

1. Create corresponding test file in `tests/`
2. Test both success and failure cases
3. Include edge cases and parameter validation
4. Ensure tests are fast and isolated
