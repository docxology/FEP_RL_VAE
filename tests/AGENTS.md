# AGENTS.md - Tests Module

## Testing Strategy

### Philosophy
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Regression Tests**: Prevent functionality breakage
- **Performance Tests**: Monitor computational efficiency

### Coverage Goals
- **Models**: 90%+ coverage of encoder/decoder logic
- **Data**: 95%+ coverage of loading and preprocessing
- **Utils**: 85%+ coverage of utility functions
- **Integration**: End-to-end training loops

## Test Organization

### test_models.py
**Purpose**: Validate neural network components

**Test Cases**:
- Encoder instantiation with various configurations
- Decoder forward passes with different entropy modes
- Output shape validation
- Loss function correctness
- Gradient flow verification

**Key Fixtures**:
```python
@pytest.fixture
def sample_batch():
    return torch.randn(8, 28, 28, 1)

@pytest.fixture
def image_encoder():
    return ImageEncoder()
```

### test_data.py
**Purpose**: Validate data loading pipeline

**Test Cases**:
- MNIST dataset loading and preprocessing
- Batch generation correctness
- Sequence generation algorithms
- Memory efficiency
- Edge case handling

**Key Fixtures**:
```python
@pytest.fixture
def mnist_loader():
    return MNISTLoader()
```

### test_utils.py
**Purpose**: Validate utility functions

**Test Cases**:
- History accumulation logic
- Plotting function execution
- Data structure transformations
- Error handling

## Test Patterns

### Model Testing
```python
def test_image_encoder_forward(image_encoder, sample_batch):
    latent = image_encoder(sample_batch)
    assert latent.shape == (sample_batch.shape[0], 256)
    assert latent.requires_grad
```

### Data Testing
```python
def test_batch_generation(mnist_loader):
    images, labels = mnist_loader.get_batch(16)
    assert images.shape == (16, 28, 28, 1)
    assert labels.shape == (16, 10)
    assert 0 <= images.min() <= images.max() <= 1
```

### Integration Testing
```python
def test_end_to_end_training():
    # Setup minimal training loop
    # Verify loss decreases over time
    # Check output quality
    pass
```

## Performance Testing

### Benchmarks
- Model forward pass latency
- Data loading throughput
- Memory usage patterns
- GPU utilization

### Profiling
```python
import torch.profiler

def test_model_performance():
    with torch.profiler.profile() as prof:
        # Run model inference
        pass
    print(prof.key_averages().table())
```

## CI/CD Integration

### GitHub Actions
```yaml
- name: Run Tests
  run: |
    pip install -e ".[dev]"
    pytest --cov=fep_rl_vae --cov-report=xml
```

### Quality Gates
- Minimum coverage: 85%
- All tests pass
- No performance regressions
- Code style compliance

## Debugging Tests

### Common Issues
- **Import Errors**: Check package structure and __init__.py
- **CUDA Issues**: Skip GPU tests in CI environments
- **Randomness**: Use fixed seeds for reproducible tests
- **Memory Leaks**: Monitor GPU memory in long-running tests

### Test Debugging
```python
# Add debugging prints
def test_problematic_function():
    result = function_under_test()
    print(f"Debug: result = {result}")  # Temporary debug
    assert condition(result)
```

## Test Maintenance

### When to Update Tests
- API changes in main codebase
- New functionality added
- Bug fixes that change behavior
- Performance improvements

### Test Refactoring
- Keep tests DRY but readable
- Use fixtures for common setup
- Parameterize similar test cases
- Document complex test logic
