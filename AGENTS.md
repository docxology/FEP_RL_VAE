# AGENTS.md - FEP-RL-VAE Technical Documentation

## System Architecture

This repository implements a Free Energy Principle (FEP) inspired Variational Autoencoder (VAE) with reinforcement learning components for multimodal learning.

### Core Components

#### Encoders (`src/fep_rl_vae/encoders/`)
- **ImageEncoder**: Convolutional encoder for MNIST images (28x28x1)
  - Architecture: Conv2d → PReLU → PixelUnshuffle → Conv2d → PReLU → PixelUnshuffle → Linear
  - Output: 256-dimensional latent space
  - Input shape: (batch, channels, height, width)

- **NumberEncoder**: Embedding-based encoder for categorical digits
  - Architecture: Embedding → PReLU → Linear
  - Output: 16-dimensional latent space
  - Configurable vocabulary size via `number_of_digits`

- **DescriptionEncoder**: Linear encoder for description vectors
  - Architecture: Linear → PReLU → Linear
  - Output: 64-dimensional latent space
  - Input shape: (batch, sequence_length, feature_dim)

#### Decoders (`src/fep_rl_vae/decoders/`)
- **ImageDecoder**: Convolutional decoder with upsampling
  - Architecture: Linear → Reshape → Conv2d → Interpolate → Conv2d → Interpolate → Conv2d
  - Uses mu_std for probabilistic output (when entropy=True)
  - Output: Reconstructed images with log probabilities

- **NumberDecoder**: Categorical decoder
  - Architecture: Linear → Softmax
  - Uses mu_std for entropy-based training
  - Output: Probability distribution over digit classes

- **DescriptionDecoder**: Linear decoder for descriptions
  - Architecture: Linear → PReLU → Linear
  - Uses mu_std for probabilistic output
  - Output: Reconstructed description vectors

#### Data Loading (`src/fep_rl_vae/data/`)
- **MNISTLoader**: Class-based MNIST dataset manager
  - Pre-loads and normalizes MNIST data
  - Provides batch sampling and sequence generation
  - Supports random digit sequences with configurable patterns

#### Utilities (`src/fep_rl_vae/utils/`)
- **logging.py**: Training history tracking
  - `add_to_epoch_history()`: Accumulates training metrics
  - `print_epoch_summary()`: Displays training progress
  - `print_epoch_dict()`: Debug current epoch data

- **plotting.py**: Visualization utilities
  - `plot_images()`: Grid visualization of image batches
  - `plot_training_history()`: Comprehensive loss/reward plotting

## Training Paradigms

### Exploratory Training (`examples/exploratory_training.py`)
- **Dual Modality**: Simultaneous image and number observation
- **RL Integration**: Agent learns to predict digit sequences
- **Multi-step Episodes**: 25-step sequences with reward accumulation
- **Loss Components**: Accuracy loss, complexity loss, actor loss, critic loss

### Basic VAE Training (`examples/basic_vae_training.py`)
- **Single Modality**: Image-only VAE training
- **Sequence Prediction**: Learns temporal patterns in digit sequences
- **Simplified RL**: Minimal actor-critic for VAE regularization

## Key Hyperparameters

### Model Configuration
- `hidden_state_size`: Core latent dimensionality (default: 512)
- `number_of_critics`: Number of critic networks (default: 1)
- `tau`: Soft update parameter (default: 0.99)
- `lr`: Learning rate (default: 0.003)
- `gamma`: Discount factor (default: 0.99)

### Modality-Specific Parameters
- **Images**: beta=0.01, eta=1.0, accuracy_scalar=1.0
- **Numbers**: beta=0.01, eta=0.0, accuracy_scalar=0.1
- **Actions**: target_entropy=-1, alpha_normal=0.1

## Data Flow

1. **Observation Processing**:
   - Raw data → Encoder → Latent representation
   - Latent → Decoder → Reconstruction + log probabilities

2. **Loss Computation**:
   - Reconstruction loss: Binary cross-entropy for images, MSE for numbers
   - Complexity loss: KL divergence regularization
   - RL losses: Actor-critic for policy optimization

3. **Training Loop**:
   - Sample batch from replay buffer
   - Forward pass through encoder-decoder
   - Compute losses and backpropagate
   - Update target networks (soft update)

## Dependencies and Integration

- **general_FEP_RL**: Core RL agent framework
  - Provides Agent class with step/batch training methods
  - Handles replay buffer and network updates
  - Path dependency: `../active-inference-sim-lab/src`

- **torch**: Neural network operations
  - Custom utilities in `general_FEP_RL.utils_torch`
  - `model_start/model_end`: Shape handling for batch processing

## Testing Strategy

### Unit Tests (`tests/`)
- **test_models.py**: Encoder/decoder functionality
- **test_data.py**: Data loading and preprocessing
- **test_utils.py**: Utility function correctness

### Integration Tests
- End-to-end training loops
- Loss convergence verification
- Gradient flow validation

## Development Guidelines

### Code Style
- **Naming**: PascalCase for classes, snake_case for functions/variables
- **Imports**: Absolute imports within package, relative for external
- **Documentation**: Docstrings for all public functions
- **Type Hints**: Where beneficial for clarity

### Architecture Patterns
- **Composition over Inheritance**: Modular component design
- **Configuration Objects**: Dict-based parameter passing
- **Factory Pattern**: Encoder/decoder instantiation
- **Observer Pattern**: Training history accumulation

### Performance Considerations
- **Batching**: All operations support batch processing
- **GPU Compatibility**: Full PyTorch GPU support
- **Memory Efficiency**: Lazy loading and streaming where possible
- **Numerical Stability**: Proper initialization and normalization

## Extension Points

### Adding New Modalities
1. Create encoder/decoder pair in respective directories
2. Add to `__init__.py` exports
3. Update observation/action dictionaries in examples
4. Add modality-specific hyperparameters

### Custom Training Loops
1. Import required components
2. Configure observation/action dictionaries
3. Initialize Agent with appropriate parameters
4. Implement custom episode logic

### Integration with Other Frameworks
- Modular design allows easy integration
- Standard PyTorch interface for maximum compatibility
- Configuration-driven setup for flexibility
