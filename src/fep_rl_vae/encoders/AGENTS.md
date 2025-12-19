# AGENTS.md - Encoders Module

## Architecture Details

### ImageEncoder
**Input**: (batch, 1, 28, 28) - Grayscale MNIST images
**Output**: (batch, 256) - Latent representation

**Network Architecture**:
1. Conv2d(1→64, 3x3, padding=1) → PReLU
2. PixelUnshuffle(downscale=2) → Conv2d(256→16, 3x3, padding=1) → PReLU
3. PixelUnshuffle(downscale=2) → Flatten → Linear(1024→256) → PReLU

**Key Features**:
- Spatial downsampling via PixelUnshuffle (modern alternative to MaxPool)
- Progressive feature extraction
- Reflection padding for boundary handling

### NumberEncoder
**Input**: (batch, 1) - Integer indices (0 to number_of_digits-1)
**Output**: (batch, 16) - Latent representation

**Network Architecture**:
1. Embedding(num_embeddings, embedding_dim=16) → PReLU

**Key Features**:
- Learned embeddings for categorical variables
- Configurable vocabulary size
- Efficient parameter usage

### DescriptionEncoder
**Input**: (batch, seq_len, feature_dim) - Sequential description vectors
**Output**: (batch, 64) - Latent representation

**Network Architecture**:
1. Linear(feature_dim→128) → PReLU
2. Linear(128→64) → PReLU

**Key Features**:
- Handles variable-length sequences
- Progressive dimensionality reduction
- Smooth feature transformation

## Design Patterns

### Configuration via arg_dict
```python
# Image encoder - no special args needed
encoder = ImageEncoder(arg_dict={})

# Number encoder - configure vocabulary
encoder = NumberEncoder(arg_dict={"number_of_digits": 10})

# Description encoder - flexible input handling
encoder = DescriptionEncoder(arg_dict={})
```

### Consistent Interface
All encoders follow the same initialization pattern:
- `arg_dict`: Modality-specific configuration
- `verbose`: Enable/disable architecture printing
- Return latent vectors of consistent batch-first format

### Memory Efficiency
- Lazy parameter initialization
- Efficient tensor operations
- Minimal intermediate storage

## Integration with Training

### FEP-RL Agent Configuration
```python
observation_dict = {
    "image_modality": {
        "encoder": ImageEncoder,
        "decoder": ImageDecoder,
        "arg_dict": {},
        "beta": 0.01,  # KL regularization
        "eta": 1.0,    # Entropy weight
    }
}
```

### Gradient Flow
- All operations are differentiable
- Proper initialization for stable training
- Compatible with standard PyTorch optimizers
