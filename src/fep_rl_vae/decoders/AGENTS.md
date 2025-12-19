# AGENTS.md - Decoders Module

## Architecture Details

### ImageDecoder
**Input**: (batch, hidden_state_size) - Latent representation
**Output**: ((batch, 28, 28, 1), (batch,)) - Reconstruction and log probabilities

**Network Architecture**:
1. Linear(hidden_state_size→16*7*7) → PReLU → Reshape(16, 7, 7)
2. Conv2d(16→64, 3x3, padding=1) → PReLU → Interpolate(scale=2)
3. Conv2d(64→64, 3x3, padding=1) → PReLU → Interpolate(scale=2)
4. Conv2d(64→64, 3x3, padding=1) → PReLU
5. Conv2d(64→1, 1x1) → Sigmoid (for reconstruction) + log probability

**Key Features**:
- Symmetric upsampling to encoder
- Probabilistic output via mu_std
- Configurable entropy for training modes

### NumberDecoder
**Input**: (batch, hidden_state_size) - Latent representation
**Output**: ((batch, num_digits), (batch,)) - Class probabilities and log probabilities

**Network Architecture**:
1. Linear(hidden_state_size→num_digits) → Softmax (for probabilities) + log probability

**Key Features**:
- Categorical distribution output
- Entropy-based training support
- Configurable number of classes

### DescriptionDecoder
**Input**: (batch, hidden_state_size) - Latent representation
**Output**: ((batch, feature_dim), (batch,)) - Reconstruction and log probabilities

**Network Architecture**:
1. Linear(hidden_state_size→128) → PReLU
2. Linear(128→feature_dim) → probabilistic output via mu_std

**Key Features**:
- Flexible output dimensionality
- Smooth reconstruction
- Probabilistic training support

## Loss Functions

### ImageDecoder.loss_func
```python
@staticmethod
def loss_func(true_values, predicted_values):
    return F.binary_cross_entropy(predicted_values, true_values, reduction="none")
```

### NumberDecoder.loss_func
```python
@staticmethod
def loss_func(true_values, predicted_values):
    return F.mse_loss(predicted_values, true_values, reduction="none")
```

### DescriptionDecoder.loss_func
```python
@staticmethod
def loss_func(true_values, predicted_values):
    return F.mse_loss(predicted_values, true_values, reduction="none")
```

## Training Modes

### Deterministic Mode (entropy=False)
- Standard reconstruction training
- Point estimates from decoder
- Lower computational cost

### Entropy Mode (entropy=True)
- Probabilistic training with entropy regularization
- mu_std provides distribution parameters
- Better exploration of latent space

## Integration Patterns

### FEP-RL Agent Configuration
```python
observation_dict = {
    "image_modality": {
        "encoder": ImageEncoder,
        "decoder": ImageDecoder,
        "arg_dict": {"number_of_digits": 10},
        "beta": 0.01,      # KL weight
        "eta": 1.0,        # Entropy weight
        "accuracy_scalar": 1.0,
        "complexity_scalar": 0.001,
    }
}
```

### Standalone Training
```python
decoder = ImageDecoder(hidden_state_size=256, entropy=True)
output, log_prob = decoder(latent)

# Loss computation
loss = decoder.loss_func(target, output)
kl_loss = -log_prob.mean()  # KL regularization
total_loss = loss + beta * kl_loss
```
