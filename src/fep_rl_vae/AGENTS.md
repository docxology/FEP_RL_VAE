# AGENTS.md - FEP-RL-VAE Core Package

## Package Structure

### Design Philosophy
- **Modularity**: Each component can be used independently
- **Composition**: Models are composed from encoders/decoders
- **Configuration**: Dict-based parameter passing for flexibility
- **Extensibility**: Easy to add new modalities and training methods

### Component Interfaces

#### Encoder Interface
```python
class Encoder(nn.Module):
    def __init__(self, arg_dict=None, verbose=False):
        # Configure model architecture
        pass

    def forward(self, x):
        # Return latent representation
        return latent
```

#### Decoder Interface
```python
class Decoder(nn.Module):
    def __init__(self, hidden_state_size, encoded_action_size=0, entropy=False, arg_dict=None, verbose=False):
        # Configure model architecture
        pass

    def forward(self, hidden_state):
        # Return reconstruction and log probabilities
        return output, log_prob

    @staticmethod
    def loss_func(true_values, predicted_values):
        # Return loss tensor
        return loss
```

#### Data Loader Interface
```python
class DataLoader:
    def __init__(self):
        # Initialize dataset
        pass

    def get_batch(self, batch_size, test=False):
        # Return batch of data
        return x, y

    def get_sequences(self, batch_size, steps, **kwargs):
        # Return sequential data
        return sequences
```

## Integration Patterns

### With General FEP-RL Agent
```python
from general_FEP_RL.agent import Agent
from fep_rl_vae import encoders, decoders

observation_dict = {
    "modality_name": {
        "encoder": encoders.SomeEncoder,
        "decoder": decoders.SomeDecoder,
        "arg_dict": {...},
        # Training parameters
    }
}

agent = Agent(
    hidden_state_size=512,
    observation_dict=observation_dict,
    action_dict=action_dict,
    # ... other params
)
```

### Standalone VAE Training
```python
import torch.optim as optim
from fep_rl_vae import encoders, decoders, data

# Setup
encoder = encoders.ImageEncoder()
decoder = decoders.ImageDecoder(hidden_state_size=256)
loader = data.loader.MNISTLoader()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()))

# Training loop
for batch_x, _ in loader:
    latent = encoder(batch_x)
    recon, log_prob = decoder(latent)
    loss = decoder.loss_func(batch_x, recon)
    # ... backprop
```

## Performance Characteristics

### Memory Usage
- Models use efficient PyTorch operations
- Batch processing for GPU utilization
- Lazy loading prevents memory bloat

### Training Stability
- Xavier initialization in all layers
- Proper gradient flow through encoder-decoder
- Configurable regularization parameters

### Scalability
- Linear scaling with batch size
- Parallel processing on multiple GPUs
- Streaming data loading for large datasets
