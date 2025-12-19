# FEP-RL-VAE Package

Core implementation of the Free Energy Principle - Reinforcement Learning Variational Autoencoder.

## Modules

- **encoders/**: Neural encoders for different data modalities
- **decoders/**: Neural decoders for reconstruction and generation
- **data/**: Dataset loading and preprocessing utilities
- **utils/**: Training utilities for logging and visualization

## Usage

```python
from fep_rl_vae import encoders, decoders, data, utils

# Load data
loader = data.loader.MNISTLoader()
images, labels = loader.get_batch(batch_size=32)

# Create models
encoder = encoders.ImageEncoder()
decoder = decoders.ImageDecoder(hidden_state_size=256)

# Training utilities
history = {}
utils.logging.add_to_epoch_history(history, epoch_data)
utils.plotting.plot_training_history(history)
```
