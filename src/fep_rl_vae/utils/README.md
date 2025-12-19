# Utils Module

Training utilities for logging, plotting, and data management.

## Submodules

- **logging.py**: Training history tracking and reporting
- **plotting.py**: Visualization utilities for training progress

## Usage

```python
from fep_rl_vae.utils import add_to_epoch_history, plot_training_history

# Track training progress
epoch_history = {}
add_to_epoch_history(epoch_history, epoch_data)

# Visualize results
plot_training_history(epoch_history)
```
