# AGENTS.md - Utils Module

## Logging System

### Data Structures
**epoch_history**: Nested dictionary tracking training metrics over time
```
{
    "accuracy_losses": {"image": [0.5, 0.3, 0.1], "number": [0.8, 0.6, 0.2]},
    "complexity_losses": {"image": [0.01, 0.008, 0.005]},
    "total_reward": [10.5, 12.3, 15.1],
    "actor_loss": [0.9, 0.7, 0.5],
    "critic_losses": [[0.8, 0.6], [0.7, 0.5], [0.6, 0.4]]
}
```

### add_to_epoch_history(epoch_history, epoch_dict)
**Purpose**: Accumulate training metrics across epochs
**Algorithm**:
- Float values: Append to list
- List values: Extend nested lists (for multi-critic setups)
- Dict values: Recursively merge nested dictionaries

### print_epoch_summary(epoch_history)
**Purpose**: Display current training state
**Algorithm**:
- Traverse nested structure
- Show data types and lengths
- Provide overview without full data dumps

### print_epoch_dict(epoch_dict)
**Purpose**: Debug current epoch data
**Algorithm**:
- Pretty-print epoch_dict contents
- Show structure and sample values
- Useful for development debugging

## Plotting System

### plot_images(images, title, show=True, name="", folder="")
**Purpose**: Visualize image grids
**Algorithm**:
1. Calculate optimal grid layout (sqrt approximation)
2. Create matplotlib figure with subplots
3. Display images with grayscale colormap
4. Optional saving to disk

### plot_training_history(epoch_history)
**Purpose**: Comprehensive training visualization
**Plots Generated**:
1. **Loss Curves**: Accuracy vs Complexity losses over time
2. **Reward Curves**: Total reward and curiosity components
3. **Actor Loss**: Policy loss and entropy terms
4. **Critic Losses**: Value function losses (multiple critics)

**Features**:
- Automatic subplot layout
- Legend management
- Grid lines for readability
- Consistent color schemes

## Integration Patterns

### Training Loop Integration
```python
from fep_rl_vae.utils import add_to_epoch_history, plot_training_history

epoch_history = {}

for epoch in range(num_epochs):
    # Training logic...
    epoch_data = agent.epoch(batch_size=32)

    # Track progress
    add_to_epoch_history(epoch_history, epoch_data)

    # Periodic visualization
    if epoch % 10 == 0:
        plot_training_history(epoch_history)
```

### Custom Metrics
```python
# Add custom metric tracking
epoch_dict["custom_metric"] = 0.95
add_to_epoch_history(epoch_history, epoch_dict)
```

## Performance Considerations

### Memory Efficiency
- History storage scales linearly with epochs
- No redundant data structures
- Efficient nested dictionary operations

### I/O Optimization
- Plotting uses vector graphics for scalability
- Optional disk saving prevents memory accumulation
- Lazy figure creation and cleanup

### Thread Safety
- Pure functions with no shared state
- Safe for parallel training loops
- Matplotlib handles thread isolation
