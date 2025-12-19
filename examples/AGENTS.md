# AGENTS.md - Examples Module

## Training Paradigms

### Exploratory Training
**Goal**: Learn multimodal representations through reinforcement learning
**Architecture**: Dual observation modalities (vision + language)
**RL Integration**: Agent learns to predict next digits in sequence

#### Key Components
- **Observation Modalities**:
  - `see_number`: Categorical digit observation
  - `see_image`: Visual digit observation
- **Action Modality**:
  - `make_number`: Predicted next digit
- **Reward Structure**: Implicit through reconstruction accuracy

#### Training Dynamics
1. **Episode Start**: Random digit initialization
2. **Step Loop**: Agent observes current digit (both modalities)
3. **Action**: Predict next digit in sequence
4. **State Transition**: Move to predicted digit
5. **Experience Storage**: Buffer current transition
6. **Batch Learning**: Sample from buffer for training

### Basic VAE Training
**Goal**: Learn temporal patterns in digit sequences
**Architecture**: Single visual modality VAE
**RL Integration**: Minimal policy for regularization

#### Key Components
- **Observation Modality**:
  - `see_image`: Visual digit sequences
- **Action Modality**:
  - `make_number`: Dummy action (not used)
- **Sequence Learning**: Predict continuation of digit patterns

#### Training Dynamics
1. **Sequence Generation**: Create repeating digit patterns
2. **Forward Pass**: Encode entire sequence
3. **Reconstruction**: Decode predicted sequence
4. **Loss Computation**: Compare prediction vs ground truth

## Configuration Patterns

### Observation Dictionary
```python
observation_dict = {
    "modality_name": {
        "encoder": EncoderClass,
        "decoder": DecoderClass,
        "arg_dict": {"param": value},  # Modality-specific params
        "accuracy_scalar": 1.0,       # Reconstruction weight
        "complexity_scalar": 0.001,   # KL regularization weight
        "beta": 0.01,                 # Complexity coefficient
        "eta": 1.0,                   # Entropy weight
    }
}
```

### Action Dictionary
```python
action_dict = {
    "action_name": {
        "encoder": EncoderClass,
        "decoder": DecoderClass,
        "arg_dict": {"param": value},
        "target_entropy": -1.0,       # Entropy target
        "alpha_normal": 0.1,          # Entropy temperature
    }
}
```

### Agent Configuration
```python
agent = Agent(
    hidden_state_size=512,      # Latent dimensionality
    observation_dict=observation_dict,
    action_dict=action_dict,
    number_of_critics=1,        # Number of value functions
    tau=0.99,                   # Soft update parameter
    lr=0.003,                   # Learning rate
    weight_decay=0.00001,       # L2 regularization
    gamma=0.99,                 # Discount factor
    capacity=16,                # Replay buffer size
    max_steps=26,               # Maximum episode length
)
```

## Data Generation Strategies

### Labeled Digits
```python
labeled_digits = get_labeled_digits()
# Returns: {0: image0, 1: image1, ..., 9: image9}
```
- One representative image per digit class
- Used for visualization and evaluation

### Repeating Sequences
```python
sequences, labels = get_repeating_digit_sequence(
    batch_size=16, steps=25, n_digits=4
)
# Returns: (batch, steps, 28, 28, 1), (batch, steps, 10)
```
- Temporal patterns with wraparound
- Configurable sequence length and vocabulary

## Training Loop Structure

### Episode Execution
```python
for episode in range(episodes_per_epoch):
    # Initialize episode
    agent.begin()

    # Execute steps
    for step in range(steps):
        obs = create_observation(current_state)
        step_dict = agent.step_in_episode(obs)
        # Process action and reward
        # Store transition

    # Finalize episode
    final_obs = create_observation(final_state)
    step_dict = agent.step_in_episode(final_obs)
```

### Batch Training
```python
epoch_dict = agent.epoch(batch_size=batch_size)
# Returns comprehensive training metrics
```

### Progress Tracking
```python
add_to_epoch_history(epoch_history, epoch_dict)
if epoch % visualization_interval == 0:
    plot_training_history(epoch_history)
```

## Performance Monitoring

### Key Metrics
- **Accuracy Losses**: Reconstruction quality per modality
- **Complexity Losses**: KL divergence regularization
- **Actor Loss**: Policy optimization objective
- **Critic Losses**: Value function accuracy
- **Total Reward**: Cumulative episode rewards

### Visualization
- **Real vs Predicted**: Image sequence comparison
- **Loss Curves**: Training convergence over time
- **Reward Components**: Breakdown of reward sources

## Debugging and Development

### Common Issues
- **Memory Growth**: Monitor replay buffer size
- **Training Instability**: Check gradient norms
- **Mode Collapse**: Verify entropy regularization
- **Poor Reconstruction**: Adjust beta/eta parameters

### Development Workflow
1. **Start Simple**: Use basic_vae_training.py for debugging
2. **Add Complexity**: Progress to exploratory_training.py
3. **Monitor Metrics**: Watch loss curves for convergence
4. **Tune Parameters**: Adjust learning rates and regularization
5. **Scale Up**: Increase batch size and model capacity
