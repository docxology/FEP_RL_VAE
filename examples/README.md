# Examples

Complete training scripts demonstrating FEP-RL-VAE usage.

## Available Examples

### exploratory_training.py
Full FEP-RL-VAE training with multimodal observations (images + numbers).
- **Features**: Dual modality learning, reinforcement learning integration
- **Duration**: ~2000 epochs, ~30 minutes on GPU
- **Output**: Trained agent that can predict digit sequences

### basic_vae_training.py
Basic VAE training focused on digit sequence prediction.
- **Features**: Single modality VAE, temporal sequence learning
- **Duration**: ~2000 epochs, ~15 minutes on GPU
- **Output**: VAE that reconstructs digit sequences

## Running Examples

```bash
# Install dependencies
pip install -e .

# Run exploratory training
python examples/exploratory_training.py

# Run basic VAE training
python examples/basic_vae_training.py
```

## Configuration

Both examples can be modified by changing:
- `number_of_digits`: Vocabulary size (default: 4)
- `epochs`: Training duration
- `batch_size`: Batch size for training
- `steps`: Episode length

## Expected Output

- **Training Progress**: Real-time epoch numbers and episode progress
- **Visualization**: Automatic plotting of losses, rewards, and reconstructions
- **Saved Images**: Generated samples in `images/` directory

## Understanding the Code

### Training Loop Structure
1. **Data Generation**: Create digit sequences or labeled examples
2. **Episode Execution**: Agent interacts with environment for N steps
3. **Experience Collection**: Store transitions in replay buffer
4. **Batch Training**: Sample and train on accumulated experience
5. **Visualization**: Periodic plotting of training progress

### Key Components
- **Agent**: FEP-RL agent managing encoder/decoder networks
- **Buffer**: Experience replay for stable training
- **Observation Dict**: Defines what modalities the agent observes
- **Action Dict**: Defines what actions the agent can take
