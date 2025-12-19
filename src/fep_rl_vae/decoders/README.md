# Decoders

Neural decoders for reconstructing data from latent representations.

## Available Decoders

- **ImageDecoder**: Convolutional decoder for image reconstruction
- **NumberDecoder**: Categorical decoder for number prediction
- **DescriptionDecoder**: Linear decoder for description reconstruction

## Usage

```python
from fep_rl_vae.decoders import ImageDecoder, NumberDecoder

# Image decoder
image_decoder = ImageDecoder(hidden_state_size=256)
reconstruction, log_prob = image_decoder(latent)

# Number decoder
number_decoder = NumberDecoder(hidden_state_size=128, number_of_digits=10)
prediction, log_prob = number_decoder(latent)
```
