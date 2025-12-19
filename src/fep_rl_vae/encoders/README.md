# Encoders

Neural encoders for converting different data modalities into latent representations.

## Available Encoders

- **ImageEncoder**: Convolutional encoder for image data
- **NumberEncoder**: Embedding encoder for categorical numbers
- **DescriptionEncoder**: Linear encoder for description vectors

## Usage

```python
from fep_rl_vae.encoders import ImageEncoder, NumberEncoder

# Image encoder
image_encoder = ImageEncoder()
latent = image_encoder(images)  # Shape: (batch, 256)

# Number encoder
number_encoder = NumberEncoder(number_of_digits=10)
latent = number_encoder(numbers)  # Shape: (batch, 16)
```
