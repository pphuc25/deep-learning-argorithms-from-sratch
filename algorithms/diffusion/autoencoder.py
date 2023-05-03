import math
from typing import Any, Tuple

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import struct


class AutoEncoder(nn.Module):
    def setup(self, input) -> None:
        self.encoder = Encoder()
        self.latent_conv = LatentConvolution()
        self.decoder = Decoder()

    def __call__(self, x) -> Any:
        posterior = self.encoder(x)
        hidden_states = self.latent_conv(posterior)
        sample = self.decoder(hidden_states)


class Encoder(nn.Module):
    def setup(self) -> None:
        self.dense1 = nn.Conv(features=32, kernel_size=(3, 3), strides=1, padding="SAME")
        self.dense2 = nn.Conv(features=64, kernel_size=(3, 3), strides=1, padding="SAME")

    def __call__(self, x) -> Any:

        x = self.dense1(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))

        x = self.dense2(x)
        x = nn.relu(x)
        encoded = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return encoded
    
class LatentConvolution(nn.Module):
    @nn.compact
    def __call__(self, encoded) -> Any:
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=1, padding="SAME")(encoded)
        latent_conv = nn.relu(x)
        return latent_conv
    
class Decoder(nn.Module):
    def setup(self) -> None:
        self.convTrans1 = nn.ConvTranspose(features=32, kernel_size=(2, 2), strides=(2, 2), padding="SAME")
        self.conv1 = nn.Conv(features=32, kernel_size=(3, 3), strides=1, padding="SAME")
        self.convTrans2 = nn.ConvTranspose(features=16, kernel_size=(2, 2), strides=(2, 2), padding="SAME")
        self.conv2 = nn.Conv(features=1, kernel_size=(3, 3), strides=1, padding="SAME")

    def __call__(self, latent_conv) -> Any:
        x = self.convTrans1(latent_conv)
        x = self.conv1(x)
        x = nn.relu(x)

        x = self.convTrans2(x)
        decoded = self.conv2(x)
        return decoded
