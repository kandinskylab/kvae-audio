import math
from typing import List, Union

import numpy as np
import torch
from audiotools.ml import BaseModel
from torch import nn

from kvae_audio.nn.layers import Snake1d, WNConv1d, WNConvTranspose1d

from .base import CodecMixin


def init_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight, std=0.02)
        nn.init.constant_(m.bias, 0)


class ResidualUnit(nn.Module):
    def __init__(self, dim: int = 16, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Snake1d(dim),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x):
        y = self.block(x)
        pad = (x.shape[-1] - y.shape[-1]) // 2
        if pad > 0:
            x = x[..., pad:-pad]
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, dim: int = 16, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(dim // 2, dilation=1),
            ResidualUnit(dim // 2, dilation=3),
            ResidualUnit(dim // 2, dilation=9),
            Snake1d(dim // 2),
            WNConv1d(
                dim // 2,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        strides: list = [2, 3, 4, 5, 8],
        d_latent: int = 64,
        d_in: int = 1,
    ):
        super().__init__()
        layers = [WNConv1d(d_in, d_model, kernel_size=7, padding=3)]

        for stride in strides:
            d_model *= 2
            layers += [EncoderBlock(d_model, stride=stride)]

        layers += [
            Snake1d(d_model),
            WNConv1d(d_model, d_latent, kernel_size=3, padding=1),
        ]

        self.model = nn.Sequential(*layers)
        self.enc_dim = d_model

    def forward(self, x):
        return self.model(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_dim: int = 16, output_dim: int = 8, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(input_dim),
            # odd strides need output_padding=1 to invert the encoder's downsampling exactly, or decode() silently truncates
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
                output_padding=stride % 2,
            ),
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        )

    def forward(self, x):
        return self.block(x)


class Decoder(nn.Module):
    def __init__(
        self,
        d_latent: int = 64,
        d_model: int = 1536,
        strides: list = [8, 5, 4, 3, 2],
        d_out: int = 1,
    ):
        super().__init__()
        layers = [WNConv1d(d_latent, d_model, kernel_size=7, padding=3)]

        for i, stride in enumerate(strides):
            input_dim = d_model // 2**i
            output_dim = d_model // 2 ** (i + 1)
            layers += [DecoderBlock(input_dim, output_dim, stride)]

        layers += [
            Snake1d(output_dim),
            WNConv1d(output_dim, d_out, kernel_size=7, padding=3),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class KVAEAudio(BaseModel, CodecMixin):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 3, 4, 5, 8],
        latent_dim: int = 2048,
        codebook_dim: Union[int, list] = 64,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 5, 4, 3, 2],
        sample_rate: int = 48000,
        num_channels: int = 1,
        **kwargs,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate
        self.num_channels = num_channels

        self.latent_dim = latent_dim
        if codebook_dim is None:
            codebook_dim = latent_dim
        self.codebook_dim = codebook_dim

        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(
            d_model=encoder_dim,
            strides=encoder_rates,
            d_latent=latent_dim,
            d_in=num_channels,
        )

        self.in_proj = WNConv1d(latent_dim, codebook_dim * 2, kernel_size=1)
        self.out_proj = WNConv1d(codebook_dim, latent_dim, kernel_size=1)

        self.decoder = Decoder(
            d_latent=latent_dim,
            d_model=decoder_dim,
            strides=decoder_rates,
            d_out=num_channels,
        )
        self.apply(init_weights)

        self.delay = self.get_delay()

        self.attn = nn.MultiheadAttention(latent_dim, 8, batch_first=True)

    def preprocess(self, audio_data):
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        return nn.functional.pad(audio_data, (0, right_pad))

    def encode(
        self,
        audio_data: torch.Tensor,
        sample: bool = False,
    ):
        audio_data = self.preprocess(audio_data)
        q = self.encoder(audio_data)
        q = q.transpose(1, 2)
        q, _ = self.attn(q, q, q)
        q = q.transpose(1, 2)

        h = self.in_proj(q)
        mu, logvar = torch.chunk(h, 2, dim=1)

        if sample:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        kl_per_sample = 0.5 * torch.sum(
            mu.pow(2) + logvar.exp() - 1.0 - logvar, dim=[1, 2]
        )
        kl_loss = kl_per_sample.mean()

        return z, mu, logvar, kl_loss

    def decode(self, z: torch.Tensor):
        z = self.out_proj(z)
        return self.decoder(z)

    def forward(
        self,
        audio_data: torch.Tensor,
        sample: bool = False,
    ):
        length = audio_data.shape[-1]

        z, mu, logvar, kl_loss = self.encode(audio_data, sample=sample)
        x = self.decode(z)
        return {
            "audio": x[..., :length],
            "z": z,
            "mu": mu,
            "logvar": logvar,
            "kl/loss": kl_loss,
        }
