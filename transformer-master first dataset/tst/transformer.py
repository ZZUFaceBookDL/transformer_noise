import torch
import torch.nn as nn

from tst.encoder import Encoder
from tst.decoder import Decoder
from tst.utils import generate_original_PE, generate_regular_PE
import torch.nn.functional as F


import math


class Transformer(nn.Module):
    def __init__(self,
                 d_input: int,
                 d_model: int,
                 d_channel: int,
                 d_output: int,
                 q: int,
                 v: int,
                 h: int,
                 N: int,
                 dropout: float = 0.3,
                 mask: bool = False,
                 pe: bool = False,
                 noise: float = 0):
        """Create transformer structure from Encoder and Decoder blocks."""
        super().__init__()

        self._d_input = d_input
        self._d_model = d_model
        self._d_channel = d_channel
        self.pe = pe
        self._noise = noise

        self.layers_encoding = nn.ModuleList([Encoder(d_model,
                                                      q,
                                                      v,
                                                      h,
                                                      dropout=dropout,
                                                      mask=mask) for _ in range(N)])

        self._embedding = nn.Linear(d_input, d_model)

        self._linear = nn.Linear(d_model * d_channel, d_output)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        x = x.unsqueeze(-1)
        x = x.expand(x.shape[0], x.shape[1], self._d_channel)

        if self._noise:
            mask = torch.rand_like(x[0])
            x = torch.where(mask <= self._noise, torch.Tensor([0]).expand_as(x[0]), x)

        x = x.transpose(-1, -2)

        encoding = self._embedding(x)  # 降维 13 -> 512

        # 位置编码 ------------------------------------- 自己写的
        if self.pe:
            pe = torch.ones_like(encoding[0])
            position = torch.arange(0, self._d_channel).unsqueeze(-1)
            temp = torch.Tensor(range(0, self._d_model, 2))
            temp = temp * -(math.log(10000) / self._d_model)
            temp = torch.exp(temp).unsqueeze(0)
            temp = torch.matmul(position.float(), temp)  # shape:[input, d_model/2]
            pe[:, 0::2] = torch.sin(temp)
            pe[:, 1::2] = torch.cos(temp)

            encoding = encoding + pe

        # Encoding stack
        for layer in self.layers_encoding:
            encoding = layer(encoding)

        encoding = encoding.reshape(encoding.shape[0], -1)

        output = self._linear(encoding)

        return output
