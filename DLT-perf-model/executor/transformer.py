import math
from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import MSELoss

from .base_module import MModule
import logging

class PositionalEncoder(nn.Module):
    def __init__(
            self,
            dropout: float = 0.1,
            max_seq_len: int = 5000,
            d_model: int = 512,
            batch_first: bool = False
    ):

        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model
                     (Vaswani et al, 2017)
        """

        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        self.batch_first = batch_first
        # adapted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        if self.batch_first:
            pe = torch.zeros(1, max_seq_len, d_model)
            pe[0, :, 0::2] = torch.sin(position * div_term)
            if d_model % 2 != 0:
                pe[0, :, 1::2] = torch.cos(position * div_term)[:, 0:-1]
            else:
                pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_seq_len, 1, d_model)
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            if d_model % 2 != 0:
                pe[:, 0, 1::2] = torch.cos(position * div_term)[:, 0:-1]
            else:
                pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or
               [enc_seq_len, batch_size, dim_val]
        """
        if self.batch_first:
            x = x + self.pe[:, :x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(MModule):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 d_hid: int,
                 nlayers: int,
                 dropout: float,
                 output_d: int):
        super().__init__()
        logging.info("Transformer model inited, hyper parameters: {}".format(locals()))
        self.pos_encoder = PositionalEncoder(dropout=0.1,
                                             max_seq_len=100,
                                             d_model=d_model,
                                             batch_first=True)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.output_d = output_d
        self.decoder = nn.Linear(d_model, output_d)
        self.loss_fn = MSELoss()

    def prepare_transfer(self, freeze_layers: int | None = None, reinit_proj: bool = True, **kwargs):
        if freeze_layers is not None:
            layers = self.transformer_encoder.layers
            if freeze_layers > len(layers):
                raise ValueError(f"freeze_layers ({freeze_layers}) must be less than the number of layers ")
            for layer in layers[freeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = False
        if reinit_proj:
            self.decoder.reset_parameters()

    def forward(self, X: Dict) -> Tensor:
        """
        Arguments:
            X: Tensor, key "x_subgraph_feature" has shape ``[batch_size, seq_len, embedding]``

        Returns:
            output Tensor of shape ``[batch_size, output_d]``
        """
        src = X["x_subgraph_feature"]
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

    def compute_loss(self, outputs, Y):
        y_nodes_durations = Y["y_nodes_durations"]
        loss = self.loss_fn(outputs, y_nodes_durations)
        return loss
