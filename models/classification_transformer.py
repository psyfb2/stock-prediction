from typing import Union, Callable

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules.normalization import LayerNorm
from torch.nn.init import xavier_uniform_ 
from torchinfo import summary

from models.positional_encoding import PositionalEncoding


class ClassificationTransformer(nn.Module):
    def __init__(self, seq_len: int, in_features: int, num_classes: int, 
                 d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6, 
                 dim_feedforward: int = 2048, dropout: float = 0.1, 
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu, 
                 layer_norm_eps: float = 1e-5, norm_first: bool = False, 
                 bias: bool = True, device=None, dtype=None):
        """Transformer as provided in "Attention is all you need". 
        However, modified for time-series classification by:
            1. replace word embeddings with a linear layer
            2. no masking
            3. no decoder
            4. classification head on encoder output

        (batch_size, seq_len, in_features) => Transformer => (batch_size, num_classes)

        Args:
            d_model (int, optional):  the number of expected features in the encoder/decoder inputs. Defaults to 512.
            nhead (int, optional): the number of heads in the multiheadattention models. Defaults to 8.
            num_encoder_layers (int, optional):  the number of sub-encoder-layers in the encoder. Defaults to 6.
            dim_feedforward (int, optional): the dimension of the feedforward network model. Defaults to 2048.
            dropout (float, optional): the dropout value. Defaults to 0.1.
            activation (Union[str, Callable[[Tensor], Tensor]], optional):  the activation function of encoder 
                intermediate layer, can be a string ("relu" or "gelu") or a unary callable.  Defaults to F.relu.
            layer_norm_eps (float, optional): the eps value in layer normalization components. Defaults to 1e-5.
            norm_first (bool, optional): if ``True``, encoder and decoder layers will perform LayerNorms before
                other attention and feedforward operations, otherwise after. Defaults to False.
            bias (bool, optional):  If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
                bias. Defaults to True.
        """
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        # layers to prepare data for encoder
        self.input_layer = nn.Linear(in_features, d_model, **factory_kwargs)
        self.pos_embedding = PositionalEncoding(d_model, dropout=dropout, max_len=seq_len)

        # create encoder
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                activation, layer_norm_eps, True, norm_first,
                                                bias, **factory_kwargs)
        encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # create classification head
        self.classification_head = nn.Sequential(nn.Flatten(), nn.Linear(d_model*seq_len, num_classes, **factory_kwargs))

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
    
    def forward(self, x: Tensor) -> Tensor:
        """ Feed x through classification Transformer

        Args:
            x (Tensor): Tensor with shape (batch, seq_len, feature) if self.batch_first
                else (seq_len, batch, features)

        Returns:
            Tensor: output with shape (batch, num_classes)
        """
        x = self.input_layer(x)  # (batch, seq_len, d_model)

        # pos_embedding expects x shape to be (seq_len, batch, d_model)
        x = x.transpose(0, 1)      # (seq_len, batch, d_model)
        x = self.pos_embedding(x)  # (seq_len, batch, d_model)
        x = x.transpose(0, 1)      # (batch, seq_len, d_model)

        x = self.encoder(x)  # (batch, seq_len, d_model)

        y = self.classification_head(x)  #  (batch, num_classes)

        return y

    def _reset_parameters(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)


if __name__ == "__main__":
    # test transformer with dummy input
    batch, seq_len, features = 64, 128, 32
    num_classes = 2
    x = torch.rand((batch, seq_len, features))

    transformer = ClassificationTransformer(seq_len, features, num_classes)
    summary(transformer, input_size=(batch, seq_len, features))

    y = transformer(x)
    assert y.size() == torch.Size([batch, num_classes])
