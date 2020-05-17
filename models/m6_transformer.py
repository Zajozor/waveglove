import math

import torch
from torch.nn import TransformerEncoderLayer, TransformerEncoder

from models.lightning_common import CommonModel, common_test, common_train


# PositionalEncoding module adopted from https://github.com/pytorch/examples/blob/master/word_language_model/model.py
# Fixed an issue to work with odd dims
class PositionalEncoding(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, (d_model + 1) // 2 * 2)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :, :x.size(2)]
        return self.dropout(x)


class TransformerModel(CommonModel):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""

    def __init__(self, hparams, xst, yst, xsv, ysv):
        super().__init__(hparams, xst, yst, xsv, ysv)

        self.sensor_expansion = torch.nn.Linear(hparams['channels'], hparams['sensor_embed_dim'])

        self.pos_encoder = PositionalEncoding(hparams['sensor_embed_dim'], dropout=hparams['dropout'])
        encoder_layer = TransformerEncoderLayer(hparams['sensor_embed_dim'], hparams['encoder_heads'],
                                                hparams['encoder_hidden'], dropout=hparams['dropout'])
        self.transformer_encoder = TransformerEncoder(encoder_layer, hparams['encoder_layers'])

        self.global_temporal = torch.nn.MultiheadAttention(hparams['sensor_embed_dim'],
                                                           hparams['temporal_attention_heads'])
        self.final_linear = torch.nn.Linear(hparams['sensor_embed_dim'], hparams['class_count'])

    def forward(self, x, has_mask=True):
        x = self.sensor_expansion(x)
        x = x + torch.tanh(x)

        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        query = x[-1:]
        x = self.global_temporal(query, x, x)[0]
        x = x.squeeze(0)

        x = self.final_linear(x)
        return x


def feature_extraction(xs):
    return xs


def train(x_train, y_train, class_count, *args, **kwargs):
    folds = kwargs.pop('folds')
    return common_train(x_train, y_train, TransformerModel,
                        {
                            **kwargs,
                            'class_count': class_count,
                            'channels': x_train.shape[2],
                        },
                        folds)


def test(model, x_test):
    return common_test(model, x_test)
