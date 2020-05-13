import torch

from models.lightning_common import CommonModel, common_test, common_train
from torch.nn import functional as F


# Architecture based on
# https://github.com/dspanah/Sensor-Based-Human-Activity-Recognition-LSTMsEnsemble-Pytorch/blob/master/notebooks/1.0-dsp-LSTMsEnsemle.ipynb
class LSTMModel(CommonModel):
    def __init__(self, hparams, xst, yst, xsv, ysv):
        super().__init__(hparams, xst, yst, xsv, ysv)

        self.n_layers = hparams['layer_count']
        self.n_hidden = hparams['hidden_size']
        self.drop_prob = hparams['drop_prob']

        self.n_classes = hparams['class_count']
        self.n_channels = hparams['channels']

        self.lstm = torch.nn.LSTM(self.n_channels, self.n_hidden, self.n_layers, dropout=self.drop_prob,
                                  batch_first=True)
        self.fc = torch.nn.Linear(self.n_hidden, self.n_classes)
        self.dropout = torch.nn.Dropout(self.drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.lstm(x)
        x = self.dropout(x)

        # Dot product attention
        attention = torch.bmm(x, x[:, -1].unsqueeze(2))
        # attention = F.softmax(attention, dim=1)
        x = torch.bmm(attention.permute(0, 2, 1), x).squeeze(1)

        x = self.fc(x)
        return x


def feature_extraction(xs):
    return xs


def train(x_train, y_train, class_count, *args, **kwargs):
    return common_train(x_train, y_train, LSTMModel,
                        {
                            'layer_count': kwargs['layer_count'],
                            'hidden_size': kwargs['hidden_size'],
                            'drop_prob': kwargs['drop_prob'],
                            'temporal_length': x_train.shape[1],

                            'lr': kwargs['lr'],
                            'class_count': class_count,
                            'channels': x_train.shape[2],
                        },
                        kwargs['folds'])


def test(model, x_test):
    return common_test(model, x_test)
