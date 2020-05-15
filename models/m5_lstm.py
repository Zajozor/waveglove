import torch

from models.lightning_common import CommonModel, common_test, common_train


# Architecture inspired by, but added the pre embedding layer
# https://github.com/dspanah/Sensor-Based-Human-Activity-Recognition-LSTMsEnsemble-Pytorch/blob/master/notebooks/1.0-dsp-LSTMsEnsemle.ipynb
class LSTMModel(CommonModel):
    def __init__(self, hparams, xst, yst, xsv, ysv):
        super().__init__(hparams, xst, yst, xsv, ysv)

        self.n_layers = hparams['layer_count']
        self.n_hidden = hparams['hidden_size']
        self.drop_prob = hparams['drop_prob']

        self.n_classes = hparams['class_count']
        self.n_channels = hparams['channels']

        self.pre_fc = torch.nn.Linear(hparams['temporal_length'], hparams['pre_length'])
        self.lstm = torch.nn.LSTM(self.n_channels, self.n_hidden, self.n_layers, dropout=self.drop_prob,
                                  batch_first=True)
        self.dropout = torch.nn.Dropout(p=self.drop_prob)
        self.fc = torch.nn.Linear(self.n_hidden, self.n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.pre_fc(x)
        x = torch.tanh(x)
        x = x.permute(0, 2, 1)

        x, _ = self.lstm(x)
        x = self.dropout(x)
        x = x[:, -1]  # Take last output
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
                            'pre_length': kwargs['pre_length'],

                            'lr': kwargs['lr'],
                            'class_count': class_count,
                            'channels': x_train.shape[2],
                        },
                        kwargs['folds'])


def test(model, x_test):
    return common_test(model, x_test)
