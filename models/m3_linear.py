import numpy as np
import torch
from torch.nn import functional as F

from models.lightning_common import CommonModel, common_train, common_test


class LinearNNModel(CommonModel):
    def __init__(self, hparams, xst, yst, xsv, ysv):
        super().__init__(hparams, xst, yst, xsv, ysv)

        self.l1 = torch.nn.Linear(hparams['temporal_length'] * hparams['channels'], hparams['l1'])
        self.l2 = torch.nn.Linear(hparams['l1'], hparams['l2'])
        self.l3 = torch.nn.Linear(hparams['l2'], hparams['class_count'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(x.reshape(x.shape[0], -1)))
        x = torch.tanh(self.l2(x))
        return F.softmax(self.l3(x), dim=1)


def feature_extraction(xs):
    return xs


def train(x_train, y_train, class_count, **kwargs):
    return common_train(x_train, y_train, LinearNNModel,
                        {
                            'l1': kwargs['l1'],
                            'l2': kwargs['l2'],
                            'lr': kwargs['lr'],

                            'class_count': class_count,
                            'temporal_length': x_train.shape[1],
                            'channels': x_train.shape[2],
                        },
                        kwargs['folds'])


def test(model, x_test):
    return common_test(model, x_test)
