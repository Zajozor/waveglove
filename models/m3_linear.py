import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F

from models.lightning_common import CommonModel
from models.utils.common import TENSORBOARD_ROOT


class LinearNNModel(CommonModel):
    def __init__(self, hparams, xs, ys):
        super().__init__(hparams, xs, ys)

        self.l1 = torch.nn.Linear(hparams['temporal_length'] * hparams['channels'], 1024)
        self.l2 = torch.nn.Linear(1024, 512)
        self.l3 = torch.nn.Linear(512, hparams['class_count'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.l1(x.reshape(x.shape[0], -1)))
        x = torch.tanh(self.l2(x))
        return F.softmax(self.l3(x))


def feature_extraction(xs):
    return xs


def train(x_train, y_train, class_count, *args, **kwargs):
    model = LinearNNModel({
        'class_count': class_count,
        'temporal_length': x_train.shape[1],
        'channels': x_train.shape[2],
    }, x_train, y_train)
    logger = TensorBoardLogger(TENSORBOARD_ROOT, name='m3')
    trainer = Trainer(gpus=1, logger=logger)
    trainer.fit(model)
    return model


def test(model, x_test):
    model.freeze()
    x_test = torch.tensor(x_test, dtype=torch.float32, device='cuda')
    y_hat = model(x_test).cpu().numpy()

    target_class = np.argmax(y_hat, axis=1)
    return target_class
