import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from models.lightning_common import CommonModel
from models.utils.common import TENSORBOARD_ROOT, get_formatted_datetime


# Architecture like, but without one maxpool,
# https://github.com/arturjordao/WearableSensorData/blob/master/ChenXue2015.py
class CNNModel(CommonModel):
    def __init__(self, hparams, xs, ys):
        super().__init__(hparams, xs, ys)

        self.c1 = torch.nn.Conv2d(1, 18, kernel_size=(13, 2))
        self.mp1 = torch.nn.MaxPool2d(kernel_size=(2, 1))

        self.c2 = torch.nn.Conv2d(18, 36, kernel_size=(7, 1))

        self.c3 = torch.nn.Conv2d(36, 24, kernel_size=(7, 1))
        self.mp3 = torch.nn.MaxPool2d(kernel_size=(2, 1))

        final_width = ((hparams['temporal_length'] - 12) // 2 - 6 - 6) // 2
        self.final = torch.nn.Linear(24 * final_width * (hparams['channels'] - 1), hparams['class_count'])
        self.final_softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view((x.shape[0], 1, x.shape[1], x.shape[2]))
        x = self.mp1(torch.relu(self.c1(x)))
        x = torch.relu(self.c2(x))
        x = self.mp3(torch.relu(self.c3(x)))

        return self.final_softmax(self.final(x.flatten(start_dim=1)))


def feature_extraction(xs):
    return xs


def train(x_train, y_train, class_count, *args, **kwargs):
    model = CNNModel({
        'class_count': class_count,
        'temporal_length': x_train.shape[1],
        'channels': x_train.shape[2],
    }, x_train, y_train)
    logger = TensorBoardLogger(TENSORBOARD_ROOT, name='m4')
    trainer = Trainer(gpus=1, logger=logger)
    trainer.fit(model)
    # trainer.test()
    # TODO use trainer.test
    trainer.save_checkpoint(f'checkpoints/m4.{get_formatted_datetime()}.ckpt')
    return model


def test(model, x_test):
    model.freeze()
    x_test = torch.tensor(x_test, dtype=torch.float32, device='cuda')

    batch_size = 64

    target_class = np.empty((((x_test.shape[0] - 1) // batch_size + 1) * batch_size,), dtype='int64')

    batch_count = (x_test.shape[0] - 1) // batch_size + 1
    for i in range(batch_count):
        y_hat = model(x_test[i * batch_size: (i + 1) * batch_size]).cpu().numpy()
        target_class[i * batch_size: i * batch_size + y_hat.shape[0]] = np.argmax(y_hat, axis=1)
    target_class = target_class[:x_test.shape[0]]
    return target_class
