import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from models.utils.common import TENSORBOARD_ROOT


class HARModel(pl.LightningModule):
    def __init__(self, hparams, xs, ys):
        super().__init__()

        self.hparams = hparams
        self.l1 = torch.nn.Linear(hparams['temporal_length'] * hparams['channels'], 1024)
        self.l2 = torch.nn.Linear(1024, 512)
        self.l3 = torch.nn.Linear(512, hparams['class_count'])
        self._xs = xs
        self._ys = ys

    def prepare_data(self):
        train_count = int(self._xs.shape[0] * 0.9)
        valid_count = self._xs.shape[0] - train_count

        xs = torch.tensor(self._xs, dtype=torch.float32, device='cuda')
        ys = torch.tensor(self._ys, dtype=torch.int64, device='cuda')

        dataset = TensorDataset(xs, ys)
        self.train_ds, self.valid_ds = torch.utils.data.random_split(dataset, [train_count, valid_count])
        print(f'Loaded data. xs: {xs.shape}, ys: {ys.shape}')
        print(f'Train count: {len(self.train_ds)}, Validation count: {len(self.valid_ds)}')

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=64, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=64, num_workers=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO configure forward
        x = torch.relu(self.l1(x.reshape(x.shape[0], -1)))
        x = torch.relu(self.l2(x))
        return torch.relu(self.l3(x))

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss, 'log': {'val_loss': avg_loss}}


def feature_extraction(xs):
    return xs


def train(x_train, y_train, class_count, *args, **kwargs):
    model = HARModel({
        'class_count': class_count,
        'temporal_length': x_train.shape[1],
        'channels': x_train.shape[2],
    }, x_train, y_train)
    logger = TensorBoardLogger(TENSORBOARD_ROOT, name='m3')
    trainer = Trainer(gpus=1, logger=logger, max_epochs=100)
    trainer.fit(model)
    return model


def test(model, x_test):
    model.freeze()
    x_test = torch.tensor(x_test, dtype=torch.float32, device='cuda')
    y_hat = model(x_test).cpu().numpy()

    target_class = np.argmax(y_hat, axis=1)
    return target_class
