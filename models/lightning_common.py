import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset

from models.utils.common import get_logger

BATCH_SIZE = 32


class CommonModel(pl.LightningModule):
    def __init__(self, hparams, xst, yst, xsv, ysv):
        super().__init__()

        self.hparams = hparams
        self._xst = torch.tensor(xst, dtype=torch.float32, device='cuda')
        self._yst = torch.tensor(yst, dtype=torch.int64, device='cuda')

        self._xsv = torch.tensor(xsv, dtype=torch.float32, device='cuda')
        self._ysv = torch.tensor(ysv, dtype=torch.int64, device='cuda')
        print(f'Created model. '
              f'Data shape: '
              f'x-train {xst.shape} y-train {yst.shape} '
              f'x-valid {xsv.shape} y-valid {ysv.shape}')

    def configure_optimizers(self) -> Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])

    def train_dataloader(self) -> DataLoader:
        ds = TensorDataset(self._xst, self._yst)
        return DataLoader(ds, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        ds = TensorDataset(self._xsv, self._ysv)
        return DataLoader(ds, batch_size=BATCH_SIZE)

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
        return {'val_loss': avg_loss, 'progress_bar': {'val_loss': avg_loss}, 'log': {'val_loss': avg_loss}}

    def forward(self, *args, **kwargs):
        raise NotImplementedError('This class should be subclassed, not used directly.')


def common_train(x_train, y_train, model_class, model_hparams, folds=None):
    if folds is None:
        folds = [train_test_split(np.arange(x_train.shape[0]), test_size=0.15, random_state=42)]
    else:
        folds = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42).split(x_train, y_train)

    best_model, best_f1 = None, 0
    f1_scores = []
    for train_idx, valid_idx in folds:
        xst, yst = x_train[train_idx], y_train[train_idx]
        xsv, ysv = x_train[valid_idx], y_train[valid_idx]
        model = model_class(
            model_hparams,
            xst, yst, xsv, ysv
        )
        trainer = Trainer(gpus=1,
                          logger=get_logger(),
                          early_stop_callback=EarlyStopping(
                              monitor='val_loss',
                              min_delta=0,
                              patience=10,
                              verbose=True,
                              mode='min'
                          ),
                          min_epochs=10)
        trainer.fit(model)

        y_hat = common_test(model, xsv)

        f1 = f1_score(ysv, y_hat, average='macro')
        f1_scores.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
    print(f'Chosen f1 {best_f1} from {f1_scores}')
    return best_model


def common_test(model, x_test):
    model.freeze()
    x_test = torch.tensor(x_test, dtype=torch.float32, device='cuda')
    target_class = np.empty((((x_test.shape[0] - 1) // BATCH_SIZE + 1) * BATCH_SIZE,), dtype='int64')

    batch_count = (x_test.shape[0] - 1) // BATCH_SIZE + 1
    for i in range(batch_count):
        y_hat_prob = model(x_test[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]).cpu().numpy()
        target_class[i * BATCH_SIZE: i * BATCH_SIZE + y_hat_prob.shape[0]] = np.argmax(y_hat_prob, axis=1)
    target_class = target_class[:x_test.shape[0]]
    return target_class
