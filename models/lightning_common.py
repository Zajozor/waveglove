import pytorch_lightning as pl
import torch
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset


class CommonModel(pl.LightningModule):
    def __init__(self, hparams, xs, ys):
        super().__init__()

        self.hparams = hparams
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
        return DataLoader(self.train_ds, batch_size=64)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=64)

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
