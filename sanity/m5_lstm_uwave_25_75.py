import warnings

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset

from models.lightning_common import common_test
from models.utils import metrics as u_metrics, plot as u_plot

warnings.filterwarnings('ignore', 'The dataloader,')
torch.manual_seed(1337)
BATCH_SIZE = 128

ds_name = 'uwave'


class LSTMSAModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.lstm = torch.nn.LSTM(hparams['in_channels'], hparams['hidden'], hparams['lstm_layers'], batch_first=True,
                                  dropout=hparams['dropout'])
        self.lin = torch.nn.Linear(hparams['hidden'], hparams['class_count'])

        self.tds = None
        self.vds = None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams['lr'])

    def prepare_data(self):
        with h5py.File(f'/ext/zajo/data/{ds_name}.h5', 'r') as h5f:
            xs = torch.tensor(h5f['x'], dtype=torch.float32)
            xs = xs[:, 25:75, :]
            xs[torch.isnan(xs)] = 0
            sensor_means = xs.reshape(-1, xs.shape[2]).mean(axis=0)
            sensor_stds = xs.reshape(-1, xs.shape[2]).std(axis=0)
            xs = (xs - sensor_means) / sensor_stds
            # xs = ((xs - xs.mean(axis=2).unsqueeze(2)) / xs.std(axis=2).unsqueeze(2))

            ys = torch.tensor(h5f['y']['class'], dtype=torch.long)


        idxs = torch.randperm(xs.shape[0])
        ds = TensorDataset(xs, ys)
        self.tds = Subset(ds, idxs[:500])
        self.vds = Subset(ds, idxs[:500])
        print(f'Total histogram: {np.histogram(ds[:][1].cpu().numpy(), self.hparams["class_count"])[0]}')
        print(f'Train histogram: {np.histogram(self.tds[:][1].cpu().numpy(), self.hparams["class_count"])[0]}')
        print(f'Val histogram: {np.histogram(self.vds[:][1].cpu().numpy(), self.hparams["class_count"])[0]}')

    def train_dataloader(self):
        return DataLoader(self.tds, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.vds, batch_size=BATCH_SIZE, )

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        return {'loss': loss, 'log': {'train_loss': loss}}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        print(' -- train loss:', float(avg_loss), end='')
        return {}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        print(' -- val_loss:', float(avg_loss))
        return {'val_loss': avg_loss, 'log': {'val_loss': avg_loss}}

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.lin(x)
        return x


def init_weights(m):
    if type(m) == torch.nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                # torch.nn.init.xavier_uniform_(param.data)
                # param.data.fill_(0)
                pass
    elif type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        # torch.nn.init.xavier_uniform_(m.bias)
        # m.bias.data.fill_(0)
        # self.bias.data.uniform_(-stdv, stdv)



if __name__ == '__main__':
    with h5py.File(f'/ext/zajo/data/{ds_name}.h5', 'r') as h5f:
        xs = h5f['x']
        length = xs.shape[1]
        channels = xs.shape[2]
        classes = h5f['y'].attrs['classes'].shape[0]

    for lstm_layers in [1]:
        for dropout in [0]:
            for hidden in [256]:
                for lr in [0.001]:
                    model = LSTMSAModule({
                        'in_length': length, 'in_channels': channels, 'class_count': classes, 'lr': lr,
                        'hidden': hidden,
                        'lstm_layers': lstm_layers,
                        'dropout': dropout,
                    })
                    model.apply(init_weights)
                    trainer = Trainer(gpus=1,
                                      min_epochs=12,
                                      max_epochs=250,
                                      progress_bar_refresh_rate=0,
                                      )
                    trainer.fit(model)

                    xs = model.tds[:][0]
                    ys = model.tds[:][1]
                    y_hat = common_test(model, xs.cpu().numpy())

                    fig, ax = plt.subplots(1, 1)
                    cm_train = u_metrics.create_confusion_matrix(classes, y_hat, ys)
                    u_plot.plot_confusion_matrix(cm_train, title=f'All', ax=ax)
                    plt.show()
                    print('Dist: ', np.histogram(ys.cpu().numpy(), bins=classes)[0])

                    acc_train, recall_train, f1_train = u_metrics.get_metrics(y_hat, ys.cpu().numpy())
                    print(
                        f'layers: {lstm_layers:6} hidden: {hidden:6} dout: {dropout:6} LR: {lr:6} Acc: {acc_train:6.3f}, Rec: {recall_train:6.3f}, F1: {f1_train:6.3f}')
