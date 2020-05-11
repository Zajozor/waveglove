import h5py
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from matplotlib import pyplot as plt
from models.lightning_common import common_test
from models.utils import metrics as u_metrics, plot as u_plot

torch.manual_seed(42)
BATCH_SIZE = 32

ds_name = 'skoda'


class LSTMSAModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.l1 = torch.nn.Linear(hparams['in_length'] * hparams['in_channels'], 2048)
        self.l2 = torch.nn.Linear(2048, 1024)
        self.l3 = torch.nn.Linear(1024, hparams['class_count'])
        self.tds = None
        self.vds = None

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def prepare_data(self):
        with h5py.File(f'/ext/zajo/data/{ds_name}.h5', 'r') as h5f:
            xs = torch.tensor(h5f['x'], dtype=torch.float32, device='cuda')
            ys = torch.tensor(h5f['y']['class'], dtype=torch.long, device='cuda')
            xs[torch.isnan(xs)] = 0
        ds = TensorDataset(xs, ys)

        train_size = int(xs.shape[0] * 0.9)
        self.tds, self.vds = random_split(ds, [train_size, xs.shape[0] - train_size])

    def train_dataloader(self):
        return DataLoader(self.tds, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.vds, batch_size=BATCH_SIZE)

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
        return {'val_loss': avg_loss, 'log': {'val_loss': avg_loss}, 'progress_bar': {'val_loss': avg_loss}}

    def forward(self, x):
        x = torch.relu(self.l1(x.reshape(x.shape[0], -1)))
        x = torch.tanh(self.l2(x))
        return self.l3(x)


if __name__ == '__main__':

    with h5py.File(f'/ext/zajo/data/{ds_name}.h5', 'r') as h5f:
        xs = h5f['x']
        length = xs.shape[1]
        channels = xs.shape[2]
        classes = h5f['y'].attrs['classes'].shape[0]

    model = LSTMSAModule({
        'in_length': length, 'in_channels': channels, 'class_count': classes,
    })

    trainer = Trainer(gpus=1,
                      early_stop_callback=EarlyStopping(
                          monitor='val_loss',
                          min_delta=0,
                          patience=10,
                          verbose=True,
                          mode='min'),
                      )
    trainer.fit(model)

    with h5py.File(f'/ext/zajo/data/{ds_name}.h5', 'r') as h5f:
        xs = torch.tensor(h5f['x'], dtype=torch.float32, device='cuda')
        xs[torch.isnan(xs)] = 0
        ys = torch.tensor(h5f['y']['class'], dtype=torch.long, device='cuda')

    y_hat = common_test(model, xs)

    fig, ax = plt.subplots(1, 1)
    cm_train = u_metrics.create_confusion_matrix(classes, y_hat, ys)
    u_plot.plot_confusion_matrix(cm_train, title=f'All', ax=ax)

    acc_train, recall_train, f1_train = u_metrics.get_metrics(y_hat, ys.cpu().numpy())
    print(f'Acc: {acc_train}, Rec: {recall_train}, F1: {f1_train}')
