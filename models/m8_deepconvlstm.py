import torch

from models.lightning_common import CommonModel, common_test, common_train


# Architecture based on
# Ordonez et al.
# https://github.com/sussexwearlab/DeepConvLSTM/blob/master/DeepConvLSTM.ipynb
class LSTMModel(CommonModel):
    def __init__(self, hparams, xst, yst, xsv, ysv):
        super().__init__(hparams, xst, yst, xsv, ysv)

        hidden_size = 32
        cnn_filters = 3

        self.c1 = torch.nn.Conv2d(1, cnn_filters, kernel_size=(5, 1))
        self.c2 = torch.nn.Conv2d(cnn_filters, cnn_filters, kernel_size=(5, 1))
        self.c3 = torch.nn.Conv2d(cnn_filters, cnn_filters, kernel_size=(5, 1))
        self.c4 = torch.nn.Conv2d(cnn_filters, cnn_filters, kernel_size=(5, 1))

        self.lstm = torch.nn.LSTM(cnn_filters * hparams['channels'], hidden_size, 1, dropout=0, batch_first=True)

        self.fc = torch.nn.Linear(hidden_size, hparams['class_count'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        x = torch.relu(self.c1(x))
        x = torch.relu(self.c2(x))
        x = torch.relu(self.c3(x))
        x = torch.relu(self.c4(x))

        x = x.permute(0, 2, 1, 3).flatten(start_dim=2)
        x, _ = self.lstm(x)
        x = x[:, -1]

        x = self.fc(x)
        return x


def feature_extraction(xs):
    return xs


def train(x_train, y_train, class_count, *args, **kwargs):
    return common_train(x_train, y_train, LSTMModel,
                        {
                            'lr': kwargs['lr'],
                            'class_count': class_count,
                            'channels': x_train.shape[2],
                        },
                        kwargs['folds'])


def test(model, x_test):
    return common_test(model, x_test)
