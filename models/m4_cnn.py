import torch

from models.lightning_common import CommonModel, common_test, common_train


# Architecture like, but without one maxpool,
# https://github.com/arturjordao/WearableSensorData/blob/master/ChenXue2015.py
class CNNModel(CommonModel):
    def __init__(self, hparams, xst, yst, xsv, ysv):
        super().__init__(hparams, xst, yst, xsv, ysv)

        self.zp1 = torch.nn.ZeroPad2d((0, 1, 0, hparams['kernel_width1'] - 1))
        self.c1 = torch.nn.Conv2d(1, hparams['filters1'], kernel_size=(hparams['kernel_width1'], 2))
        self.mp1 = torch.nn.MaxPool2d(kernel_size=(2, 1))

        self.zp2 = torch.nn.ZeroPad2d((0, 0, 0, hparams['kernel_width2'] - 1))
        self.c2 = torch.nn.Conv2d(hparams['filters1'], hparams['filters2'], kernel_size=(hparams['kernel_width2'], 1))
        self.mp2 = torch.nn.MaxPool2d(kernel_size=(2, 1))

        self.zp3 = torch.nn.ZeroPad2d((0, 0, 0, hparams['kernel_width3'] - 1))
        self.c3 = torch.nn.Conv2d(hparams['filters2'], hparams['filters3'], kernel_size=(hparams['kernel_width3'], 1))
        self.mp3 = torch.nn.MaxPool2d(kernel_size=(2, 1))

        width_after3 = hparams['temporal_length'] // 8

        self.final = torch.nn.Linear(hparams['filters3'] * width_after3 * hparams['channels'],
                                     hparams['class_count'])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view((x.shape[0], 1, x.shape[1], x.shape[2]))

        x = self.mp1(torch.relu(self.c1(self.zp1(x))))
        x = self.mp2(torch.relu(self.c2(self.zp2(x))))
        x = self.mp3(torch.relu(self.c3(self.zp3(x))))

        return self.final(x.flatten(start_dim=1))


def feature_extraction(xs):
    return xs


def train(x_train, y_train, class_count, **kwargs):
    return common_train(x_train, y_train, CNNModel,
                        {
                            'filters1': kwargs['filters1'],
                            'kernel_width1': kwargs['kernel_width1'],
                            'filters2': kwargs['filters2'],
                            'kernel_width2': kwargs['kernel_width2'],
                            'filters3': kwargs['filters3'],
                            'kernel_width3': kwargs['kernel_width3'],
                            'lr': kwargs['lr'],

                            'class_count': class_count,
                            'temporal_length': x_train.shape[1],
                            'channels': x_train.shape[2],
                        },
                        kwargs['folds'])


def test(model, x_test):
    return common_test(model, x_test)
