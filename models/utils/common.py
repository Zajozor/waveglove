import socket
import datetime
from typing import List, Union

from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter

hostname = socket.gethostname()

DATA_ROOTS = {
    'zor-empowered.local': '/Users/zajozor/Code/awesome-har-datasets/data',
    'naiveneuron-s1': '/ext/zajo/data',
}
DATA_ROOT = DATA_ROOTS[hostname]

TENSORBOARD_ROOTS = {
    'zor-empowered.local': '/Users/zajozor/Code/school/waveglove/tensorboard',
    'naiveneuron-s1': '/ext/zajo/tensorboard',
}
TENSORBOARD_ROOT = TENSORBOARD_ROOTS[hostname]

# Wrapped in a list to make "referencable", would be nicer in a class
logger: List[Union[None, TensorBoardLogger]] = [None]


def get_formatted_datetime():
    return datetime.datetime.now().strftime("%b%d-%H:%M:%S")


def set_logger(model_name, dataset):
    global logger
    logger[0] = TensorBoardLogger(TENSORBOARD_ROOT,
                                  name=f'{model_name}-{dataset.value}',
                                  version=get_formatted_datetime())


def get_logger():
    if logger[0] is None:
        raise ValueError('Writer needs to be set first!')
    return logger[0]
