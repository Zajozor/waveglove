import datetime
import socket
from typing import List, Union

from pytorch_lightning.loggers import TensorBoardLogger

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

RUN_LOG_FILE = 'hparams-{model}.log'


def add_log(model, line, newline=True):
    print(f'[{model}]', line)
    with open(RUN_LOG_FILE.format(model=model), 'a') as f:
        f.write(line)
        if newline:
            f.write('\n')


# Wrapped in a list to make "referencable", would be nicer in a class
logger: List[Union[None, TensorBoardLogger]] = [None]


def get_formatted_datetime():
    return datetime.datetime.now().strftime("%b%d-%H:%M:%S")


def set_logger(model_name, dataset, hp_id='', hparams=None):
    global logger
    dt = get_formatted_datetime()
    logger[0] = TensorBoardLogger(TENSORBOARD_ROOT,
                                  name=f'{model_name}-{dataset.value}',
                                  version=f'{hp_id}-{dt}')
    add_log(model_name, f'[{dt}] id{hp_id} on {dataset.value} with {hparams}: ', newline=False)


def get_logger():
    if logger[0] is None:
        raise ValueError('Writer needs to be set first!')
    return logger[0]
