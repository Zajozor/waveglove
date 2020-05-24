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
    'zor-empowered.local': '/Users/zajozor/Code/school/waveglove-ml23/tensorboard',
    'naiveneuron-s1': '/ext/zajo/tensorboard-ml23',
}
TENSORBOARD_ROOT = TENSORBOARD_ROOTS[hostname]

RUN_LOG_FILE = 'hparams-{model}.log'

_buffer = ''


def add_log(model, line, newline=True):
    global _buffer
    print(f'[{model}]', line)
    _buffer += line
    if newline:
        with open(RUN_LOG_FILE.format(model=model), 'a') as f:
            f.write(_buffer)
            f.write('\n')
        _buffer = ''


# Wrapped in a list to make "referencable", would be nicer in a class
logger: List[Union[None, TensorBoardLogger]] = [None]


def get_formatted_datetime():
    return datetime.datetime.now().strftime("%b%d-%H:%M:%S")


def set_logger(model_name, dataset, hp_id='', hparams=None, prefold=None):
    global logger
    if prefold is None:
        prefold = ''

    dt = get_formatted_datetime()
    logger[0] = TensorBoardLogger(TENSORBOARD_ROOT,
                                  name=f'{model_name}-{dataset.value}',
                                  version=f'{hp_id}-{prefold}-{dt}')
    add_log(model_name, f'{dt} | {model_name:12} | {hp_id:3} | {prefold:2} | {dataset.value:30} | {hparams} || ',
            newline=False)


def get_logger():
    if logger[0] is None:
        raise ValueError('Writer needs to be set first!')
    return logger[0]
