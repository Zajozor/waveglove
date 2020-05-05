import socket
import datetime

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

writer = None


def get_formatted_datetime():
    return datetime.datetime.now().strftime("%b%d_%H:%M:%S")


def set_writer(model_name, dataset):
    global writer
    writer = SummaryWriter(f'{TENSORBOARD_ROOT}/'
                           f'{get_formatted_datetime()}-'
                           f'{model_name}-{dataset.value}')

