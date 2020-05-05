from enum import Enum

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

from models.utils.common import DATA_ROOT


class Dataset(Enum):
    WAVEGLOVE_MULTI = 'waveglove_multi'
    WAVEGLOVE_SINGLE = 'waveglove_single'
    UWAVE = 'uwave'
    OPPORTUNITY = 'opportunity'
    MHEALTH = 'mhealth'
    SKODA = 'skoda'
    PAMAP2 = 'pamap2'
    # USCHAD = 'uschad'
    # WISDM = 'wisdm'

    @staticmethod
    def get_path(dataset):
        return f'{DATA_ROOT}/{dataset.value}.h5'

    @staticmethod
    def load(dataset):
        return h5py.File(Dataset.get_path(dataset), 'r')


def split_data(x, y, nan_to_num=True):
    if nan_to_num:
        x, y = np.nan_to_num(x), np.nan_to_num(y)
    # returns x_train, x_test, y_train, y_test
    return train_test_split(x, y, test_size=0.2, random_state=42)


def load_split_dataset(dataset):
    with Dataset.load(dataset) as h5f:
        x = np.array(h5f['x'])
        y = np.array(h5f['y']['class'])
        class_count = len(h5f['y'].attrs['classes'])

    return split_data(x, y), class_count
