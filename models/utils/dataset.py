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
    PAMAP2 = 'pamap2'
    SKODA = 'skoda'
    MHEALTH = 'mhealth'

    @staticmethod
    def get_path(dataset):
        return f'{DATA_ROOT}/{dataset.value}.h5'

    @staticmethod
    def load(dataset):
        return h5py.File(Dataset.get_path(dataset), 'r')


def load_split_dataset(dataset):
    with Dataset.load(dataset) as h5f:
        x = np.array(h5f['x'])
        y = np.array(h5f['y']['class'])
        class_count = len(h5f['y'].attrs['classes'])
        x[np.isnan(x)] = 0
        sensor_means = x.reshape(-1, x.shape[2]).mean(axis=0)
        sensor_stds = x.reshape(-1, x.shape[2]).std(axis=0)
        x = (x - sensor_means) / sensor_stds

    return train_test_split(x, y, test_size=0.15, random_state=42), class_count
