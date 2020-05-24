from enum import Enum

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

from models.utils.common import DATA_ROOT


class Dataset(Enum):
    J_LOSO_MHEALTH = 'LOSO_MHEALTH'
    J_LOSO_USCHAD = 'LOSO_USCHAD'
    J_LOSO_UTD_MHAD1_1s = 'LOSO_UTD-MHAD1_1s'
    J_LOSO_UTD_MHAD2_1s = 'LOSO_UTD-MHAD2_1s'
    J_LOSO_WHARF = 'LOSO_WHARF'
    J_LOSO_WISDM = 'LOSO_WISDM'

    J_LOTO_MHEALTH = 'LOTO_MHEALTH'
    J_LOTO_USCHAD = 'LOTO_USCHAD'
    J_LOTO_UTD_MHAD1_1s = 'LOTO_UTD-MHAD1_1s'
    J_LOTO_UTD_MHAD2_1s = 'LOTO_UTD-MHAD2_1s'
    J_LOTO_WHARF = 'LOTO_WHARF'
    J_LOTO_WISDM = 'LOTO_WISDM'

    @staticmethod
    def get_path(dataset):
        return f'{DATA_ROOT}/{dataset.value}.h5'

    @staticmethod
    def load(dataset):
        return h5py.File(Dataset.get_path(dataset), 'r')

    @staticmethod
    def get_prefold_range(dataset):
        with Dataset.load(dataset) as h5f:
            if 'folds' not in h5f:
                return [None]
            return range(h5f['folds'].shape[0])


def load_dataset(dataset):
    with Dataset.load(dataset) as h5f:
        x = np.array(h5f['x'])
        y = np.array(h5f['y']['class'])
        class_count = len(h5f['y'].attrs['classes'])
        x[np.isnan(x)] = 0
        sensor_means = x.reshape(-1, x.shape[2]).mean(axis=0)
        sensor_stds = x.reshape(-1, x.shape[2]).std(axis=0)
        x = (x - sensor_means) / sensor_stds
    return x, y, class_count


def load_split_dataset(dataset, prefold, random_state=42, test_size=0.15):
    x, y, class_count = load_dataset(dataset)

    if prefold is not None:
        with Dataset.load(dataset) as h5f:
            fold = h5f['folds'][prefold]
            return (x[fold == 1], x[fold == 0], y[fold == 1], y[fold == 0]), class_count
    # Order of return values is: x_train, x_test, y_train, y_test
    return train_test_split(x, y, test_size=test_size, random_state=random_state), class_count
