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

    J_FNOW_MHEALTH = 'jordao_etal/FNOW_MHEALTH'
    J_FNOW_USCHAD = 'jordao_etal/FNOW_USCHAD'
    J_FNOW_UTD_MHAD1_1s = 'jordao_etal/FNOW_UTD-MHAD1_1s'
    J_FNOW_UTD_MHAD2_1s = 'jordao_etal/FNOW_UTD-MHAD2_1s'
    J_FNOW_WHARF = 'jordao_etal/FNOW_WHARF'
    J_FNOW_WISDM = 'jordao_etal/FNOW_WISDM'

    J_LOSO_MHEALTH = 'jordao_etal/LOSO_MHEALTH'
    J_LOSO_USCHAD = 'jordao_etal/LOSO_USCHAD'
    J_LOSO_UTD_MHAD1_1s = 'jordao_etal/LOSO_UTD-MHAD1_1s'
    J_LOSO_UTD_MHAD2_1s = 'jordao_etal/LOSO_UTD-MHAD2_1s'
    J_LOSO_WHARF = 'jordao_etal/LOSO_WHARF'
    J_LOSO_WISDM = 'jordao_etal/LOSO_WISDM'

    J_LOTO_MHEALTH = 'jordao_etal/LOTO_MHEALTH'
    J_LOTO_USCHAD = 'jordao_etal/LOTO_USCHAD'
    J_LOTO_UTD_MHAD1_1s = 'jordao_etal/LOTO_UTD-MHAD1_1s'
    J_LOTO_UTD_MHAD2_1s = 'jordao_etal/LOTO_UTD-MHAD2_1s'
    J_LOTO_WHARF = 'jordao_etal/LOTO_WHARF'
    J_LOTO_WISDM = 'jordao_etal/LOTO_WISDM'

    J_SNOW_MHEALTH = 'jordao_etal/SNOW_MHEALTH'
    J_SNOW_USCHAD = 'jordao_etal/SNOW_USCHAD'
    J_SNOW_UTD_MHAD1_1s = 'jordao_etal/SNOW_UTD-MHAD1_1s'
    J_SNOW_UTD_MHAD2_1s = 'jordao_etal/SNOW_UTD-MHAD2_1s'
    J_SNOW_WHARF = 'jordao_etal/SNOW_WHARF'
    J_SNOW_WISDM = 'jordao_etal/SNOW_WISDM'

    @staticmethod
    def get_path(dataset):
        return f'{DATA_ROOT}/{dataset.value}.h5'

    @staticmethod
    def load(dataset):
        return h5py.File(Dataset.get_path(dataset), 'r')


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


def load_split_dataset(dataset, prefold, random_state=42):
    x, y, class_count = load_dataset(dataset)

    if prefold is not None:
        with Dataset.load(dataset) as h5f:
            fold = h5f['folds'][prefold]
            return (x[fold == 1], x[fold == 0], y[fold == 1], y[fold == 0]), class_count
    # Order of return values is: x_train, x_test, y_train, y_test
    return train_test_split(x, y, test_size=0.15, random_state=random_state), class_count
