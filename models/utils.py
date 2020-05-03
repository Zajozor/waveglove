from enum import Enum

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
import socket
from sklearn.preprocessing import normalize


DATA_ROOTS = {
    'zor-empowered.local': '/Users/zajozor/Code/awesome-har-datasets/data',
    'naiveneuron-s1': '/ext/zajo/data',
}
DATA_ROOT = DATA_ROOTS[socket.gethostname()]


class Dataset(Enum):
    WAVEGLOVE_MULTI = 'waveglove_multi'
    WAVEGLOVE_SINGLE = 'waveglove_single'
    UWAVE = 'uwave'
    OPPORTUNITY = 'opportunity'
    MHEALTH = 'mhealth'
    SKODA = 'skoda'
    PAMAP2 = 'pamap2'
    USCHAD = 'uschad'
    WISDM = 'wisdm'

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


def create_confusion_matrix(class_count, y_hat, y):
    matrix = np.zeros((class_count, class_count), dtype=np.int)
    for i in range(y_hat.shape[0]):
        matrix[y[i]][y_hat[i]] += 1
    return matrix


def plot_confusion_matrix(matrix, annot=None, title=None, ax=None, norm_row=True, fmt='d',
                          xlabel=None, ylabel=None):
    if norm_row:
        matrix = normalize(matrix, axis=1, norm='l1')
        fmt = '.2f'

    if annot is None:
        annot = [[f'{item:.2f}'.rstrip('0.') for item in row] for row in matrix]
        fmt = ''

    if not ax:
        ax = plt.axes()
    if title:
        ax.set_title(title)

    sns.heatmap(matrix, ax=ax, annot=annot, fmt=fmt, cmap='Blues')
    ax.set_xlabel('Prediction' if xlabel is None else xlabel)
    ax.set_ylabel('Truth' if ylabel is None else ylabel)
