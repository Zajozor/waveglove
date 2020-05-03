import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize


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