import numpy as np
from scipy.spatial import distance
from matplotlib import pyplot as plt
from models import utils


def train(x_train, y_train, class_count, **kwargs):
    gesture_length = x_train.shape[1]
    channel_count = x_train.shape[2]

    # TRAIN
    averages = np.empty((class_count, gesture_length, channel_count))
    for class_id in range(class_count):
        templates = x_train[np.equal(y_train, class_id)]
        averages[class_id] = np.mean(templates, axis=0)

    return averages


def test(model, x_test):
    averages = model
    # TEST
    distances = distance.cdist(
        averages.reshape(averages.shape[0], -1),
        x_test.reshape(x_test.shape[0], -1)
    )
    return distances.argmin(axis=0)  # y_hat


def run(dataset=utils.Dataset.OPPORTUNITY):
    with utils.Dataset.load(dataset) as h5f:
        x = np.array(h5f['x'])
        y = np.array(h5f['y']['class'])
        class_count = len(h5f['y'].attrs['classes'])

    x_train, x_test, y_train, y_test = utils.split_data(x, y)
    model = train(x_train, y_train, class_count)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    y_hat_train = test(model, x_train)
    cm_train = utils.create_confusion_matrix(class_count, y_hat_train, y_train)
    utils.plot_confusion_matrix(cm_train, title=f'{dataset.value} on training', ax=ax1)

    y_hat = test(model, x_test)
    cm = utils.create_confusion_matrix(class_count, y_hat, y_test)
    utils.plot_confusion_matrix(cm, title=f'{dataset.value} on test', ax=ax2)

    fig.show()


if __name__ == '__main__':
    run()

# SKODA lepsia segmentacia
# PAMAP prepisat labely
# ODSTRANIT NAN ZA NULY pri loadingu?
# labels k datasetom lepsie cisla v cmatrixe?
