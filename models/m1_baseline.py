import numpy as np
from scipy.spatial import distance


def feature_extraction(xs):
    # No feature extraction for baseline
    return xs


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
