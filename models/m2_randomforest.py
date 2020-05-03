import numpy as np
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.ensemble import BaggingClassifier

from models import features as fs
from models import utils


def feature_extraction(xs):
    # Extracts the features, as mentioned by Catal et al. 2015
    used_features = [fs.channel_correlation, fs.channel_mean]
    x_features = []
    count = 0
    for sample in xs:
        x_features.append(np.hstack([feature(sample) for feature in used_features]))
        count += 1
        if count % 100 == 0:
            print('Extracting sample', count)

    return np.array(x_features)


def train(x_train, y_train, *args, **kwargs):
    classifier = BaggingClassifier(tree.DecisionTreeClassifier(),
                                   n_estimators=100,
                                   max_samples=0.5, max_features=0.5,
                                   n_jobs=-1,
                                   verbose=1)
    classifier.fit(x_train, y_train)
    return classifier


def test(model, x_test):
    return model.predict(x_test)
