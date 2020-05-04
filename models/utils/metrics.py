import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score


def create_confusion_matrix(class_count, y_hat, y):
    matrix = np.zeros((class_count, class_count), dtype=np.int)
    for i in range(y_hat.shape[0]):
        matrix[y[i]][y_hat[i]] += 1
    return matrix


def get_metrics(y_hat, y):
    return accuracy_score(y, y_hat), \
           recall_score(y, y_hat, average='macro'), \
           f1_score(y, y_hat, average='macro')
