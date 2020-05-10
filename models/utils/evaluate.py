import matplotlib.pyplot as plt

from models.utils import metrics as u_metrics, plot as u_plot
from models.utils.common import get_logger


def create_results(model, test_f,
                   x_train, x_test, y_train, y_test, class_count,
                   dataset, model_name):
    logger = get_logger()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # To validate we can overfit
    y_hat_train = test_f(model, x_train)
    cm_train = u_metrics.create_confusion_matrix(class_count, y_hat_train, y_train)
    u_plot.plot_confusion_matrix(cm_train, title=f'Training', ax=ax1)

    # Actual prediction results
    y_hat = test_f(model, x_test)
    cm = u_metrics.create_confusion_matrix(class_count, y_hat, y_test)
    u_plot.plot_confusion_matrix(cm, title=f'Test', ax=ax2)

    fig.suptitle(f'Model: {model_name}, Dataset: {dataset.value}', fontsize=16)
    fig.subplots_adjust(top=.9)
    logger.experiment.add_figure('confusion_matrix', fig)

    acc, recall, f1 = u_metrics.get_metrics(y_hat, y_test)
    logger.log_metrics({
        'perf/accuracy': acc, 'perf/recall': recall, 'perf/f1': f1
    })
    print(f'Accuracy: {acc}, Recall: {recall}, F1: {f1}')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    u_plot.plot_class_histogram(y_train, ax=ax1, title='Class distribution on train')
    u_plot.plot_class_histogram(y_test, ax=ax2, title='Class distribution on test')
    fig.suptitle(f'Model: {model_name}, dataset: {dataset.value}', fontsize=16)
    fig.subplots_adjust(top=.9)
    logger.experiment.add_figure('y_dist', fig)

    logger.save()
