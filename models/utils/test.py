from models.utils import metrics as u_metrics, plot as u_plot
import matplotlib.pyplot as plt


def create_results(model, test_f,
                   x_train, x_test, y_train, y_test, class_count,
                   dataset):
    from models.utils.common import writer

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

    # To validate we can overfit
    y_hat_train = test_f(model, x_train)
    cm_train = u_metrics.create_confusion_matrix(class_count, y_hat_train, y_train)
    u_plot.plot_confusion_matrix(cm_train, title=f'{dataset.value} on training', ax=ax1)

    # Actual prediction results
    y_hat = test_f(model, x_test)
    cm = u_metrics.create_confusion_matrix(class_count, y_hat, y_test)
    u_plot.plot_confusion_matrix(cm, title=f'{dataset.value} on test', ax=ax2)

    acc, recall, f1 = u_metrics.get_metrics(y_hat, y_test)
    print(f'Accuracy: {acc}, Recall: {recall}, F1: {f1}')

    fig.show()
    writer.add_scalar('accuracy', acc)
    writer.add_scalar('recall', recall)
    writer.add_scalar('f1', f1)
    writer.add_figure('confusion_matrix', fig)


