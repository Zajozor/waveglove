import torch

from models import m1_baseline, m2_randomforest, m3_linear, m4_cnn, m5_lstm
from models.utils import dataset as u_dataset, evaluate as u_test
from models.utils.common import set_logger
from models.utils.dataset import Dataset
from models.utils.hparams import iter_hparams

torch.manual_seed(42)


def run(model_choice, ds=Dataset.WAVEGLOVE_MULTI,
        model_name='Some model', hp=None):
    if hp is None:
        hp = {}

    (x_train, x_test, y_train, y_test), class_count = \
        u_dataset.load_split_dataset(ds)

    x_train, x_test = model_choice.feature_extraction(x_train), \
        model_choice.feature_extraction(x_test)

    m = model_choice.train(x_train, y_train, class_count, **hp)

    u_test.create_results(m, model_choice.test,
                          x_train, x_test, y_train, y_test, class_count,
                          ds, model_name)


if __name__ == '__main__':
    # Configure this
    for dataset in [
        Dataset.WAVEGLOVE_SINGLE,
        Dataset.WAVEGLOVE_MULTI,
        Dataset.UWAVE,
        Dataset.OPPORTUNITY,
        Dataset.PAMAP2,
        Dataset.SKODA,
        Dataset.MHEALTH,
    ]:
        for model, name, hparams_sweep in [
            # (m1, 'baseline'),
            # (m2, 'randomforest'),
            (m3_linear, 'nn1', {
                'l1': [64],#, 64, 128, 256, 512],
                'l2': [128],#, 64, 128, 256, 512],
                'lr': [0.001],#, 0.01, 0.001],
                'folds': [2],
            }),
            # (m4_cnn, 'cnn1', {
            #     'filters1': [9, 18, 36],
            #     'kernel_width1': [13, 26],
            #     'filters2': [9, 18, 36],
            #     'kernel_width2': [4, 8, 12],
            #     'filters3': [9, 18, 36],
            #     'kernel_width3': [4, 8, 12],
            #     'lr': [0.1, 0.01, 0.001],
            # }),
            (m4_cnn, 'cnn1', {
                'filters1': [18],
                'kernel_width1': [13],
                'filters2': [36],
                'kernel_width2': [7],
                'filters3': [24],
                'kernel_width3': [7],
                'lr': [0.001],
                'folds': [2],
            })
        ]:
            for hparams in iter_hparams(hparams_sweep):
                print('Running', name, 'on', dataset.value, 'with', hparams)
                set_logger(name, dataset)
                run(model, dataset, name, hparams)

# TODO add confidence intervals to runs
# TODO labels k zvysnym datasetom s wsd

# Just so these are not code-styled away
t = [m1_baseline, m2_randomforest, m3_linear, m4_cnn, m5_lstm]
