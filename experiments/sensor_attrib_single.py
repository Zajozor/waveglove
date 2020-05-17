import torch
from sklearn.preprocessing import normalize

from models import m1_baseline, m2_trees, m3_linear, m4_cnn, m5_lstm, m6_transformer
from models.utils import dataset as u_dataset, evaluate as u_test
from models.utils.common import set_logger
from models.utils.dataset import Dataset
from models.utils.hparams import iter_hparams
import numpy as np

torch.manual_seed(42)
dataset = Dataset.WAVEGLOVE_MULTI

def run(chosen_model, model_name='Some model', hp=None):

    fingers = hparams.pop('fingers')
    columns = sum(list(map(lambda f: list(range(6 * f, 6 * (f + 1))), fingers)), [])

    (x_train, x_test, y_train, y_test), class_count = \
        u_dataset.load_split_dataset(dataset, None, test_size=0.95)

    x_train = np.array(x_train[:, :, columns])
    x_test = np.array(x_test[:, :, columns])

    x_train = chosen_model.feature_extraction(x_train)
    x_test = chosen_model.feature_extraction(x_test)

    m = chosen_model.train(x_train, y_train, class_count, **hp)

    u_test.create_results(m, chosen_model.test,
                          x_train, x_test, y_train, y_test, class_count,
                          dataset, model_name)


if __name__ == '__main__':
    for model, name, hparams_sweep in [
        (m2_trees, 'trees', {
            'fingers': [
                (0,), (1,), (2,), (3,), (4,),

                (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),

                (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4),
                (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4),

                (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 3, 4), (0, 2, 3, 4), (1, 2, 3, 4),

                (0, 1, 2, 3, 4)
            ]
        }),
    ]:
        for hp_id, hparams in enumerate(iter_hparams(hparams_sweep)):
            set_logger(f'attrib-{name}', dataset, hp_id, hparams)
            run(model, name, hparams)

# Just so these are not code-styled away
t = [m1_baseline, m2_trees, m3_linear, m4_cnn, m5_lstm, m6_transformer]
