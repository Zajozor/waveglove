import numpy as np
import torch
from matplotlib import pyplot as plt

from models import m2_trees, m6_transformer
from models.utils import dataset as u_dataset, metrics as u_metrics, plot as u_plot
from models.utils.common import set_logger
from models.utils.dataset import Dataset
from models.utils.hparams import iter_hparams
import seaborn as sns
torch.manual_seed(42)
dataset = Dataset.WAVEGLOVE_MULTI
sns.set(font_scale=1.9)


def run(chosen_model, hp=None):
    fingers = hparams.pop('fingers')
    columns = sum(list(map(lambda f: list(range(6 * f, 6 * (f + 1))), fingers)), [])

    x_train = np.array(x_traing[:, :, columns])
    x_test = np.array(x_testg[:, :, columns])

    x_train = chosen_model.feature_extraction(x_train)
    x_test = chosen_model.feature_extraction(x_test)

    m = chosen_model.train(x_train, y_traing, class_count, **hp)
    fig, ax = plt.subplots(1, 1, figsize=(1.3*class_count, 1.3*class_count))
    # Actual prediction results
    y_hat = chosen_model.test(m, x_test)
    cm = u_metrics.create_confusion_matrix(class_count, y_hat, y_testg)
    u_plot.plot_confusion_matrix(cm, ax=ax, cbar=False)

    class_names = ['Null', 'Hand sw. left', 'Hand sw. right', 'Pinch in', 'Pinch out',
                   'Thumb d. tap', 'Grab', 'Ungrab', 'Page flip', 'Peace', 'Metal']
    class_namese = ['']*10
    ax.set_xticklabels(class_names, rotation=90)
    ax.invert_yaxis()
    ax.set_yticklabels(class_names, rotation=0)
    fig.show()


if __name__ == '__main__':
    # for test_part in [0.15, 0.5, 0.8, 0.95, 0.99]:
    for test_part in [0.95]:
        (x_traing, x_testg, y_traing, y_testg), class_count = \
            u_dataset.load_split_dataset(dataset, None, test_size=test_part)

        for model, name, hparams_sweep in [
            (m2_trees, 'trees', {
                'fingers': [
                    (0,),
                    (1,),
                    (2,),
                    (3,),
                    (4,),

                    (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),

                    (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4),
                    (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4),

                    (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 3, 4), (0, 2, 3, 4), (1, 2, 3, 4),

                    (0, 1, 2, 3, 4)
                ],
            }),
            (m6_transformer, 'tfm1', {
                'fingers': [
                    (0,), (1,), (2,), (3,), (4,),

                    (0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),

                    (0, 1, 2), (0, 1, 3), (0, 1, 4), (0, 2, 3), (0, 2, 4),
                    (0, 3, 4), (1, 2, 3), (1, 2, 4), (1, 3, 4), (2, 3, 4),

                    (0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 3, 4), (0, 2, 3, 4), (1, 2, 3, 4),

                    (0, 1, 2, 3, 4)
                ],
                'sensor_embed_dim': [32],
                'dropout': [0.2],
                'encoder_heads': [8],
                'encoder_hidden': [128],
                'encoder_layers': [4],
                'temporal_attention_heads': [4],
                'lr': [0.001],
                'folds': [None],
            }),
        ]:
            for hp_id, hparams in enumerate(iter_hparams(hparams_sweep)):
                set_logger(f'attrib-{name}-{test_part}', dataset, hp_id, hparams)
                run(model, hparams)

# Just so these are not code-styled away
t = [m2_trees, m6_transformer]
