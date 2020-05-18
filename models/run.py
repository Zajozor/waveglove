import torch
import numpy as np

from models import m1_baseline, m2_trees, m3_linear, m4_cnn, m5_lstm, m6_transformer, m7_lstm_att, m8_deepconvlstm, \
    m9_dclstm_att
from models.utils import dataset as u_dataset, evaluate as u_test
from models.utils.common import set_logger
from models.utils.dataset import Dataset
from models.utils.hparams import iter_hparams

np.random.seed(42)
torch.manual_seed(42)


def run(model_choice, ds=Dataset.WAVEGLOVE_MULTI,
        model_name='Some model', hp=None, prefold=None):
    if hp is None:
        hp = {}

    (x_train, x_test, y_train, y_test), class_count = u_dataset.load_split_dataset(ds, prefold)

    x_train, x_test = \
        model_choice.feature_extraction(x_train), model_choice.feature_extraction(x_test)

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

        Dataset.J_LOSO_MHEALTH,
        Dataset.J_LOSO_USCHAD,
        Dataset.J_LOSO_UTD_MHAD1_1s,
        Dataset.J_LOSO_UTD_MHAD2_1s,
        Dataset.J_LOSO_WHARF,
        Dataset.J_LOSO_WISDM,

        Dataset.J_LOTO_MHEALTH,
        Dataset.J_LOTO_USCHAD,
        Dataset.J_LOTO_UTD_MHAD1_1s,
        Dataset.J_LOTO_UTD_MHAD2_1s,
        Dataset.J_LOTO_WHARF,
        Dataset.J_LOTO_WISDM,

    ]:
        for model, name, hparams_sweep in [
            (m1_baseline, 'baseline', {}),
            (m2_trees, 'trees', {}),

            (m3_linear, 'nn1', {
                'l1': [256],
                'l2': [128],
                'lr': [0.001],
                'folds': [None],
            }),

            (m4_cnn, 'cnn1', {
                'filters1': [18],
                'kernel_width1': [12],
                'filters2': [36],
                'kernel_width2': [13],
                'filters3': [24],
                'kernel_width3': [12],
                'lr': [0.001],
                'folds': [None],
            }),

            (m5_lstm, 'lstm1', {
                'layer_count': [1],
                'pre_length': [60],
                'hidden_size': [256],
                'drop_prob': [0.1],
                'lr': [0.001],
                'folds': [None],
            }),

            (m6_transformer, 'tfm1', {
                'sensor_embed_dim': [32],
                'dropout': [0.2],

                'encoder_heads': [8],  # Must divide sensor embed
                'encoder_hidden': [128],
                'encoder_layers': [4],

                'temporal_attention_heads': [4],  # Must divide sensor embed

                'lr': [0.001],
                'folds': [None],
            }),

            (m7_lstm_att, 'lstmatt', {
                'layer_count': [1],
                'hidden_size': [64],
                'drop_prob': [0.2],
                'lr': [0.001],
                'folds': [None],
            }),
            (m8_deepconvlstm, 'deepconvlstm', {
                'lr': [0.001],
                'folds': [None],
            }),
            (m9_dclstm_att, 'dclstm_att', {
                'lr': [0.0001],
                'folds': [None],
            })
        ]:
            for prefold in Dataset.get_prefold_range(dataset):
                for hp_id, hparams in enumerate(iter_hparams(hparams_sweep)):
                    set_logger(name, dataset, hp_id, hparams, prefold)
                    run(model, dataset, name, hparams, prefold)

# Just so these are not code-styled away
t = [m1_baseline, m2_trees, m3_linear, m4_cnn, m5_lstm, m6_transformer, m7_lstm_att, m8_deepconvlstm, m9_dclstm_att]
