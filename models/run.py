import numpy as np
import torch

from models import m8_deepconvlstm, m9_dclstm_att
from models.utils import dataset as u_dataset, evaluate as u_test
from models.utils.common import set_logger
from models.utils.dataset import Dataset

np.random.seed(42)
torch.manual_seed(42)


def run(model_choice, ds, model_name, hp, prefold):
    (x_train, x_test, y_train, y_test), class_count = u_dataset.load_split_dataset(ds, prefold)
    m = model_choice.train(x_train, y_train, class_count, **hp)
    u_test.create_results(m, model_choice.test, x_train, x_test, y_train, y_test, class_count, ds, model_name)


if __name__ == '__main__':
    for dataset in [
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
        for prefold in Dataset.get_prefold_range(dataset):
            set_logger('deepconvlstm', dataset, 0, {'lr': 0.001, 'folds': None}, prefold)
            run(m8_deepconvlstm, dataset, 'deepconvlstm', {'lr': 0.001, 'folds': None}, prefold)

            set_logger('dclstm_att', dataset, 0, {'lr': 0.0001, 'folds': None}, prefold)
            run(m9_dclstm_att, dataset, 'dclstm_att', {'lr': 0.0001, 'folds': None}, prefold)
