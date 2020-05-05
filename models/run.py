import models.m1_baseline as m1
import models.m2_randomforest as m2
import models.m3_linear as m3
import models.m4_cnn as m4
from models.utils import dataset as u_dataset, test as u_test
from models.utils.dataset import Dataset
from models.utils.common import set_writer
import torch

torch.manual_seed(42)


def run(model_choice, dataset=Dataset.WAVEGLOVE_MULTI, model_name='Some model'):
    (x_train, x_test, y_train, y_test), class_count = \
        u_dataset.load_split_dataset(dataset)

    x_train, x_test = model_choice.feature_extraction(x_train), \
        model_choice.feature_extraction(x_test)

    model = model_choice.train(x_train, y_train, class_count)

    u_test.create_results(model, model_choice.test,
                          x_train, x_test, y_train, y_test, class_count,
                          dataset, model_name)


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
        for model, name in [
            # (m1, 'baseline'),
            # (m2, 'randomforest'),
            (m3, 'basenn'),
            (m4, 'basiccnn'),
        ]:
            print('Running', name, 'on', dataset.value)
            set_writer(name, dataset)
            run(model, dataset, name)

# TODO add confidence intervals to runs
# TODO labels k zvysnym datasetom s wsd
