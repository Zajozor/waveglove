import models.m1_baseline as m1
import models.m2_randomforest as m2
from models.utils import dataset as u_dataset, test as u_test
from models.utils.dataset import Dataset


def run(model_choice, dataset=Dataset.WAVEGLOVE_MULTI):
    (x_train, x_test, y_train, y_test), class_count = \
        u_dataset.load_split_dataset(dataset)

    x_train, x_test = model_choice.feature_extraction(x_train), \
        model_choice.feature_extraction(x_test)

    model = model_choice.train(x_train, y_train, class_count)

    u_test.create_results(model, model_choice.test,
                          x_train, x_test, y_train, y_test, class_count,
                          dataset)


if __name__ == '__main__':
    # Configure this
    run(m2, Dataset.UWAVE)

