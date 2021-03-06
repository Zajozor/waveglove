import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from models import m1_baseline
from models.utils import metrics as u_metrics
from models.utils.dataset import Dataset, load_dataset

sns.set(font_scale=1.2)

if __name__ == '__main__':
    model = m1_baseline
    df = pd.DataFrame()

    datasets = [Dataset.WAVEGLOVE_MULTI, Dataset.WAVEGLOVE_SINGLE, Dataset.UWAVE,
                Dataset.OPPORTUNITY, Dataset.PAMAP2, Dataset.SKODA, Dataset.MHEALTH]
    dsnames = ['WaveGlove-multi', 'WaveGlove-single', 'uWave', 'OPPORTUNITY', 'PAMAP2', 'Skoda', 'MHEALTH']

    for dataset, dsname in zip(datasets, dsnames):
        print('Evaluating dataset', dsname, end='')
        accs, recalls, f1s = [], [], []
        x, y, class_count = load_dataset(dataset)

        for random_state in range(100):
            print('o' if random_state % 10 == 0 else '.', end='')
            (x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.1, random_state=random_state)
            x_train, x_test = model.feature_extraction(x_train), model.feature_extraction(x_test)
            trained_model = model.train(x_train, y_train, class_count)

            y_hat = model.test(trained_model, x_test)
            acc, recall, f1 = u_metrics.get_metrics(y_hat, y_test)
            accs.append(acc)
            recalls.append(recall)
            f1s.append(f1)
        print()

        df = df.append({'Metric value': accs, 'Evaluation metric': 'Accuracy', 'Dataset': dsname},
                       ignore_index=True)
        df = df.append({'Metric value': recalls, 'Evaluation metric': 'Recall', 'Dataset': dsname},
                       ignore_index=True)
        df = df.append({'Metric value': f1s, 'Evaluation metric': 'Macro F1 score', 'Dataset': dsname},
                       ignore_index=True)

    df = df.explode('Metric value')

    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    sns.boxplot(y='Metric value', x='Dataset', hue='Evaluation metric', data=df, ax=ax)
    plt.subplots_adjust(left=0.05, right=0.99, top=1)
    ax.legend(loc='lower left')
    df.to_csv('../plots/results_baseline_df.csv', index=False)
    for i, dsname in enumerate(dsnames):
        print(dsname, ':', 'Acc',
              f"{df[(df['Dataset'] == dsname) & (df['Evaluation metric'] == 'Accuracy')]['Metric value'].mean():.3f}",
              'Recall',
              f"{df[(df['Dataset'] == dsname) & (df['Evaluation metric'] == 'Recall')]['Metric value'].mean():.3f}",
              'F1',
              f"{df[(df['Dataset'] == dsname) & (df['Evaluation metric'] == 'Macro F1 score')]['Metric value'].mean():.3f}"
              )

    fig.savefig('../plots/results_baseline.png')
    fig.show()
