import re
from datetime import datetime
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

pattern = re.compile(r'([^|]+) \| +([^|]+) +\| +(\d+) +\| +(\d*) +\| +([^ ]*) +\| ({[^}]*\}) \|\|'
                     r' +TAcc: +([\d.]+) +\| TRec: +([\d.]+) +\| TF1: +([\d.]+) +\|\|'
                     r' +Acc: +([\d.]+) +\| Rec: +([\d.]+) +\| F1: +([\d.]+) *\n')


def plot_metrics(df, dset, dsname, models, model_names):
    section = df[df['Dataset'] == dset]
    section = section.assign(**{
        'Model name': section['Model name'].map(lambda x: dict(zip(models, model_names))[x[:x.rfind('-')]]),
        # replace(dict(zip(models, model_names))),
        'Count of sensors': section['Hyperparameters'].map(lambda x: len(x['fingers'])),
        'Train set size': section['Model name'].map(
            lambda x: str(100 - int((x.split('.')[-1] + '00')[:2])) + '%'
        ),
    })
    section = section.assign(**{
        'Model, Training set size': section['Model name'] + ', ' + section['Train set size'],
    })

    fig, axs = plt.subplots(1, 1, figsize=(6, 8))

    sns.lineplot(x='Count of sensors', y='Accuracy', hue='Model, Training set size', data=section, ax=axs)
    axs.set(xticks=[1, 2, 3, 4, 5], ylim=(0.1, 1), yticks=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1])

    fig.show()


if __name__ == '__main__':
    with open('attrib-trees-single.log', 'r') as f:
        lines = f.readlines()

    df = pd.DataFrame()
    for line in lines:
        match = pattern.search(line)
        if match is None:
            print('!! Skipping', line[:20])
            continue
        date, model_name, hp_id, prefold, dataset, hparams, tacc, trec, tf1, acc, rec, f1 = match.groups()
        date = datetime.strptime(date, "%b%d-%H:%M:%S")
        hp_id = int(hp_id)
        prefold = int(prefold) if prefold else 0
        hparams = eval(hparams)  # RIP
        tacc, trec, tf1, acc, rec, f1 = map(float, (tacc, trec, tf1, acc, rec, f1))

        df = df.append({
            'Date': date,
            'Model name': model_name,
            'Hyperparameter ID': hp_id,
            'Prefold ID': prefold,
            'Dataset': dataset,
            'Hyperparameters': hparams,
            'Training accuracy': tacc,
            'Training recall': trec,
            'Training F1 score': tf1,
            'Accuracy': acc,
            'Recall': rec,
            'F1 score': f1,
        }, ignore_index=True)

    models = ['attrib-trees', 'attrib-tfm1']
    model_names = ['Bagging Decision Tree', 'Self-attention with sensor embedding']
    plot_metrics(df, 'waveglove_single', 'WaveGlove-single', models, model_names)
    # plot_metrics(df, 'waveglove_multi', 'WaveGlove-multi', models, model_names)
