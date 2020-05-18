import re
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set(font_scale=1.4)

# Line format:
# {dt} | {model_name:12} | {hp_id:3} | {prefold:2} | {dataset.value:30} | {hparams} || TAcc: {acc_train:.3f} | TRec: {recall_train:.3f} | TF1: {f1_train:.3f} || Acc: {acc:.3f} | Rec: {recall:.3f} | F1: {f1:.3f}
pattern = re.compile(r'([^|]+) \| +(\w+) +\| +(\d+) +\| +(\d*) +\| +([^ ]*) +\| ({[^}]*\}) \|\|'
                     r' +TAcc: +([\d.]+) +\| TRec: +([\d.]+) +\| TF1: +([\d.]+) +\|\|'
                     r' +Acc: +([\d.]+) +\| Rec: +([\d.]+) +\| F1: +([\d.]+) *\n')


def print_nn_results(df, datasets, dsnames):
    section = df[(df['Model name'] == 'nn1') & df['Dataset'].isin(datasets)]
    for i, ds in enumerate(datasets):
        subsection = section[section['Dataset'] == ds]
        top = subsection.loc[subsection['F1 score'].idxmax()]

        print(f'{dsnames[i]:17} & ', end='')
        print(' & '.join(map(lambda x: f'{x:6}', [
            top['Accuracy'],
            top['Recall'],
            top['F1 score'],
            top['Hyperparameters']['l1'],
            top['Hyperparameters']['l2'],
            top['Hyperparameters']['lr']
        ])))


def print_cnn_results(df, datasets, dsnames):
    section = df[(df['Model name'] == 'cnn1') & df['Dataset'].isin(datasets)]
    for i, ds in enumerate(datasets):
        subsection = section[section['Dataset'] == ds]
        top = subsection.loc[subsection['F1 score'].idxmax()]

        print(f'{dsnames[i]:17} & ', end='')
        print(' & '.join(map(lambda x: f'{x:6}', [
            top['Accuracy'],
            top['Recall'],
            top['F1 score'],
            top['Hyperparameters']['filters1'],
            top['Hyperparameters']['kernel_width1'],
            top['Hyperparameters']['filters2'],
            top['Hyperparameters']['kernel_width2'],
            top['Hyperparameters']['filters3'],
            top['Hyperparameters']['kernel_width3'],
            top['Hyperparameters']['lr']
        ])))


def print_average_perf(df, datasets, model_name):
    section = df[(df['Model name'] == model_name) & df['Dataset'].isin(datasets)]
    grouped = section.groupby('Dataset').mean()[['Accuracy', 'Recall', 'F1 score']]
    print('model', model_name)
    print(grouped)
    print('Hparams:', section.iloc[0]['Hyperparameters'])
    print(grouped.mean())
    print()


def plot_metrics(df, datasets, ds_names, models, model_names):
    section = df[(df['Model name'].isin(models)) & df['Dataset'].isin(datasets)]
    section = section.assign(**{
        'Dataset': section['Dataset'].replace(dict(zip(datasets, ds_names))),
        'Model name': section['Model name'].replace(dict(zip(models, model_names))),
    })
    fig, axs = plt.subplots(3, 1, figsize=(10.63, 15))
    # for i, dsn in enumerate(ds_names):
    #     part = section[section['Dataset'] == dsn]
    #     sns.boxplot(x='Dataset', y='Accuracy', hue='Model name', data=part, ax=axs[0][i])
    #     sns.boxplot(x='Dataset', y='Recall', hue='Model name', data=part, ax=axs[1][i])
    #     sns.boxplot(x='Dataset', y='F1 score', hue='Model name', data=part, ax=axs[2][i])

    sns.boxplot(x='Dataset', y='Accuracy', hue='Model name', data=section, ax=axs[0])
    sns.boxplot(x='Dataset', y='Recall', hue='Model name', data=section, ax=axs[1])
    sns.boxplot(x='Dataset', y='F1 score', hue='Model name', data=section, ax=axs[2])

    # for axr in axs:
    #     for ax in axr:
    #         ax.get_legend().set_visible(False)
    #         ax.set(xlabel='')
    #         # ax.set(ylim=(0.6, 1))
    #     for ax in axr[1:]:
    #         ax.set(ylabel ='')
    # for axr in axs[:-1]:
    #     for ax in axr:
    #         ax.set(xticks=[])
    fig.subplots_adjust(hspace=0)
    fig.show()


def plot_metrics_b(df, datasets, ds_names, models, model_names):
    section = df[(df['Model name'].isin(models)) & df['Dataset'].isin(datasets)]
    section = section.assign(**{
        'Dataset': section['Dataset'].replace(dict(zip(datasets, ds_names))),
        'Model name': section['Model name'].replace(dict(zip(models, model_names))),
    })
    section = section.groupby(['Dataset', 'Model name']).mean().reset_index()

    for metric in ['Accuracy', 'Recall', 'F1 score']:
        matrix = np.empty((len(models), len(datasets)), dtype=np.float32)
        for i, md in enumerate(model_names):
            for j, ds in enumerate(ds_names):
                matrix[i][j] = section.loc[
                    (section['Dataset'] == ds) & (section['Model name'] == md)
                    ].iloc[0][metric]
        annot = [[(f'{item:.3f}' if metric == 'F1 score' else f'{item:.2%}') for item in row] for row in matrix]

        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        sns.heatmap(matrix, vmin=0.9 if metric == 'Accuracy' else 0.7, ax=ax, fmt='', annot=annot, cmap='YlGnBu')
        ax.set_xticklabels(ds_names, rotation=30)
        ax.set_yticklabels(model_names, rotation=0)
        ax.set_title(metric)

        fig.show()


if __name__ == '__main__':
    with open('use.log', 'r') as f:
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

    datasets = ['waveglove_single', 'waveglove_multi', 'uwave', 'opportunity', 'pamap2', 'skoda', 'mhealth']
    dsnames = ['WaveGlove-single', 'WaveGlove-multi', 'uWave', 'OPPORTUNITY', 'PAMAP2', 'Skoda', 'MHEALTH (FNOW)']

    lotodatasets = ['jordao_etal/LOTO_MHEALTH', 'jordao_etal/LOTO_USCHAD', 'jordao_etal/LOTO_UTD-MHAD1_1s',
                    'jordao_etal/LOTO_UTD-MHAD2_1s', 'jordao_etal/LOTO_WHARF', 'jordao_etal/LOTO_WISDM']
    lotodsnames = ['MHEALTH', 'USC-HAD', 'UTD-MHAD1', 'UTD-MHAD2', 'WHARF', 'WISDM']

    losodatasets = ['jordao_etal/LOSO_MHEALTH', 'jordao_etal/LOSO_USCHAD', 'jordao_etal/LOSO_UTD-MHAD1_1s',
                    'jordao_etal/LOSO_UTD-MHAD2_1s', 'jordao_etal/LOSO_WHARF', 'jordao_etal/LOSO_WISDM']
    losodsnames = ['MHEALTH', 'USC-HAD', 'UTD-MHAD1', 'UTD-MHAD2', 'WHARF', 'WISDM']

    # print(' --- NN 1 RESULTS --')
    # print_nn_results(df, datasets, dsnames)
    # print()
    # print(' --- CNN 1 RESULTS --')
    # print_cnn_results(df, datasets, dsnames)
    # print()
    print(' --- TREES RESULTS --')
    print_average_perf(df, lotodatasets, 'trees')
    print_average_perf(df, losodatasets, 'trees')
    print()

    print(' --- CNN 1 RESULTS --')
    print_average_perf(df, lotodatasets, 'cnn1')
    print_average_perf(df, losodatasets, 'cnn1')
    print()

    print(' --- LSTM 1 RESULTS --')
    print_average_perf(df, lotodatasets, 'lstm1')
    print_average_perf(df, losodatasets, 'lstm1')
    print()

    print(' --- LSTM ATT RESULTS --')
    print_average_perf(df, lotodatasets, 'lstmatt')
    print_average_perf(df, losodatasets, 'lstmatt')
    print()

    print(' --- DEEPCONVLSTM RESULTS --')
    print_average_perf(df, lotodatasets, 'deepconvlstm')
    print_average_perf(df, losodatasets, 'deepconvlstm')
    print()

    print(' --- DCLSTM_ATT RESULTS --')
    print_average_perf(df, lotodatasets, 'dclstm_att')
    print_average_perf(df, losodatasets, 'dclstm_att')
    print()

    print(' --- TFM 1 RESULTS --')
    print_average_perf(df, lotodatasets, 'tfm1')
    print_average_perf(df, losodatasets, 'tfm1')
    print()

    print(' --- FNOW RESULTS ---')
    print_average_perf(df, datasets, 'nn1')
    print_average_perf(df, datasets, 'cnn1')
    print_average_perf(df, datasets, 'lstm1')
    print_average_perf(df, datasets, 'deepconvlstm')
    print_average_perf(df, datasets, 'dclstm_att')
    # print_average_perf(df, datasets, 'tfm1')
    print()

    datasets1 = ['waveglove_single', 'waveglove_multi', 'uwave', 'opportunity', 'pamap2', 'skoda', 'mhealth']
    dsnames1 = ['WaveGlove-single', 'WaveGlove-multi', 'uWave', 'OPPORTUNITY', 'PAMAP2', 'Skoda', 'MHEALTH (FNOW)']

    models = ['nn1', 'cnn1', 'lstm1', 'lstmatt', 'deepconvlstm', 'tfm1']
    model_names = ['Linear NN', 'CNN', 'LSTM', 'LSTM w. self-att.', 'DeepConvLSTM', 'Self-att. w. sensor-emb.']
    plot_metrics_b(df, datasets1, dsnames1, models, model_names)
