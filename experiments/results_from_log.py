# {dt} | {model_name:12} | {hp_id:3} | {prefold:2} | {dataset.value:30} | {hparams} || TAcc: {acc_train:.3f} | TRec: {recall_train:.3f} | TF1: {f1_train:.3f} || Acc: {acc:.3f} | Rec: {recall:.3f} | F1: {f1:.3f}

import re
from datetime import datetime
import pandas as pd

pattern = re.compile(r'([^|]+) \| +(\w+) +\| +(\d+) +\| +(\d*) +\| +([^ ]*) +\| ({[^}]*\}) \|\|'
                     r' +TAcc: +([\d.]+) +\| TRec: +([\d.]+) +\| TF1: +([\d.]+) +\|\|'
                     r' +Acc: +([\d.]+) +\| Rec: +([\d.]+) +\| F1: +([\d.]+) *\n')


def print_nn_results(df, datasets, dsnames):
    # hparam search
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
    # hparam search
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


def print_cnn_average_perf(df, datasets, dsnames):
    section = df[(df['Model name'] == 'cnn1') & df['Dataset'].isin(datasets)]
    print(section.groupby('Dataset').mean()[['Accuracy', 'Recall', 'F1 score']])
    print('Hparams:', section.iloc[0]['Hyperparameters'])


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

    print(' --- NN 1 RESULTS --')
    print_nn_results(df, datasets, dsnames)
    print()
    print(' --- CNN 1 RESULTS --')
    print_cnn_results(df, datasets, dsnames)
    print()
    print(' --- LOTO CNN 1 RESULTS --')
    print_cnn_average_perf(df, lotodatasets, lotodsnames)
    print()

