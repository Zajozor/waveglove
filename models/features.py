import numpy as np
import pandas as pd


def channel_correlation(instance):
    df = pd.DataFrame(instance)
    corr = df.corr().fillna(0)
    tr_upper = np.triu(np.ones(corr.shape), k=1).astype('bool')
    return corr.where(tr_upper).stack().to_numpy()


def channel_mean(instance):
    return np.mean(instance, axis=0)
