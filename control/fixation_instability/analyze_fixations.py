# Created by David Coggan on 2024 07 18
# attn = on, participants performed object discrimination task
# attn = off, participants performed letter discrimination task

import pandas as pd
import numpy as np

PPD = 40.4
data = pd.read_csv('gaze_data.csv')

def remove_outliers(df):
    df = df[np.abs(df.x - df.x.mean()) < 3 * df.x.std()]
    df = df[np.abs(df.y - df.y.mean()) < 3 * df.y.std()]
    return df


def get_sigma(df):
    sigma = np.mean(np.sqrt((df.x - df.x.mean())**2 +
                            (df.y - df.y.mean())**2)) / PPD
    return pd.DataFrame(dict(sigma=[sigma]))


data_clean = (data
    .groupby(['subject', 'attn'])
    .apply(remove_outliers)
    .reset_index(drop=True))

sigma_data = (data_clean
    .groupby(['subject', 'attn', 'block', 'object', 'occluder'])
    .apply(get_sigma))

sigma_summary = sigma_data.groupby(['attn']).agg('mean').reset_index()
sigma_summary = pd.concat([sigma_summary, pd.DataFrame({
    'attn': ['all'], 'sigma': sigma_summary.sigma.mean()})])
print(sigma_summary)
sigma_summary.to_csv('results.csv', index=False)

