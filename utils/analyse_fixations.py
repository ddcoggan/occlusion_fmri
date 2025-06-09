# Created by David Coggan on 2024 07 18

import pandas as pd
import os.path as op
import numpy as np
from config import PROJ_DIR

PPD = 40.4
data_path = op.join(PROJ_DIR, 'data/in_vivo/behavioral/fMRI_fixation/data.csv')
data = pd.read_csv(data_path)


def remove_outliers(df):
    df = df[np.abs(df.x - df.x.mean()) < 3 * df.x.std()]
    df = df[np.abs(df.y - df.y.mean()) < 3 * df.y.std()]
    return df


def get_sigma(df):
    sigma_x = np.mean(np.abs(df.x - df.x.mean())) / PPD
    sigma_y = np.mean(np.abs(df.y - df.y.mean())) / PPD
    sigma = np.mean(np.sqrt((df.x - df.x.mean())**2 +
                            (df.y - df.y.mean())**2)) / PPD
    return pd.DataFrame(dict(sigma_x=[sigma_x],
                        sigma_y=[sigma_y],
                        sigma=[sigma]))


data_clean = (data
    .groupby(['subject', 'attn'])
    .apply(remove_outliers)
    .reset_index(drop=True))

sigma_data = (data_clean
    .groupby(['subject', 'attn', 'block', 'object', 'occluder'])
    .apply(get_sigma))

sigma_summary = sigma_data.groupby(['attn']).agg('mean').reset_index()
print(sigma_summary)

