import os
import numpy as np
import pandas as pd
from urllib import request
import shutil
import zipfile
import json
from generate_mask import generate_mask

DATA_DIR = 'datasets'

def train_test_split(dataname, ratio = 0.7, mask_prob = 0.3):
    data_dir = f'{DATA_DIR}/{dataname}'
    path = f'{DATA_DIR}/{dataname}/data.csv'
    info_path = f'{DATA_DIR}/Info/{dataname}.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    cat_idx = info['cat_col_idx']
    num_idx = info['num_col_idx']

    data_df = pd.read_csv(path)
    total_num = data_df.shape[0]

    if len(cat_idx) == 0:
        data_values = data_df.values[:, :-1].astype(np.float32)

        nan_idx = np.isnan(data_values).nonzero()[0]

        keep_idx = list(set(np.arange(data_values.shape[0])) - set(list(nan_idx)))
        keep_idx = np.array(keep_idx)
    else:
        keep_idx = np.arange(total_num)

    num_train = int(keep_idx.shape[0] * ratio)
    num_test = total_num - num_train
    seed = 1234

    np.random.seed(seed)
    np.random.shuffle(keep_idx)

    train_idx = keep_idx[:num_train]
    test_idx = keep_idx[-num_test:]

    train_df = data_df.loc[train_idx]
    test_df = data_df.loc[test_idx]        

    train_path = f'{data_dir}/train.csv'
    test_path = f'{data_dir}/test.csv'

    train_df.to_csv(train_path, index = False)
    test_df.to_csv(test_path, index = False)

    print(f'Spliting Trainig and Testing data for {dataname} is done.')
    print(f'Training data shape: {train_df.shape}, Testing data shape: {test_df.shape}')
    print(f'Training data saved at {train_path}, Testing data saved at {test_path}.')
    

   

if __name__ == '__main__':

    name = 'california'

    for mask_type in ['MCAR', 'MAR', 'MNAR_logistic_T2']:
        for mask_p in [0.3]:
            
            generate_mask(dataname = name,
                            mask_type = mask_type,
                            mask_num = 10,
                            p = mask_p,
                            )
