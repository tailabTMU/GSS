import os
import joblib
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os
import argparse
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from collections import Counter


def seconds_to_human_readable(seconds: int):
    if seconds is not None:
        seconds = int(seconds)
        d = seconds // (3600 * 24)
        h = seconds // 3600 % 24
        m = seconds % 3600 // 60
        s = seconds % 3600 % 60
        if d > 0:
            return '{:02d}D {:02d}H {:02d}m {:02d}s'.format(d, h, m, s)
        elif h > 0:
            return '{:02d}H {:02d}m {:02d}s'.format(h, m, s)
        elif m > 0:
            return '{:02d}m {:02d}s'.format(m, s)
        elif s > 0:
            return '{:02d}s'.format(s)
    return '-'


def get_split_data(iter_num: int, split_num: int, path: str):
    pjoin = os.path.join
    splits_dir = pjoin(path, 'ids')
    if not os.path.exists(splits_dir):
        raise Exception('Split Directory Does Not Exist')
    if (not os.path.exists(pjoin(splits_dir, f'train_ids_iter_{iter_num}_fold_{split_num}.txt'))) or \
            (not os.path.exists(pjoin(splits_dir, f'test_ids_iter_{iter_num}_fold_{split_num}.txt'))) or \
            (not os.path.exists(pjoin(splits_dir, f'val_ids_iter_{iter_num}_fold_{split_num}.txt'))) or \
            (not os.path.exists(pjoin(splits_dir, f'final_test_ids.txt'))):
        raise Exception('Split Does Not Exists')
    pg_mp = pd.read_csv(pjoin(path, 'patient_graph_mapping.csv'))

    train_ids = np.loadtxt(pjoin(splits_dir, f'train_ids_iter_{iter_num}_fold_{split_num}.txt'), dtype=str)
    test_ids = np.loadtxt(pjoin(splits_dir, f'test_ids_iter_{iter_num}_fold_{split_num}.txt'), dtype=str)
    val_ids = np.loadtxt(pjoin(splits_dir, f'val_ids_iter_{iter_num}_fold_{split_num}.txt'), dtype=str)
    final_test_ids = np.loadtxt(pjoin(splits_dir, f'final_test_ids.txt'), dtype=str)

    train_ids = pg_mp[pg_mp['PID'].isin(train_ids.astype(str).tolist())]
    train_ids = train_ids['GID'].values
    test_ids = pg_mp[pg_mp['PID'].isin(test_ids.astype(str).tolist())]
    test_ids = test_ids['GID'].values
    val_ids = pg_mp[pg_mp['PID'].isin(val_ids.astype(str).tolist())]
    val_ids = val_ids['GID'].values

    final_test_ids = pg_mp[pg_mp['PID'].isin(final_test_ids.astype(str).tolist())]
    final_test_ids = final_test_ids['GID'].values
    return train_ids, test_ids, val_ids, final_test_ids


def get_split_sampled_data(iter_num: int, splits_dir: str, suffix: str = ''):
    pjoin = os.path.join
    if not os.path.exists(splits_dir):
        raise Exception('Split Directory Does Not Exist')
    # if (not os.path.exists(pjoin(splits_dir, f'train_ids_iter_{iter_num}{suffix}.txt'))) or \
    #         (not os.path.exists(pjoin(splits_dir, f'test_ids_iter_{iter_num}{suffix}.txt'))) or \
    #         (not os.path.exists(pjoin(splits_dir, f'val_ids_iter_{iter_num}{suffix}.txt'))):
    #     raise Exception('Split Does Not Exists')
    if (not os.path.exists(pjoin(splits_dir, f'train_ids_iter_{iter_num}{suffix}.txt'))) or \
            (not os.path.exists(pjoin(splits_dir, f'val_ids_iter_{iter_num}{suffix}.txt'))):
        raise Exception('Split Does Not Exists')

    train_ids = np.loadtxt(pjoin(splits_dir, f'train_ids_iter_{iter_num}{suffix}.txt'), dtype=str)
    # test_ids = np.loadtxt(pjoin(splits_dir, f'test_ids_iter_{iter_num}{suffix}.txt'), dtype=str)
    val_ids = np.loadtxt(pjoin(splits_dir, f'val_ids_iter_{iter_num}{suffix}.txt'), dtype=str)

    return train_ids.astype(int), val_ids.astype(int)
