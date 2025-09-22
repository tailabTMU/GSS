from typing import List
import os.path as osp
import torch
from utils.mh_dataset import read_ehr_data
import pandas as pd
import numpy as np

from torch_geometric.data import Dataset


class MHEHRGraphDataset(Dataset):

    def __init__(self, root: str, transform=None,
                 subdirectory='',
                 pre_transform=None,
                 pre_filter=None, ids=None):
        self.data_directory = 'data'
        if subdirectory.strip() != '':
            self.data_directory = osp.join('data', subdirectory.strip())
        if not osp.exists(self.data_directory):
            raise FileNotFoundError(f'{self.data_directory} does not exist')
        if ids is None:
            df = pd.read_csv(osp.join(root, self.data_directory, 'fixed_questionnaires.csv'))
            self.ids = range(1, len(df) + 1)
            del df
        else:
            self.ids = ids
        super().__init__(root, transform, pre_transform, pre_filter)
        for file in self.raw_paths:
            if not osp.exists(file):
                raise FileNotFoundError(f'{file} does not exist')


    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.data_directory)

    @property
    def base_directory(self) -> str:
        return osp.join(self.raw_dir, 'large_ehr_graph')

    @property
    def processed_dir(self) -> str:
        suffix = ''
        return osp.join(self.base_directory, f'processed_data{suffix}')

    @property
    def preprocessed_dir(self) -> str:
        return osp.join(self.base_directory, 'preprocessed')

    @property
    def raw_file_names(self) -> List[str]:
        names = [
            'fixed_questionnaires',
            'fixed_ed',
        ]
        return [f'{name}.csv' for name in names]

    @property
    def raw_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        downloading."""
        files = self.raw_file_names
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        names = []
        for p_id in self.ids:
            names.append(f'data_{p_id}.pt')
        return names

    def download(self):
        pass

    def process(self):
        read_ehr_data(osp.join(self.raw_dir), base_directory=self.base_directory,
                      preprocessed_directory=self.preprocessed_dir,
                      recreate=False, pre_transform=self.pre_transform, pre_filter=self.pre_filter,
                      graph_dir=self.processed_dir)

    def len(self):
        return len(self.ids)

    def get(self, idx):
        p_id = self.ids[idx]
        data = torch.load(osp.join(self.processed_dir, f'data_{p_id}.pt'))
        return data

    def __repr__(self) -> str:
        return f'MHEHRGraphDataset()'
