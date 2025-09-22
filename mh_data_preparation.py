import os
import pandas as pd
import argparse
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, StratifiedKFold
import numpy as np
from utils.mh_graph import MHEHRGraphDataset
from utils.mh_fused_graph import MHFusedGraphDataset

from functools import reduce


def main(args):
    cwd = os.getcwd()
    path = os.path
    pjoin = path.join


    print('Preparing the MHEHRGraphDataset...')
    mh_ehr_dataset = MHEHRGraphDataset(root=cwd, subdirectory='fused_dataset')
    print('Done!')
    print('Preparing the MHFusedGraphDataset...')
    mh_fused_dataset = MHFusedGraphDataset(root=cwd, subdirectory='fused_dataset')
    print('Done!')
    print()
    print('Preparing the Splits...')

    df = pd.read_csv(pjoin(cwd, 'data', 'fused_dataset', 'fixed_questionnaires.csv'))
    patient_graph_mapping_df = df[['PatientID', 'C_1']].copy()
    patient_graph_mapping_df['GID'] = range(1, (patient_graph_mapping_df.shape[0] + 1))
    patient_graph_mapping_df = patient_graph_mapping_df.rename(
        columns={"PatientID": "PID", "C_1": "QDate", "GID": "GID"})
    patient_graph_mapping_df.to_csv(pjoin(cwd, 'data', 'fused_dataset', 'patient_graph_mapping.csv'), index=False)

    majority = df[df['180_days_ed_visit_label'] == 0]
    minority = df[df['180_days_ed_visit_label'] == 1]
    if len(minority) > len(majority):
        majority = df[df['180_days_ed_visit_label'] == 1]
        minority = df[df['180_days_ed_visit_label'] == 0]

    data_dir = path.join(cwd, 'data', 'fused_dataset', 'ids')
    os.makedirs(data_dir, exist_ok=True)

    majority_test_ids = resample(
        majority,
        replace=False,  # No replacement
        n_samples=15,  # Select 15
        random_state=10  # For reproducibility
    )['PatientID'].values
    minority_test_ids = resample(
        minority,
        replace=False,  # No replacement
        n_samples=15,  # Select 10
        random_state=10  # For reproducibility
    )['PatientID'].values

    majority = majority[~majority['PatientID'].isin(majority_test_ids)]
    minority = minority[~minority['PatientID'].isin(minority_test_ids)]

    final_test_ids = np.concatenate((majority_test_ids, minority_test_ids))
    final_test_ids = final_test_ids.astype(str).tolist()
    np.savetxt(pjoin(data_dir, f'final_test_ids.txt'), final_test_ids, fmt='%s')

    for iteration in range(1, 6):
        # Downsample majority class to the size of the minority class
        majority_downsampled = resample(
            majority,
            replace=False,  # No replacement
            n_samples=len(minority),  # Match minority class size
            random_state=(42 * iteration)  # For reproducibility
        )
        # Combine the undersampled majority class with the minority class
        undersampled_data = pd.concat([majority_downsampled, minority])

        # Shuffle the result
        undersampled_data = undersampled_data.sample(frac=1, random_state=42).reset_index(drop=True)
        skf = StratifiedKFold(n_splits=5)
        ids = undersampled_data['PatientID'].values
        labels = undersampled_data['180_days_ed_visit_label'].values
        for i, (train_index, test_index) in enumerate(skf.split(ids, labels)):
            X_train, X_test = ids[train_index], ids[test_index]
            y_train, y_test = labels[train_index], labels[test_index]
            X_train, X_val, _, _ = train_test_split(X_train, y_train, test_size=0.1, random_state=10)

            X_train = X_train.astype(str).tolist()
            np.savetxt(pjoin(data_dir, f'train_ids_iter_{iteration}_fold_{i+1}.txt'), X_train, fmt='%s')
            X_val = X_val.astype(str).tolist()
            np.savetxt(pjoin(data_dir, f'val_ids_iter_{iteration}_fold_{i+1}.txt'), X_val, fmt='%s')
            X_test = X_test.astype(str).tolist()
            np.savetxt(pjoin(data_dir, f'test_ids_iter_{iteration}_fold_{i+1}.txt'), X_test, fmt='%s')

    print(f'Data Stored to {pjoin(data_dir)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare the data')

    main(parser.parse_args())
