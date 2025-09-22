import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import math

if __name__ == '__main__':
    cwd = os.getcwd()
    path = os.path
    pjoin = path.join

    ################### Self-Distillation Model ######################
    suffix = '_ehr_graph_self_dist'
    num_layers = 3
    hidden_size = 128
    results_dir = pjoin(cwd, 'results', f'results{suffix}', f'{num_layers}_layers',
                        f'{hidden_size}_hidden_size')
    data_dir = pjoin(cwd, 'data', 'fused_dataset', f'sampled_graph_ids{suffix}', f'{num_layers}_layers',
                     f'{hidden_size}_hidden_size')
    os.makedirs(data_dir, exist_ok=True)
    model_name = 'SelfDistillationGraphConvGNN'
    print('Processing...')
    for iter_num in range(1, 6):
        print(f'Iteration {iter_num}')
        for n_clusters in [5, 10, 15, 20]:
            # n_clusters = 10
            print(f'{n_clusters} Clusters')
            dfs = []
            clusters_df = pd.read_csv(pjoin(results_dir, f'k_{n_clusters}', f'patient_cluster_iter_{iter_num}.csv'))
            uncertainty_df = pd.read_csv(pjoin(results_dir, f'{model_name}_iter_{iter_num}_uncertainty.csv'))
            for cluster_id in range(0, n_clusters):
                gids = clusters_df[clusters_df['ClusterID'] == cluster_id]['GID'].values.tolist()
                uncertainty_vals = uncertainty_df[uncertainty_df['Graph IDs'].isin(gids)]
                q1 = np.percentile(uncertainty_vals['Uncertainty'].values.tolist(), 25)  # 25th percentile
                q2 = np.percentile(uncertainty_vals['Uncertainty'].values.tolist(), 50)  # 50th percentile
                q3 = np.percentile(uncertainty_vals['Uncertainty'].values.tolist(), 75)  # 75th percentile

                d_0_25 = uncertainty_vals[uncertainty_vals['Uncertainty'] <= q1]
                d_0_25.insert(3, 'Q', ([1] * len(d_0_25)))
                d_0_25.insert(4, 'C', ([cluster_id] * len(d_0_25)))
                d_25_50 = uncertainty_vals[
                    (uncertainty_vals['Uncertainty'] <= q2) & (uncertainty_vals['Uncertainty'] > q1)]
                d_25_50.insert(3, 'Q', ([2] * len(d_25_50)))
                d_25_50.insert(4, 'C', ([cluster_id] * len(d_25_50)))
                d_50_75 = uncertainty_vals[
                    (uncertainty_vals['Uncertainty'] <= q3) & (uncertainty_vals['Uncertainty'] > q2)]
                d_50_75.insert(3, 'Q', ([3] * len(d_50_75)))
                d_50_75.insert(4, 'C', ([cluster_id] * len(d_50_75)))
                d_75_100 = uncertainty_vals[uncertainty_vals['Uncertainty'] > q3]
                d_75_100.insert(3, 'Q', ([4] * len(d_75_100)))
                d_75_100.insert(4, 'C', ([cluster_id] * len(d_75_100)))
                dfs.append(d_0_25)
                dfs.append(d_25_50)
                dfs.append(d_50_75)
                dfs.append(d_75_100)
            combined_df = pd.concat(dfs)
            df = combined_df[:][['Graph IDs', 'Q', 'C']]

            total_n = len(df)
            target_sample_size = math.ceil(3 * (total_n // 5))  # Number of Samples

            group_proportions = {
                '1': 0.1,  # '0-25': 0.1,
                '2': 0.15,  # '25-50': 0.15,
                '3': 0.30,  # '50-75': 0.30,
                '4': 0.45,  # '75-100': 0.45,
            }

            # Compute how many samples to take per cluster (proportional to size)
            cluster_sizes = df['C'].value_counts().sort_index()
            cluster_sample_sizes = ((cluster_sizes / total_n) * target_sample_size).round().astype(int)

            # Ensure the total sample size adds up (due to rounding)
            diff = target_sample_size - cluster_sample_sizes.sum()
            if diff != 0:
                # Adjust the largest cluster to make the total match
                largest_cluster = cluster_sample_sizes.idxmax()
                cluster_sample_sizes[largest_cluster] += diff

            # Collect sampled rows here
            samples = []

            # Sample within each cluster
            for cluster_id, n_samples in cluster_sample_sizes.items():
                cluster_df = df[df['C'] == int(cluster_id)]
                cluster_group_counts = {g: int(round(group_proportions[g] * n_samples)) for g in group_proportions}

                # Fix rounding again if needed
                diff = n_samples - sum(cluster_group_counts.values())
                if diff != 0:
                    largest_group = max(cluster_group_counts, key=group_proportions.get)
                    cluster_group_counts[largest_group] += diff

                for group, group_n in cluster_group_counts.items():
                    group_df = cluster_df[cluster_df['Q'] == int(group)]
                    if len(group_df) >= group_n:
                        sampled = group_df.sample(n=group_n, random_state=42)
                    else:
                        # Take all available if not enough (or oversample if needed)
                        sampled = group_df
                    samples.append(sampled)

            # Combine and shuffle
            final_sample = pd.concat(samples).sample(frac=1, random_state=42).reset_index(drop=True)

            final_df = combined_df[combined_df['Graph IDs'].isin(final_sample['Graph IDs'])]
            # other_df = combined_df[~combined_df['Graph IDs'].isin(final_sample['Graph IDs'])]
            #
            # skf = StratifiedKFold(n_splits=4)
            # ids = other_df['Graph IDs'].values
            # labels = other_df['Label'].values
            # for i, (train_index, test_index) in enumerate(skf.split(ids, labels)):
            #     X_train, X_test = ids[train_index], ids[test_index]
            #     y_train, y_test = labels[train_index], labels[test_index]
            #     X_train, X_val, _, _ = train_test_split(X_train, y_train, test_size=0.1, random_state=10)
            #
            #     X_train = X_train.astype(int).tolist()
            #     # Add the sampled data to the training set
            #     X_train = X_train + final_df['Graph IDs'].values.astype(int).tolist()
            #     np.savetxt(pjoin(data_dir, f'train_ids_iter_{iter_num}_fold_{i + 1}.txt'), X_train, fmt='%s')
            #     X_val = X_val.astype(int).tolist()
            #     np.savetxt(pjoin(data_dir, f'val_ids_iter_{iter_num}_fold_{i + 1}.txt'), X_val, fmt='%s')
            #     X_test = X_test.astype(int).tolist()
            #     np.savetxt(pjoin(data_dir, f'test_ids_iter_{iter_num}_fold_{i + 1}.txt'), X_test, fmt='%s')
            #
            # final_df = combined_df[combined_df['Graph IDs'].isin(final_sample['Graph IDs'])]
            # other_df = combined_df[~combined_df['Graph IDs'].isin(final_sample['Graph IDs'])]

            skf = StratifiedKFold(n_splits=4)
            ids = final_df['Graph IDs'].values
            labels = final_df['Label'].values
            os.makedirs(pjoin(data_dir, f'k_{n_clusters}'), exist_ok=True)
            X_train, X_val, _, _ = train_test_split(ids, labels, test_size=0.1, random_state=10)
            X_train = X_train.astype(int).tolist()
            np.savetxt(pjoin(data_dir, f'k_{n_clusters}', f'train_ids_iter_{iter_num}.txt'), X_train, fmt='%s')
            X_val = X_val.astype(int).tolist()
            np.savetxt(pjoin(data_dir, f'k_{n_clusters}', f'val_ids_iter_{iter_num}.txt'), X_val, fmt='%s')

    print(f'Results saved to {data_dir}')
