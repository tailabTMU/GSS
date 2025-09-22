import pandas as pd
import os
from torch_geometric.datasets import TUDataset
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
from joblib import dump
from sklearn.model_selection import train_test_split, StratifiedKFold

if __name__ == '__main__':
    cwd = os.getcwd()
    path = os.path
    pjoin = path.join

    configs = {
        "params": {
            # "epochs": 1000,
            "epochs": 200,
            "batch_size": 20,
            "init_lr": 7e-4,
            "lr_reduce_factor": 0.5,
            "lr_schedule_patience": 25,
            "min_lr": 1e-6,
            "weight_decay": 0.0,
            "print_epoch_interval": 5,
        },
        'net_params': {
            'L': 3,
            'hidden_dim': 90,
            'out_dim': 90,
            'residual': True,
            'readout': 'mean',
            'in_feat_dropout': 0.0,
            'dropout': 0.0,
            'batch_norm': True,
            'sage_aggregator': 'max'
        }
    }

    config = configs['net_params']
    params = configs['params']

    suffix = '_enzymes_self_dist'
    num_layers = config['L']
    hidden_size = config['hidden_dim']
    results_dir = pjoin(cwd, 'results', f'results{suffix}', f'{num_layers}_layers',
                        f'{hidden_size}_hidden_size')
    model_name = 'sage'

    dataset = TUDataset(root=os.path.join(os.getcwd(), 'data', 'TUDataset'), name='enzymes'.upper(), use_node_attr=True)


    num_classes = dataset.num_classes

    data_idx = []
    data_y = []
    for idx, data in enumerate(dataset):
        data_idx.append(idx)
        data_y.append(data.y.item())

    data_idx = np.array(data_idx)
    data_y = np.array(data_y)

    iid_data, _, labels, _ = train_test_split(data_idx, data_y, test_size=0.1, random_state=42, stratify=data_y)

    skf = StratifiedKFold(n_splits=5)

    print('Processing...')
    dfs = []
    split_number = 0
    for _, test_index in skf.split(iid_data, labels):
        split_number += 1

        test_ids = iid_data[test_index]
        features_file = pjoin(results_dir,
                              f'{model_name}_layer_{num_layers}_split_{split_number}_test_features.txt')
        features = np.loadtxt(features_file, dtype="double")
        columns = [f'F{i + 1}' for i in range(features.shape[1])]
        features_df = pd.DataFrame(features, columns=columns)
        graph_ids_df = pd.DataFrame(test_ids, columns=['GID'])
        dfs.append(pd.concat([graph_ids_df, features_df], axis=1))


    iter_data = pd.concat(dfs, axis=0)
    iter_data.to_csv(pjoin(results_dir, f'enzymes_features.csv'), index=False)

    list_of_k_candidates = range(2, 101)
    clustering_data = iter_data.drop(['GID'], axis=1)
    clustering_data = clustering_data.values

    # X_train, X_test = train_test_split(clustering_data, test_size=0.33, random_state=42)
    #
    # print(f"Total Number of Records {len(clustering_data)}")
    # print(f"Number of Records Used to Find Optimal K Value {len(X_test)}")
    #
    # k_candidate_sum_of_squared_distances = []
    #
    # pbar = tqdm(list_of_k_candidates)
    # for k_candidate in pbar:
    #     pbar.set_description(f'Creating Clusters for K={k_candidate}')
    #     # Calculate error values for all above
    #     km_result = KMeans(n_clusters=k_candidate, random_state=42, n_init="auto").fit(X_test)
    #     k_candidate_sum_of_squared_distances.append(km_result.inertia_)
    #
    # # Plot the each value of K vs. the silhouette score at that value
    # fig, ax = plt.subplots(figsize=(30, 10))
    # ax.set_xlabel('Value of K')
    # ax.set_ylabel('Sum of Squared Distances')
    # ax.plot(list_of_k_candidates, k_candidate_sum_of_squared_distances)
    #
    # # Ticks and grid
    # xticks = np.arange(min(list_of_k_candidates), max(list_of_k_candidates) + 1)
    # ax.set_xticks(xticks, minor=False)
    # ax.set_xticks(xticks, minor=True)
    # ax.xaxis.grid(True, which='both')
    # yticks = np.arange(round(min(k_candidate_sum_of_squared_distances), 2),
    #                    max(k_candidate_sum_of_squared_distances), 1000000)
    # ax.set_yticks(yticks, minor=False)
    # ax.set_yticks(yticks, minor=True)
    # ax.yaxis.grid(True, which='both')
    # plt.savefig(pjoin(cwd, 'data', 'candidates_for_k_means_sum_of_squared_distances.png'))

    for n_clusters in [10]:
        print(f'Clustering with {n_clusters} clusters:')
        # n_clusters = 10
        km_clustering_model = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit(clustering_data)
        clustering_result = km_clustering_model.predict(clustering_data)
        clusters_df = pd.DataFrame(clustering_result.reshape(-1, 1), columns=['ClusterID'])
        clusters_df = pd.concat([pd.DataFrame(iter_data['GID'].values, columns=['GID']), clusters_df], axis=1)
        os.makedirs(pjoin(results_dir, f'k_{n_clusters}'), exist_ok=True)
        clusters_df.to_csv(pjoin(results_dir, f'k_{n_clusters}', f'enzymes_cluster.csv'),
                           index=False)
        kmeans_dir = pjoin(cwd, 'saved_models', f'k_means{suffix}', f'k_{n_clusters}')
        os.makedirs(kmeans_dir, exist_ok=True)
        dump(km_clustering_model, pjoin(kmeans_dir, f'k_means_model .pkl'))
        print()
    print("Clustering Models Fitted and Results Saved!")
