import os
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import torch
import torch.nn.functional as F
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

if __name__ == '__main__':
    cwd = os.getcwd()
    path = os.path
    pjoin = path.join

    ################### Self-Distillation Model ######################
    sampled_suffix = '_fused_graph_sampled'
    root_sampled_results_dir = pjoin(cwd, 'results', f'results{sampled_suffix}', '1_layers', '64_hidden_size')
    model_name = 'GraphConvGNN'
    print('Processing...')
    sampled_dfs = {n_cluster: {'loss': [], 'final': [], } for n_cluster in [5, 10, 15, 20]}
    for n_cluster in [5, 10, 15, 20]:
        sampled_results_dir = pjoin(root_sampled_results_dir, f'k_{n_cluster}')
        for iter_num in range(1, 6):
            for res_type in ['loss', 'final']:
                true_labels_file = pjoin(sampled_results_dir,
                                         f'test_actual_iter_{iter_num}_{model_name}_stop_on_{res_type}.txt')
                true_labels = np.loadtxt(true_labels_file, dtype="double")
                true_labels = true_labels.astype(int)
                predicted_labels_file = pjoin(sampled_results_dir,
                                              f'test_pred_iter_{iter_num}_{model_name}_stop_on_{res_type}.txt')
                predicted_labels = np.loadtxt(predicted_labels_file, dtype="double")
                predicted_labels = predicted_labels.astype(int)
                scores_file = pjoin(sampled_results_dir,
                                    f'test_scores_iter_{iter_num}_{model_name}_stop_on_{res_type}.txt')
                scores = np.loadtxt(scores_file, dtype="double")

                accuracy_1 = accuracy_score(true_labels, predicted_labels)
                y_prob = F.softmax(torch.tensor(scores).double(), dim=-1).numpy()
                roc_auc_1 = roc_auc_score(true_labels, y_prob[:, 1])
                precision_1 = precision_score(true_labels,
                                              predicted_labels)
                recall_1 = recall_score(true_labels, predicted_labels)
                f1_1 = f1_score(true_labels, predicted_labels)

                result = {
                    # 'split_number': split_num,
                    # 'iteration_number': iter_num,
                    # 'stop_on': res_type,
                    'accuracy': accuracy_1,
                    'roc_auc': roc_auc_1,
                    'precision': precision_1,
                    'recall': recall_1,
                    'f1': f1_1
                }
                sampled_dfs[n_cluster][res_type].append(result)

    # res_types = ['loss']
    res_types = ['final']
    # res_types = ['loss', 'final']

    for res_type in res_types:
        for n_cluster in [5, 10, 15, 20]:
            print(f'{n_cluster} Clusters Results')
            experiment_results_df = pd.DataFrame(sampled_dfs[n_cluster][res_type])
            mean = experiment_results_df.mean()
            std = experiment_results_df.std()
            # Combine into "mean ± std" format
            mean_std = mean.round(2).astype(str) + " ± " + std.round(2).astype(str)

            # Convert to single-row DataFrame
            result_df = pd.DataFrame([mean_std])
            print(result_df)

            print()
            print('====================================================')
            print()
