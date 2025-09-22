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
    suffix = '_fused_graph'
    sampled_suffix = '_fused_graph_sampled'
    results_dir = pjoin(cwd, 'results', f'results{suffix}', '1_layers', '64_hidden_size')
    sampled_results_dir = pjoin(cwd, 'results', f'results{sampled_suffix}', '1_layers', '64_hidden_size', 'k_10')
    model_name = 'GraphConvGNN'
    print('Processing...')
    dfs = {
        'loss': [],
        'final': [],
    }
    sampled_dfs = {
        'loss': [],
        'final': [],
    }
    for iter_num in range(1, 6):
        for split_num in range(1, 6):
            for res_type in ['loss', 'final']:
                true_labels_file = pjoin(results_dir,
                                         f'test_actual_split_{split_num}_iter_{iter_num}_{model_name}_stop_on_{res_type}.txt')
                true_labels = np.loadtxt(true_labels_file, dtype="double")
                true_labels = true_labels.astype(int)
                predicted_labels_file = pjoin(results_dir,
                                              f'test_pred_split_{split_num}_iter_{iter_num}_{model_name}_stop_on_{res_type}.txt')
                predicted_labels = np.loadtxt(predicted_labels_file, dtype="double")
                predicted_labels = predicted_labels.astype(int)
                scores_file = pjoin(results_dir,
                                    f'test_scores_split_{split_num}_iter_{iter_num}_{model_name}_stop_on_{res_type}.txt')
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
                dfs[res_type].append(result)

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
            sampled_dfs[res_type].append(result)

    # res_types = ['loss']
    res_types = ['final']
    # res_types = ['loss', 'final']

    for res_type in res_types:
        # print(f'Processing {res_type.title()}-Based Results')
        print('Random Sampling Results')
        experiment_results_df = pd.DataFrame(dfs[res_type])
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

        print('Sampling with Downstream Task Results')
        experiment_results_df = pd.DataFrame(sampled_dfs[res_type])
        mean = experiment_results_df.mean()
        std = experiment_results_df.std()
        # Combine into "mean ± std" format
        mean_std = mean.round(2).astype(str) + " ± " + std.round(2).astype(str)

        # Convert to single-row DataFrame
        result_df = pd.DataFrame([mean_std])
        print(result_df)

        print()
        print('**************************************')
        print()

        rs_df = pd.DataFrame(dfs[res_type])  # Random Sampling
        sdt_df = pd.DataFrame(sampled_dfs[res_type])  # Sampling with Downstream Task
        p_values = {}
        for col in rs_df.columns:
            exp1 = rs_df[col].values
            exp2 = sdt_df[col].values

            # Welch's t-test
            t_stat, p_value = ttest_ind(exp1, exp2, equal_var=False)

            p_values[col] = p_value

            print(f'**************** {col.title()} ****************')
            print(f"T-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
            if p_value < 0.05:
                print(
                    f'We conducted a Welch\'s t-test to compare the performance of Random Sampling (n={len(rs_df)}) and Sampling with Downstream Task (n={len(sdt_df)}). The results show a statistically significant improvement in Sampling with Downstream Task (p = {p_value}), indicating that the performance difference is unlikely due to chance.')
            else:
                print('Not Statistically Significant Improvement.')
            print()

        print()
        print()
        print('############################################################################################')
        print()
        print()

        pvals = list(p_values.values())

        print("Bonferroni Correction")
        adjusted = multipletests(pvals, alpha=0.05, method='bonferroni')

        print("Adjusted p-values:", adjusted[1])
        print("Significant:", adjusted[0])

        print()
        print()

        print("Benjamini-Hochberg (FDR control)")
        adjusted = multipletests(pvals, alpha=0.05, method='fdr_bh')

        print("Adjusted p-values:", adjusted[1])
        print("Significant:", adjusted[0])

        print()
        print()

        print()
        print('======================================================')
        print('======================================================')
        print()
