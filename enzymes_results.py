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

    suffix = '_enzymes_all_data'
    sampled_suffix = '_enzymes_sampled'
    results_dir = pjoin(cwd, 'results', f'results{suffix}', '3_layers', '90_hidden_size')
    sampled_results_dir = pjoin(cwd, 'results', f'results{sampled_suffix}', '3_layers', '90_hidden_size', 'k_10')
    model_name = 'sage'
    print('Processing...')
    dfs = []
    sampled_dfs = []
    for split_num in range(1, 6):
        true_labels_file = pjoin(results_dir,
                                 f'split_{split_num}_{model_name}_test_true_classes.txt')
        true_labels = np.loadtxt(true_labels_file, dtype="double")
        true_labels = true_labels.astype(int)
        predicted_labels_file = pjoin(results_dir,
                                      f'split_{split_num}_{model_name}_test_predicted_classes.txt')
        predicted_labels = np.loadtxt(predicted_labels_file, dtype="double")
        predicted_labels = predicted_labels.astype(int)
        scores_file = pjoin(results_dir,
                            f'split_{split_num}_{model_name}_test_logit_vals.txt')
        scores = np.loadtxt(scores_file, dtype="double")

        accuracy_1 = accuracy_score(true_labels, predicted_labels)
        y_prob = F.softmax(torch.tensor(scores).double(), dim=-1).numpy()
        roc_auc_1 = roc_auc_score(true_labels, y_prob, average='weighted', multi_class='ovo')
        precision_1 = precision_score(true_labels,
                                      predicted_labels, average='weighted')
        recall_1 = recall_score(true_labels, predicted_labels, average='weighted')
        f1_1 = f1_score(true_labels, predicted_labels, average='weighted')

        result = {
            'accuracy': accuracy_1,
            'roc_auc': roc_auc_1,
            'precision': precision_1,
            'recall': recall_1,
            'f1': f1_1
        }
        dfs.append(result)

    true_labels_file = pjoin(sampled_results_dir,
                             f'{model_name}_test_true_classes.txt')
    true_labels = np.loadtxt(true_labels_file, dtype="double")
    true_labels = true_labels.astype(int)
    predicted_labels_file = pjoin(sampled_results_dir,
                                  f'{model_name}_test_predicted_classes.txt')
    predicted_labels = np.loadtxt(predicted_labels_file, dtype="double")
    predicted_labels = predicted_labels.astype(int)
    scores_file = pjoin(sampled_results_dir,
                        f'{model_name}_test_logit_vals.txt')
    scores = np.loadtxt(scores_file, dtype="double")

    accuracy_1 = accuracy_score(true_labels, predicted_labels)
    y_prob = F.softmax(torch.tensor(scores).double(), dim=-1).numpy()
    roc_auc_1 = roc_auc_score(true_labels, y_prob, average='weighted', multi_class='ovo')
    precision_1 = precision_score(true_labels,
                                  predicted_labels, average='weighted')
    recall_1 = recall_score(true_labels, predicted_labels, average='weighted')
    f1_1 = f1_score(true_labels, predicted_labels, average='weighted')

    result = {
        'accuracy': accuracy_1,
        'roc_auc': roc_auc_1,
        'precision': precision_1,
        'recall': recall_1,
        'f1': f1_1
    }
    sampled_dfs.append(result)

    print('Random Splitting Results')
    experiment_results_df = pd.DataFrame(dfs)
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
    experiment_results_df = pd.DataFrame(sampled_dfs)
    mean = experiment_results_df.mean()
    std = experiment_results_df.std()
    # Combine into "mean ± std" format
    mean_std = mean.round(2).astype(str)

    # Convert to single-row DataFrame
    result_df = pd.DataFrame([mean_std])
    print(result_df)

    print()
    print('**************************************')
    print()