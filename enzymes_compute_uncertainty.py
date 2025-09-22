import pandas as pd

from utils.metrics import jensenshannon_metric, weighted_agreement, nonlinear_weight
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from sklearn.model_selection import train_test_split, StratifiedKFold


def smooth_probabilities(predictions, epsilon=1e-8):
    # Add epsilon to all probabilities
    smoothed = predictions + epsilon
    # Renormalize along the class dimension (last dimension)
    smoothed = smoothed / smoothed.sum(dim=-1, keepdim=True)
    return smoothed


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

    nonlinear_agreement_vals = []
    graph_ids = []
    split_nums = []
    true_labels = []
    split_number = 0
    for _, test_index in skf.split(iid_data, labels):
        split_number += 1

        test_ids = iid_data[test_index]

        ref_scores_file = pjoin(results_dir,
                                f'{model_name}_layer_{num_layers}_split_{split_number}_test_logit_vals.txt')
        ref_scores = np.loadtxt(ref_scores_file, dtype="double")
        ref_probs = F.softmax(torch.tensor(ref_scores).double(), dim=-1)
        ref_probs = ref_probs / ref_probs.sum(dim=-1, keepdim=True)
        smoothed_ref_probs = smooth_probabilities(ref_probs)
        ref_predictions = torch.argmax(smoothed_ref_probs, dim=-1).tolist()
        layers_div = [[] for _ in range(1, num_layers)]
        predictions = [[] for _ in range(1, num_layers)]

        test_ids = test_ids.astype(int).tolist()

        for layer in range(1, num_layers):

            scores_file = pjoin(results_dir,
                                f'{model_name}_layer_{layer}_split_{split_number}_test_logit_vals.txt')
            scores = np.loadtxt(scores_file, dtype="double")
            probs = F.softmax(torch.tensor(scores).double(), dim=-1)
            probs = probs / probs.sum(dim=-1, keepdim=True)
            smoothed_probs = smooth_probabilities(probs)
            predictions[layer - 1] = torch.argmax(smoothed_probs, dim=-1).tolist()
            for idx in range(len(smoothed_probs)):
                div_value = jensenshannon_metric(smoothed_probs[idx], smoothed_ref_probs[idx])
                layers_div[layer - 1].append(div_value)

        true_class_file = pjoin(results_dir,
                                f'{model_name}_layer_1_split_{split_number}_test_true_classes.txt')
        true_class = np.loadtxt(true_class_file, dtype="float").astype(int)
        true_labels += true_class.tolist()

        layers_weight = [[] for _ in range(1, num_layers)]
        for layer in range(1, num_layers):
            layers_weight[layer - 1] = nonlinear_weight(num_layers, num_layers - layer, predictions[layer - 1],
                                                        ref_predictions)
        test_agreement_values = weighted_agreement(layers_weight, layers_div, normalize=True, num_layers=num_layers,
                                                   weight_func_type='nonlinear')
        nonlinear_agreement_vals += test_agreement_values
        graph_ids += test_ids
        split_nums += [split_number] * len(test_ids)

    nonlinear_agreement_vals = np.array(nonlinear_agreement_vals)
    graph_ids = np.array(graph_ids)
    df = pd.DataFrame({'Graph IDs': graph_ids, 'Split': split_nums, 'Uncertainty': nonlinear_agreement_vals,
                       'Label': true_labels})
    df = df.sort_values(by='Uncertainty', ascending=False)
    df.to_csv(pjoin(results_dir, f'{model_name}_uncertainty.csv'), index=False)
    print(f'Results Saved to {results_dir}')
