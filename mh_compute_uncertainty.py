import pandas as pd

from utils.metrics import jensenshannon_metric, weighted_agreement, nonlinear_weight
import numpy as np
import os
import torch
import torch.nn.functional as F
from utils.common import get_split_data


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

    ################### Self-Distillation Model ######################
    suffix = '_ehr_graph_self_dist'
    num_layers = 3
    hidden_size = 128
    results_dir = pjoin(cwd, 'results', f'results{suffix}', f'{num_layers}_layers',
                        f'{hidden_size}_hidden_size')
    model_name = 'SelfDistillationGraphConvGNN'
    for iter_num in range(1, 6):
        nonlinear_agreement_vals = []
        graph_ids = []
        split_nums = []
        true_labels = []
        for split_number in range(1, 6):
            ref_scores_file = pjoin(results_dir,
                                    f'{model_name}_layer_{num_layers}_iter_{iter_num}_split_{split_number}_test_logit_vals.txt')
            ref_scores = np.loadtxt(ref_scores_file, dtype="double")
            ref_probs = F.softmax(torch.tensor(ref_scores).double(), dim=-1)
            ref_probs = ref_probs / ref_probs.sum(dim=-1, keepdim=True)
            smoothed_ref_probs = smooth_probabilities(ref_probs)
            ref_predictions = torch.argmax(smoothed_ref_probs, dim=-1).tolist()
            layers_div = [[] for _ in range(1, num_layers)]
            predictions = [[] for _ in range(1, num_layers)]
            id_path = pjoin(cwd, 'data', 'fused_dataset')
            _, test_ids, _, _ = get_split_data(iter_num, split_number, id_path)
            test_ids = test_ids.astype(int).tolist()

            for layer in range(1, num_layers):

                scores_file = pjoin(results_dir,
                                    f'{model_name}_layer_{layer}_iter_{iter_num}_split_{split_number}_test_logit_vals.txt')
                scores = np.loadtxt(scores_file, dtype="double")
                probs = F.softmax(torch.tensor(scores).double(), dim=-1)
                probs = probs / probs.sum(dim=-1, keepdim=True)
                smoothed_probs = smooth_probabilities(probs)
                predictions[layer - 1] = torch.argmax(smoothed_probs, dim=-1).tolist()
                for idx in range(len(smoothed_probs)):
                    div_value = jensenshannon_metric(smoothed_probs[idx], smoothed_ref_probs[idx])
                    layers_div[layer - 1].append(div_value)

            true_class_file = pjoin(results_dir,
                                    f'{model_name}_layer_1_iter_{iter_num}_split_{split_number}_test_true_classes.txt')
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
        df.to_csv(pjoin(results_dir, f'{model_name}_iter_{iter_num}_uncertainty.csv'), index=False)
    print(f'Results Saved to {results_dir}')
