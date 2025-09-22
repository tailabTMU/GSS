import gc
import os

import numpy as np
import torch
from utils.mh_graph import MHEHRGraphDataset
from utils.gnn import train_self_distillation_heterogeneous as gnn_train, \
    test_self_distillation_heterogeneous as gnn_test
from utils.common import get_split_data
import random
from models.mh_models import SelfDistillationGraphConvGNN as GraphConvGNN
from sklearn.metrics import classification_report
from torch_geometric.loader import DataLoader
import argparse
from utils.early_stopping import EarlyStopping
from tqdm import tqdm
import json
import re
import shutil
from utils.loss import SelfDistillationLoss


def main(args):
    configs = {
        "params": {
            # "epochs": 1000,
            "epochs": 100,
            "batch_size": 10,
            # "init_lr": 7e-4,
            "init_lr": 1e-4,
            "lr_reduce_factor": 0.5,
            "lr_schedule_patience": 25,
            "min_lr": 1e-4,
            "weight_decay": 1e-3,
            # "weight_decay": 0.0,
        }
    }

    params = configs['params']
    cwd = os.getcwd()
    path = os.path
    pjoin = path.join

    num_folds = 5
    num_epochs = params['epochs']
    batch_size = params['batch_size']

    suffix = '_ehr_graph_self_dist'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    print("=========================================================")
    print("Running Experiments with the Following Variables")
    print(f"Number of Folds: {num_folds}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Device: {device}")
    print("=========================================================")
    print()

    dataset = MHEHRGraphDataset(root=cwd, subdirectory='fused_dataset')

    num_layers = 3
    dropout = 0.5
    hidden_size = 128

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)

    print("=========================================================")
    data_path = os.path.join(cwd, 'data', 'fused_dataset')
    for iter_num in range(1, num_folds + 1):
        for split_num in range(1, num_folds + 1):
            train_ids, test_ids, val_ids, final_test_ids = get_split_data(iter_num, split_num, data_path)
            train_ids = train_ids.astype(int).tolist()
            test_ids = test_ids.astype(int).tolist()
            val_ids = val_ids.astype(int).tolist()
            # final_test_ids = final_test_ids.astype(int).tolist()

            print()
            print()
            print('*********************************************************************************')
            print(f'Iteration Number {iter_num} - Split Number {split_num}')
            print()

            train_dataset = MHEHRGraphDataset(root=cwd, subdirectory='fused_dataset', ids=train_ids)
            val_dataset = MHEHRGraphDataset(root=cwd, subdirectory='fused_dataset', ids=val_ids)
            test_dataset = MHEHRGraphDataset(root=cwd, subdirectory='fused_dataset', ids=test_ids)
            # test_dataset = MHEHRGraphDataset(root=cwd, subdirectory='fused_dataset', ids=final_test_ids)
            train_dataset = train_dataset.shuffle()
            val_dataset = val_dataset.shuffle()
            # test_dataset = test_dataset.shuffle()

            _, edge_types = dataset[0].metadata()
            metadata = {
                'edges': edge_types,
                'num_features': dataset.num_features,
            }

            gnn_train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            gnn_val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            gnn_test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

            num_classes = 2

            if not path.exists(pjoin(cwd, f'results')):
                os.mkdir(pjoin(cwd, f'results'))

            if not path.exists(pjoin(cwd, f'saved_models')):
                os.mkdir(pjoin(cwd, f'saved_models'))

            if not path.exists(pjoin(cwd, 'saved_models', f'saved_models{suffix}')):
                os.mkdir(pjoin(cwd, 'saved_models', f'saved_models{suffix}'))

            if not path.exists(pjoin(cwd, 'results', f'results{suffix}')):
                os.mkdir(pjoin(cwd, 'results', f'results{suffix}'))

            if not path.exists(pjoin(cwd, 'saved_models', f'saved_models{suffix}', f'{num_layers}_layers')):
                os.mkdir(pjoin(cwd, 'saved_models', f'saved_models{suffix}', f'{num_layers}_layers'))
            if not path.exists(pjoin(cwd, 'results', f'results{suffix}', f'{num_layers}_layers')):
                os.mkdir(pjoin(cwd, 'results', f'results{suffix}', f'{num_layers}_layers'))
            model = GraphConvGNN
            if not path.exists(pjoin(cwd, 'saved_models', f'saved_models{suffix}', f'{num_layers}_layers',
                                     f'{hidden_size}_hidden_size')):
                os.mkdir(pjoin(cwd, 'saved_models', f'saved_models{suffix}', f'{num_layers}_layers',
                               f'{hidden_size}_hidden_size'))
            if not path.exists(pjoin(cwd, 'results', f'results{suffix}', f'{num_layers}_layers',
                                     f'{hidden_size}_hidden_size')):
                os.mkdir(pjoin(cwd, 'results', f'results{suffix}', f'{num_layers}_layers',
                               f'{hidden_size}_hidden_size'))

            first_model = model(metadata, num_classes, device, hidden_size, num_layers=num_layers, dropout=dropout)
            model_name = str(first_model)
            print(f"Initializing the {model_name} hidden_size: {hidden_size} layers: {num_layers}")
            first_model = first_model.to(device)

            optimizer_1 = torch.optim.Adam(first_model.parameters(), lr=params['init_lr'],
                                           weight_decay=params['weight_decay'])
            # scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_1, mode='min',
            #                                                          factor=params['lr_reduce_factor'],
            #                                                          patience=params['lr_schedule_patience'],
            #                                                          verbose=False)
            criterion_1 = SelfDistillationLoss()
            distillation_loss_coefficients = [0.6 for _ in range(num_layers - 1)]
            distillation_loss_coefficients.append(0)
            feature_penalty_coefficients = [0.04 for _ in range(num_layers - 1)]
            feature_penalty_coefficients.append(0)
            loss_checkpoint_path = pjoin(cwd, 'saved_models', f'saved_models{suffix}',
                                         f'{num_layers}_layers',
                                         f'{hidden_size}_hidden_size',
                                         f'split_{split_num}_iter_{iter_num}_{model_name}.pt')
            final_checkpoint_path = pjoin(cwd, 'saved_models', f'saved_models{suffix}',
                                          f'{num_layers}_layers',
                                          f'{hidden_size}_hidden_size',
                                          f'split_{split_num}_iter_{iter_num}_{model_name}_final.pt')
            early_stopping = EarlyStopping(loss_checkpoint_path, patience=10, min_delta=1e-2)

            print("Removing Results for Previously Completed Iterations")
            wdir = pjoin(cwd, 'results', f'results{suffix}', f'{num_layers}_layers',
                         f'{hidden_size}_hidden_size')
            for f in os.listdir(wdir):
                if re.search(f'split_{split_num}_iter_{iter_num}_{model_name}', f):
                    os.remove(os.path.join(wdir, f))

            wdir = pjoin(cwd, 'results', f'results{suffix}', f'{num_layers}_layers',
                         f'{hidden_size}_hidden_size', 'metrics')
            if path.exists(wdir):
                if path.exists(pjoin(wdir, model_name)):
                    shutil.rmtree(pjoin(wdir, model_name))

            wdir = pjoin(cwd, 'saved_models', f'saved_models{suffix}', f'{num_layers}_layers',
                         f'{hidden_size}_hidden_size')
            for f in os.listdir(wdir):
                if re.search(f'split_{split_num}_iter_{iter_num}_{model_name}', f):
                    os.remove(os.path.join(wdir, f))

            print("GNN Training in Progress")
            epoch_train_loss = []
            epoch_train_acc = []
            epoch_val_loss = []
            epoch_val_acc = []
            with tqdm(range(1, params['epochs'] + 1), unit='epoch') as tepoch:
                for epoch in tepoch:
                    if epoch > 80:
                        distillation_loss_coefficients = [0 for _ in range(num_layers)]
                        feature_penalty_coefficients = [0 for _ in range(num_layers)]

                    tepoch.set_description(f"Epoch {epoch - 1}")

                    train_acc, train_loss, optimizer_1 = gnn_train(first_model, gnn_train_loader, device,
                                                                               criterion_1,
                                                                               optimizer_1,
                                                                               distillation_loss_coefficients,
                                                                               feature_penalty_coefficients)
                    val_acc, val_loss = gnn_test(first_model, gnn_val_loader, device, criterion_1,
                                                             distillation_loss_coefficients,
                                                             feature_penalty_coefficients)

                    postfix = {
                        'Train Loss': train_loss,
                        'Val Loss': val_loss,
                        'Train Acc': f'{train_acc * 100}%',
                        'Val Acc': f'{val_acc * 100}%',
                    }
                    tepoch.set_postfix(ordered_dict=postfix)

                    # if optimizer_1.param_groups[0]['lr'] < params['min_lr']:
                    #     print("\n!! LR EQUAL TO MIN LR SET.")
                    #     break
                    # else:
                    #     scheduler_1.step(val_loss)

                    if epoch >= 20:
                        if not early_stopping.early_stop:
                            early_stopping(val_loss, first_model)

                    epoch_train_loss.append(train_loss)
                    epoch_train_acc.append(train_acc)
                    epoch_val_loss.append(val_loss)
                    epoch_val_acc.append(val_acc)

            torch.save(first_model.state_dict(), final_checkpoint_path)
            epoch_loss_acc = {
                'train_loss': epoch_train_loss, 'train_acc': epoch_train_acc,
                'val_loss': epoch_val_loss, 'val_acc': epoch_val_acc,
            }
            epoch_loss_acc_file = pjoin(cwd, 'results', f'results{suffix}', f'{num_layers}_layers',
                                        f'{hidden_size}_hidden_size',
                                        f'split_{split_num}_iter_{iter_num}_{model_name}_epoch_loss_acc.txt')
            with open(epoch_loss_acc_file, 'w+') as myfile:
                myfile.write(str(json.dumps(epoch_loss_acc)))

            for res_type in ['loss', 'final']:
                del first_model
                first_model = model(metadata, num_classes, device, hidden_size,
                                    num_layers=num_layers, dropout=dropout)
                first_model = first_model.to(device)
                if res_type == 'loss':
                    first_model.load_state_dict(
                        torch.load('{}'.format(loss_checkpoint_path), weights_only=True))
                elif res_type == 'final':
                    first_model.load_state_dict(
                        torch.load('{}'.format(final_checkpoint_path), weights_only=True))
                else:
                    raise ValueError(f'Unrecognized res_type: {res_type}')

                first_model.eval()

                logit_vals = []
                predicted_classes = []
                features_list = []
                for layer in range(num_layers):
                    logit_vals.append([])
                    predicted_classes.append([])
                    features_list.append([])
                true_classes = []
                for data in gnn_test_loader:
                    data = data.to(device)
                    true_classes += list(data.y.cpu().numpy())
                    batch_dict = {key: data[key].batch for key, x in data.x_dict.items()}
                    output_list, feature_list = first_model(data.x_dict, data.edge_index_dict, batch_dict)
                    for idx, out in enumerate(output_list):
                        pred = out.detach().argmax(dim=1)  # Use the class with highest probability.
                        features = feature_list[idx]
                        logit_vals[idx] += list(out.cpu().detach().numpy())  # Logit values.
                        predicted_classes[idx] += list(pred.cpu().numpy())
                        features_list[idx] += list(features.cpu().detach().numpy())

                print(f"Test Data ({res_type.title()})", classification_report(true_classes, predicted_classes[-1]))

                results_dir = pjoin(cwd, 'results', f'results{suffix}', f'{num_layers}_layers',
                                    f'{hidden_size}_hidden_size')
                for layer in range(num_layers):
                    with open(pjoin(results_dir,
                                    f'{model_name}_layer_{(layer + 1)}_iter_{iter_num}_split_{split_num}_test_logit_vals.txt'),
                              'w+') as myfile:
                        np.savetxt(myfile, logit_vals[layer])

                    with open(
                            pjoin(results_dir,
                                  f'{model_name}_layer_{(layer + 1)}_iter_{iter_num}_split_{split_num}_test_true_classes.txt'),
                            'w+') as myfile:
                        np.savetxt(myfile, true_classes)

                    with open(pjoin(results_dir,
                                    f'{model_name}_layer_{(layer + 1)}_iter_{iter_num}_split_{split_num}_test_predicted_classes.txt'),
                              'w+') as myfile:
                        np.savetxt(myfile, predicted_classes[layer])
                    with open(pjoin(results_dir,
                                    f'{model_name}_layer_{(layer + 1)}_iter_{iter_num}_split_{split_num}_test_features.txt'),
                              'w+') as myfile:
                        np.savetxt(myfile, features_list[layer])

            del first_model
            del criterion_1
            del optimizer_1
            gc.collect()

            print()
            print('==================================================================================')
            print()

    print('*********************************************************************************')
    print()

    print(f"Results have been saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training a Self-Distillation GNN Model with Mental Health Dataset')

    main(parser.parse_args())
