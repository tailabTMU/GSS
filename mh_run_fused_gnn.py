import gc
import os

import numpy as np
import torch
from utils.mh_fused_graph import MHFusedGraphDataset as FusedGraphDataset
from utils.gnn import train_heterogeneous as gnn_train, test_heterogeneous as gnn_test
from utils.common import get_split_data
import random
from models.mh_models import GraphConvGNNFused as GraphConvGNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch_geometric.loader import DataLoader
import pandas as pd
import argparse
from utils.early_stopping import EarlyStopping
from tqdm import tqdm
import json
import re
import shutil
from sklearn.model_selection import train_test_split
import math


def main(args):
    configs = {
        "params": {
            # "epochs": 1000,
            # "epochs": 100,
            "epochs": 70,
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

    suffix = '_fused_graph'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    print("=========================================================")
    print("Running Experiments with the Following Variables")
    print(f"Number of Folds: {num_folds}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Device: {device}")
    print("=========================================================")
    print()

    dataset = FusedGraphDataset(root=cwd, subdirectory='fused_dataset')

    num_layers = 1
    dropout = 0.5
    hidden_size = 64

    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1)

    print("=========================================================")
    data_path = os.path.join(cwd, 'data', 'fused_dataset')
    for iter_num in range(1, num_folds + 1):
        experiment_results = []
        for split_num in range(1, num_folds + 1):
            train_ids, t_is, v_is, final_test_ids = get_split_data(iter_num, split_num, data_path)

            graphs = FusedGraphDataset(root=cwd, subdirectory='fused_dataset', ids=train_ids.astype(int).tolist())
            labels = []
            for graph in graphs:
                labels.append(graph.y.item())
            total = len(train_ids) + len(t_is) + len(v_is)
            sample_size = math.ceil((3 * (total // 5)))
            _, train_ids, _, train_labels = train_test_split(train_ids, labels, test_size=sample_size, random_state=10)
            train_ids, val_ids, _, _ = train_test_split(train_ids, train_labels, test_size=0.1, random_state=10)

            train_ids = train_ids.astype(int).tolist()
            val_ids = val_ids.astype(int).tolist()
            final_test_ids = final_test_ids.astype(int).tolist()

            print()
            print()
            print('*********************************************************************************')
            print(f'Iteration Number {iter_num} - Split Number {split_num}')
            print()

            train_dataset = FusedGraphDataset(root=cwd, subdirectory='fused_dataset', ids=train_ids)
            val_dataset = FusedGraphDataset(root=cwd, subdirectory='fused_dataset', ids=val_ids)
            test_dataset = FusedGraphDataset(root=cwd, subdirectory='fused_dataset', ids=final_test_ids)
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
            criterion_1 = torch.nn.CrossEntropyLoss()
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
                    tepoch.set_description(f"Epoch {epoch - 1}")
                    train_acc, train_actual_classes_model_1, train_predicted_classes_model_1, _, train_loss, optimizer_1 = gnn_train(
                        first_model, gnn_train_loader, device, criterion_1, optimizer_1)

                    train_acc = accuracy_score(train_actual_classes_model_1,
                                               train_predicted_classes_model_1)
                    val_acc, val_actual_classes_model_1, val_predicted_classes_model_1, _, val_loss = gnn_test(
                        first_model, gnn_val_loader, device, criterion_1)
                    val_acc = accuracy_score(val_actual_classes_model_1, val_predicted_classes_model_1)

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

                    # if epoch >= 20:
                    if epoch >= 1:
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

                train_acc, train_actual_classes_model_1, train_predicted_classes_model_1, _, train_loss = gnn_test(
                    first_model, gnn_train_loader, device, criterion_1)

                train_acc = accuracy_score(train_actual_classes_model_1,
                                           train_predicted_classes_model_1)

                val_acc, val_actual_classes_model_1, val_predicted_classes_model_1, _, val_loss = gnn_test(
                    first_model, gnn_val_loader, device, criterion_1)

                val_acc = accuracy_score(val_actual_classes_model_1, val_predicted_classes_model_1)

                postfix = {'Train Loss': train_loss, 'Validation Loss': val_loss,
                           'Train Accuracy': f'{train_acc * 100}%',
                           'Validation Accuracy': f'{val_acc * 100}%',
                           'Number of Epochs': epoch, 'Model': model_name,
                           'Number of Layers': num_layers,
                           'Hidden Size': hidden_size}

                training_info = json.dumps(postfix)
                info_file = pjoin(cwd, 'results', f'results{suffix}', f'{num_layers}_layers',
                                  f'{hidden_size}_hidden_size',
                                  f'split_{split_num}_iter_{iter_num}_{model_name}_stop_on_{res_type}_info.txt')
                with open(info_file, 'w+') as myfile:
                    myfile.write(str(training_info))

                test_acc, test_actual_classes_model_1, test_predicted_classes_model_1, test_scores_model_1, _ = gnn_test(
                    first_model, gnn_test_loader, device, criterion_1)

                with open(pjoin(cwd, 'results', f'results{suffix}', f'{num_layers}_layers',
                                f'{hidden_size}_hidden_size',
                                f'train_actual_split_{split_num}_iter_{iter_num}_{model_name}_stop_on_{res_type}.txt'),
                          'w+') as myfile:
                    np.savetxt(myfile,
                               train_actual_classes_model_1)
                with open(pjoin(cwd, 'results', f'results{suffix}', f'{num_layers}_layers',
                                f'{hidden_size}_hidden_size',
                                f'val_actual_split_{split_num}_iter_{iter_num}_{model_name}_stop_on_{res_type}.txt'),
                          'w+') as myfile:
                    np.savetxt(myfile,
                               val_actual_classes_model_1)
                with open(pjoin(cwd, 'results', f'results{suffix}', f'{num_layers}_layers',
                                f'{hidden_size}_hidden_size',
                                f'test_actual_split_{split_num}_iter_{iter_num}_{model_name}_stop_on_{res_type}.txt'),
                          'w+') as myfile:
                    np.savetxt(myfile,
                               test_actual_classes_model_1)
                with open(pjoin(cwd, 'results', f'results{suffix}', f'{num_layers}_layers',
                                f'{hidden_size}_hidden_size',
                                f'train_pred_split_{split_num}_iter_{iter_num}_{model_name}_stop_on_{res_type}.txt'),
                          'w+') as myfile:
                    np.savetxt(myfile,
                               train_predicted_classes_model_1)
                with open(pjoin(cwd, 'results', f'results{suffix}', f'{num_layers}_layers',
                                f'{hidden_size}_hidden_size',
                                f'val_pred_split_{split_num}_iter_{iter_num}_{model_name}_stop_on_{res_type}.txt'),
                          'w+') as myfile:
                    np.savetxt(myfile,
                               val_predicted_classes_model_1)
                with open(pjoin(cwd, 'results', f'results{suffix}', f'{num_layers}_layers',
                                f'{hidden_size}_hidden_size',
                                f'test_pred_split_{split_num}_iter_{iter_num}_{model_name}_stop_on_{res_type}.txt'),
                          'w+') as myfile:
                    np.savetxt(myfile,
                               test_predicted_classes_model_1)
                with open(pjoin(cwd, 'results', f'results{suffix}', f'{num_layers}_layers',
                                f'{hidden_size}_hidden_size',
                                f'test_scores_split_{split_num}_iter_{iter_num}_{model_name}_stop_on_{res_type}.txt'),
                          'w+') as myfile:
                    np.savetxt(myfile,
                               test_scores_model_1)

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
    parser = argparse.ArgumentParser(description='Training a GNN Model with Mental Health Fused Dataset')

    main(parser.parse_args())
