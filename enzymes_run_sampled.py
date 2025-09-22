import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from models.enzymes_models import SAGE
from utils.gnn import train, test
from sklearn.metrics import classification_report
import numpy as np
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
import argparse
from tqdm import tqdm
import random
import glob
import time
from sklearn.metrics import f1_score, accuracy_score


def main(args):
    cwd = os.getcwd()
    path = os.path
    pjoin = path.join

    configs = {
        "params": {
            # "epochs": 1000,
            "epochs": 100,
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

    dataset = TUDataset(root=os.path.join(os.getcwd(), 'data', 'TUDataset'), name='enzymes'.upper(), use_node_attr=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    print()
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print(f'Device: {device}')

    num_classes = dataset.num_classes

    data_idx = []
    data_y = []
    for idx, data in enumerate(dataset):
        data_idx.append(idx)
        data_y.append(data.y.item())

    data_idx = np.array(data_idx)
    data_y = np.array(data_y)

    iid_data, final_test_idx, labels, final_test_y = train_test_split(data_idx, data_y, test_size=0.1, random_state=42,
                                                                      stratify=data_y)

    test_dataset = dataset.index_select(final_test_idx)

    test_predicted_classes = {}
    test_true_classes = {}
    avg_train_time = {}

    print("=========================================================")
    root_sampled_data_path = pjoin(cwd, 'data', 'enzymes', f'sampled_graph_ids{suffix}', f'3_layers',
                                   f'{hidden_size}_hidden_size')

    for n_clusters in [10]:
        print(f'{n_clusters} Clusters')
        sampled_data_path = pjoin(root_sampled_data_path, f'k_{n_clusters}')

        train_dataset_idx = np.loadtxt(pjoin(sampled_data_path, f'train_ids.txt'), dtype=str)
        validation_dataset_idx = np.loadtxt(pjoin(sampled_data_path, f'val_ids.txt'), dtype=str)

        train_dataset_idx = train_dataset_idx.astype(int).tolist()
        validation_dataset_idx = validation_dataset_idx.astype(int).tolist()

        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1)

        train_dataset = dataset.index_select(train_dataset_idx)
        validation_dataset = dataset.index_select(validation_dataset_idx)

        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True,
                                  collate_fn=train_dataset.collate)
        val_loader = DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=False,
                                collate_fn=validation_dataset.collate)
        test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False,
                                 collate_fn=test_dataset.collate)
        start_time = time.time()
        # setting seeds
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(1)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1)

        model_name = 'sage'
        model = SAGE(dataset.num_features, num_classes, config)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=params['init_lr'], weight_decay=params['weight_decay'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                               factor=params['lr_reduce_factor'],
                                                               patience=params['lr_schedule_patience'],
                                                               verbose=False)
        criterion = torch.nn.CrossEntropyLoss()

        root_ckpt_dir = pjoin(cwd, 'saved_models', 'saved_models_enzymes_sampled', f'{config["L"]}_layers',
                              f'{config["hidden_dim"]}_hidden_size', f'k_{n_clusters}')
        os.makedirs(root_ckpt_dir, exist_ok=True)
        results_dir = pjoin(cwd, 'results', 'results_enzymes_sampled', f'{config["L"]}_layers',
                            f'{config["hidden_dim"]}_hidden_size', f'k_{n_clusters}')
        os.makedirs(results_dir, exist_ok=True)

        print(f'Training Model')
        with tqdm(range(params['epochs'])) as t:
            for epoch in t:
                t.set_description('Epoch %d' % epoch)

                epoch_train_acc, _, _, _, epoch_train_loss, optimizer = train(model, train_loader, device,
                                                                              criterion,
                                                                              optimizer)
                epoch_val_acc, _, _, _, epoch_val_loss = test(model, val_loader, device, criterion)

                t.set_postfix(lr=optimizer.param_groups[0]['lr'],
                              train_loss=epoch_train_loss, val_loss=epoch_val_loss,
                              train_acc=epoch_train_acc, val_acc=epoch_val_acc,
                              )

                # Saving checkpoint
                ckpt_dir = os.path.join(root_ckpt_dir, f"RUN")
                if not os.path.exists(ckpt_dir):
                    os.makedirs(ckpt_dir)
                torch.save(model.state_dict(), '{}.pkl'.format(pjoin(ckpt_dir, f"epoch_{str(epoch)}")))

                files = glob.glob(pjoin(ckpt_dir, '*.pkl'))
                for file in files:
                    epoch_nb = file.split('_')[-1]
                    epoch_nb = int(epoch_nb.split('.')[0])
                    if epoch_nb < epoch - 1:
                        os.remove(file)

                scheduler.step(epoch_val_loss)

                if optimizer.param_groups[0]['lr'] < params['min_lr']:
                    print("\n!! LR EQUAL TO MIN LR SET.")
                    break

        train_time = (time.time() - start_time)

        logit_vals = []
        true_classes = []
        predicted_classes = []
        for data in test_loader:
            data = data.to(device)
            true_classes += list(data.y.cpu().numpy())
            out = model(data.x, data.edge_index, data.batch)
            pred = out.detach().argmax(dim=1)  # Use the class with highest probability.
            logit_vals += list(out.cpu().detach().numpy())  # Logit values.
            predicted_classes += list(pred.cpu().numpy())

        print("Test Data", classification_report(true_classes, predicted_classes))

        with open(pjoin(results_dir,
                        f'{model_name}_test_logit_vals.txt'),
                  'w+') as myfile:
            np.savetxt(myfile, logit_vals)

        with open(
                pjoin(results_dir,
                      f'{model_name}_test_true_classes.txt'),
                'w+') as myfile:
            np.savetxt(myfile, true_classes)

        with open(pjoin(results_dir,
                        f'{model_name}_test_predicted_classes.txt'),
                  'w+') as myfile:
            np.savetxt(myfile, predicted_classes)

        with open(pjoin(results_dir,
                        f'{model_name}_time_to_train.txt'),
                  'w+') as myfile:
            myfile.write(f'Time to Train (s): {train_time}')


    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training a Graph Classifiers with CrossEntropy Loss using the Sampled Enzymes Dataset')
    main(parser.parse_args())
