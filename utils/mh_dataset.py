import pandas as pd
import os
import torch
import os.path as osp
import numpy as np
from torch_geometric.data import HeteroData
from utils.patient_dataset_common import Node, process_features, process_edge_index, read_file, cat
from dateutil import parser, relativedelta
import joblib
import torch.nn.functional as F
from multiprocessing import Pool, cpu_count
from functools import partial
import math


def read_ehr_data(raw_data_dir, base_directory=None, preprocessed_directory='preprocessed', recreate=False,
                  graph_dir=None, pre_transform=None, pre_filter=None):
    all_preprocessed_files_exist = True
    if base_directory is None:
        base_directory = raw_data_dir
    if graph_dir is None:
        graph_dir = os.path.join(base_directory, preprocessed_directory, 'patient')
    suffix = ''
    if recreate:
        all_preprocessed_files_exist = False
    elif not os.path.exists(
            os.path.join(base_directory, preprocessed_directory, f'adjacency_matrix_ehr_data_{suffix}.txt')):
        all_preprocessed_files_exist = False
    elif not os.path.exists(os.path.join(base_directory, preprocessed_directory, f'nodes_ehr_data_{suffix}.txt')):
        all_preprocessed_files_exist = False
    elif not os.path.exists(
            os.path.join(base_directory, preprocessed_directory, f'node_features_ehr_data_{suffix}.txt')):
        all_preprocessed_files_exist = False
    elif not os.path.exists(
            os.path.join(base_directory, preprocessed_directory, f'graph_indicators_ehr_data_{suffix}.txt')):
        all_preprocessed_files_exist = False
    elif not os.path.exists(
            os.path.join(base_directory, preprocessed_directory, f'graph_labels_ehr_data_{suffix}.txt')):
        all_preprocessed_files_exist = False
    if all_preprocessed_files_exist is False:
        create_files(raw_data_dir, base_directory, preprocessed_directory)

    print("Loading Files")

    batch = read_file(os.path.join(base_directory, preprocessed_directory), '',
                      f'graph_indicators_ehr_data_{suffix}',
                      dtype=torch.long, to_tensor=True)
    node_features = read_file(os.path.join(base_directory, preprocessed_directory), '',
                              f'node_features_ehr_data_{suffix}',
                              dtype=torch.long)
    nodes = read_file(os.path.join(base_directory, preprocessed_directory), '', f'nodes_ehr_data_{suffix}')

    nodes = [[int(node[0]), node[1]] for node in nodes]
    # num_cpu_workers = 2
    num_cpu_workers = math.floor(cpu_count() / 2)

    node_types = ['ed_visit', 'diagnosis', 'service']
    node_indices_dict = {}
    node_batch_indices_dict = {}
    node_features_dict = {}
    for node_type in node_types:
        node_indices_dict[node_type] = None
        node_batch_indices_dict[node_type] = None
        node_features_dict[node_type] = None

        print(f"Finding {node_type} Nodes")
        with Pool(num_cpu_workers) as p:
            node_type_indices = p.map(partial(find_indices_from_node, node_type), enumerate(nodes))
        node_type_indices = np.array(node_type_indices)
        node_type_indices = node_type_indices[node_type_indices != np.array(None)]
        node_type_indices = node_type_indices.tolist()
        node_indices_dict[node_type] = node_type_indices
        print(f"Finding {node_type} Nodes in Batch")
        node_batch_indices_dict[node_type] = batch[node_type_indices]

        print(f"{node_type} Features being Processed")
        if not os.path.exists(
                os.path.join(base_directory, preprocessed_directory,
                             f'processed_{node_type}_ehr_data_{suffix}.pkl')) or all_preprocessed_files_exist is True:
            with Pool(num_cpu_workers) as p:
                node_type_features = p.map(partial(find_feature_from_node_indices, node_features),
                                           node_type_indices)
                node_type_features = np.array(node_type_features)
            if node_type_features.shape[1] == 1:
                node_type_features = cat(
                    [process_features(torch.tensor(node_type_features).to(torch.long).squeeze(), True), None])
            else:
                node_type_features_tmp = []
                for feature_idx in range(0, node_type_features.shape[1]):
                    node_type_features_tmp.append(
                        process_features(torch.tensor(node_type_features[:, feature_idx]).to(torch.long).squeeze(),
                                         True)
                    )
                node_type_features = cat(node_type_features_tmp)

            joblib.dump(node_type_features,
                        os.path.join(base_directory, preprocessed_directory,
                                     f'processed_{node_type}_ehr_data_{suffix}.pkl'))
        else:
            node_type_features = joblib.load(
                os.path.join(base_directory, preprocessed_directory,
                             f'processed_{node_type}_ehr_data_{suffix}.pkl'))
        node_features_dict[node_type] = node_type_features

    y = read_file(os.path.join(base_directory, preprocessed_directory), '', f'graph_labels_ehr_data_{suffix}',
                  dtype=torch.long,
                  to_tensor=True)
    _, y = y.unique(sorted=True, return_inverse=True)

    edge_index = read_file(os.path.join(base_directory, preprocessed_directory), '',
                           f'adjacency_matrix_ehr_data_{suffix}',
                           dtype=torch.long, to_tensor=False)

    # Create HeteroData for each Patient
    print("Creating Dataset for Each Patient")
    unique_graph_ids = torch.unique(batch).tolist()
    params = {}
    params['batch'] = batch
    params['node_batch_indices_dict'] = node_batch_indices_dict
    params['node_indices_dict'] = node_indices_dict
    params['node_features_dict'] = node_features_dict
    params['edge_index'] = edge_index
    params['y'] = y
    params['graph_dir'] = graph_dir
    params['pre_transform'] = pre_transform
    params['pre_filter'] = pre_filter
    params['node_types'] = node_types
    # with Pool(num_cpu_workers) as p:
    #     p.map(partial(create_graph_from_data, params), unique_graph_ids)
    for graph_id in unique_graph_ids:
        create_graph_from_data(params, graph_id)


def find_indices_from_node(node_type, tuple_item):
    index, node = tuple_item
    if node[1] == node_type:
        return index


def find_batch_indices_from_node_indices(node_indices, tuple_item):
    index, batch_id = tuple_item
    if index in node_indices:
        return batch_id


def find_feature_from_node_indices(node_features, node_index):
    return node_features[node_index]


def create_graph_from_data(params, graph_id):
    batch = params['batch']
    node_batch_indices_dict = params['node_batch_indices_dict']
    node_indices_dict = params['node_indices_dict']
    node_features_dict = params['node_features_dict']
    node_types = params['node_types']
    edge_index = params['edge_index']
    y = params['y']
    graph_dir = params['graph_dir']
    pre_filter = params['pre_filter']
    pre_transform = params['pre_transform']

    b = batch == graph_id
    indices = b.nonzero().squeeze().tolist()

    node_type_indices = {}
    patient_node_type_indices = {}
    node_type_features_dict = {}
    for node_type in node_types:
        node_type_indices[node_type] = None
        patient_node_type_indices[node_type] = None
        node_type_features_dict[node_type] = None

        b_i = node_batch_indices_dict[node_type] == graph_id
        node_type_indices[node_type] = b_i.nonzero().squeeze().tolist()
        if type(node_type_indices[node_type]) != list:
            node_type_indices[node_type] = [node_type_indices[node_type]]

        # Get Nodes for node_type
        patient_node_type_indices[node_type] = [e for e in indices if e in node_indices_dict[node_type]]
        # visit_feature_indices = [index for (index, e) in enumerate(patient_visit_indices) if e in visit_node_indices]
        # patient_visits = visit_features[visit_feature_indices]
        node_type_features_dict[node_type] = node_features_dict[node_type][node_type_indices[node_type]]

    edge_indices_dict = {}
    for relation in ['ed_visit-diagnosis', 'ed_visit-ed_visit', 'diagnosis-service']:
        relation_types = relation.split('-')
        node_type_1 = relation_types[0]
        node_type_2 = relation_types[1]
        edge_indices_dict[f'{node_type_1}_{node_type_2}'] = None
        edge_indices_dict[f'{node_type_1}_{node_type_2}'] = [
            [patient_node_type_indices[node_type_1].index(e[0]),
             patient_node_type_indices[node_type_2].index(e[1])]
            for e in edge_index
            if
            (e[0] in patient_node_type_indices[node_type_1] and e[1] in patient_node_type_indices[
                node_type_2])
        ]
        edge_indices_dict[f'{node_type_1}_{node_type_2}'] = process_edge_index(
            edge_indices_dict[f'{node_type_1}_{node_type_2}'])

    # Get Patient Graph Label
    patient_y = y[(graph_id - 1)]
    data = HeteroData()

    for node_type in node_types:
        data[node_type].x = node_type_features_dict[node_type]

    for relation in ['ed_visit-diagnosis', 'ed_visit-ed_visit', 'diagnosis-service']:
        relation_types = relation.split('-')
        node_type_1 = relation_types[0]
        node_type_2 = relation_types[1]
        data[node_type_1, f'{node_type_1}_{node_type_2}', node_type_2].edge_index = edge_indices_dict[
            f'{node_type_1}_{node_type_2}']
        data[node_type_1, f'{node_type_1}_{node_type_2}', node_type_2].edge_attr = np.array([])


    data.y = patient_y

    skip = False

    if pre_filter is not None and not pre_filter(data):
        skip = True

    if not skip:
        if pre_transform is not None:
            data = pre_transform(data)

        torch.save(data, osp.join(graph_dir, f'data_{graph_id}.pt'))


def create_files(raw_data_dir, base_directory, preprocessed_directory='preprocessed'):
    print("Creating Files")
    suffix = ''
    if not os.path.exists(os.path.join(base_directory, preprocessed_directory)):
        os.makedirs(os.path.join(base_directory, preprocessed_directory))

    graph_indicator = []
    adjacency_matrix = []
    all_nodes = []
    graph_labels = []

    path = osp.join(raw_data_dir, 'fixed_questionnaires.csv')
    ed_path = osp.join(raw_data_dir, 'fixed_ed.csv')

    columns = [
        '180_days_ed_visit_label'
    ]

    questionnaire_id_col_name = 'PatientID'
    questionnaire_date_col = 'C_1'
    patient_questionnaire = pd.read_csv(path, usecols=(columns + [questionnaire_id_col_name, questionnaire_date_col]))
    ed_columns = ['Research ID', 'Patient ID', 'Registration Date', 'Patient Age Years', 'Triage Level',
                  'Visit Disposition Code',
                  'All Provider Services Involved During Encounter',
                  'Most Responsible Diagnosis Code']
    patient_ed = pd.read_csv(ed_path, usecols=ed_columns)
    service_col_name = 'All Provider Services Involved During Encounter'
    diagnosis_col_name = 'Most Responsible Diagnosis Code'
    service_col_index = patient_ed.columns.get_loc(service_col_name) + 1
    diagnosis_col_index = patient_ed.columns.get_loc(diagnosis_col_name) + 1
    patient_questionnaire['date'] = pd.to_datetime(patient_questionnaire[questionnaire_date_col])
    patient_ed['date'] = pd.to_datetime(patient_ed['Registration Date'])

    # Features for Each Node Type:
    #  1. VISIT: Triage Level, Visit Disposition Code
    #  2. SERVICE: All Provider Services Involved During Encounter
    #  3. Diagnosis: Most Responsible Diagnosis Code

    # Turn words to number for categorical columns
    categorical_columns_num_classes = {}
    categorical_columns_min_class = {}
    for column in ['Triage Level', 'Visit Disposition Code',
                   'All Provider Services Involved During Encounter',
                   'Most Responsible Diagnosis Code']:
        unique_values = patient_ed[column].unique().tolist()
        patient_ed[column] = [unique_values.index(e) for e in patient_ed[column]]
        categorical_columns_num_classes[column] = len(unique_values)
        categorical_columns_min_class[column] = min(patient_ed[column])

    graph_id_index = 1
    # TODO: Handle Categorical Data
    # TODO: Remove questionnaire nodes
    for idx in range(0, len(patient_questionnaire)):
        patient_data = patient_questionnaire.iloc[idx]
        patient_id = patient_data[questionnaire_id_col_name]
        q_date = patient_data[questionnaire_date_col]
        # patient_ed_data = patient_ed[patient_ed['Research ID'] == patient_data[questionnaire_id_col_name]]
        patient_ed_data = patient_ed[patient_ed['Patient ID'] == patient_data[questionnaire_id_col_name]]
        patient_ed_data = patient_ed_data[patient_ed_data['date'] < patient_data['date']]
        # graph_label = patient_data['180-days ED visit label']
        graph_label = patient_data['180_days_ed_visit_label']

        graph_labels.append(int(graph_label))
        nodes_dict = {}

        patient_ed_data_days = list(patient_ed_data['Registration Date'].unique())

        # Handling Event Timeline
        visit_visit_relation = {}
        visit_diagnosis_relation = {}
        diagnosis_service_relation = {}
        added_adjacency = []
        services = {}
        diagnosis = {}
        previous_visit = None
        for visit_index, visit_date in enumerate(list(patient_ed_data_days), start=1):
            visit_data = patient_ed_data[patient_ed_data['Registration Date'] == visit_date]
            visit_node_features = [
                visit_data.iloc[0]['Triage Level'],
                visit_data.iloc[0]['Visit Disposition Code'],
            ]
            node_type = 'ed_visit'
            if node_type not in nodes_dict:
                nodes_dict[node_type] = []
            visit_node = Node(node_type=node_type, node_features=visit_node_features)
            nodes_dict[node_type].append(visit_node)
            all_nodes.append(visit_node)
            graph_indicator.append(graph_id_index)
            if previous_visit is not None:
                if previous_visit.node_id not in visit_visit_relation:
                    visit_visit_relation[previous_visit.node_id] = []
                visit_visit_relation[previous_visit.node_id].append(visit_node)
                added_adjacency.append(f"{previous_visit.node_id}_{visit_node.node_id}")

            for row in visit_data.itertuples():
                if 'diagnosis' not in nodes_dict:
                    nodes_dict['diagnosis'] = []
                if 'service' not in nodes_dict:
                    nodes_dict['service'] = []
                if f"diagnosis-{row[diagnosis_col_index]}" in diagnosis:
                    diagnosis_node = nodes_dict['diagnosis'][diagnosis[f"diagnosis-{row[diagnosis_col_index]}"]]
                else:
                    diagnosis_node_features = [
                        row[diagnosis_col_index]
                    ]
                    diagnosis_node = Node(node_type='diagnosis', node_features=diagnosis_node_features)

                    nodes_dict['diagnosis'].append(diagnosis_node)
                    all_nodes.append(diagnosis_node)
                    graph_indicator.append(graph_id_index)
                    diagnosis[f"diagnosis-{row[diagnosis_col_index]}"] = len(nodes_dict['diagnosis']) - 1

                if f"{visit_node.node_id}_{diagnosis_node.node_id}" not in added_adjacency:
                    if visit_node.node_id not in visit_diagnosis_relation:
                        visit_diagnosis_relation[visit_node.node_id] = []
                    visit_diagnosis_relation[visit_node.node_id].append(diagnosis_node)
                    added_adjacency.append(f"{visit_node.node_id}_{diagnosis_node.node_id}")

                if f"service-{row[service_col_index]}" in services:
                    service_node = nodes_dict['service'][services[f"service-{row[service_col_index]}"]]
                else:
                    service_node_features = [
                        row[service_col_index]
                    ]
                    service_node = Node(node_type='service', node_features=service_node_features)
                    nodes_dict['service'].append(service_node)
                    all_nodes.append(service_node)
                    graph_indicator.append(graph_id_index)
                    services[f"service-{row[service_col_index]}"] = len(nodes_dict['service']) - 1
                if f"{diagnosis_node.node_id}_{service_node.node_id}" not in added_adjacency:
                    if diagnosis_node.node_id not in diagnosis_service_relation:
                        diagnosis_service_relation[diagnosis_node.node_id] = []
                    diagnosis_service_relation[diagnosis_node.node_id].append(service_node)
                    added_adjacency.append(f"{diagnosis_node.node_id}_{service_node.node_id}")

            previous_visit = visit_node

        for node in nodes_dict['ed_visit']:
            # Add Visit-Visit Relation
            if node.node_id in visit_visit_relation:
                for visit in visit_visit_relation[node.node_id]:
                    if [node.node_id, visit.node_id] not in adjacency_matrix:
                        adjacency_matrix.append([node.node_id, visit.node_id])
            # Add Visit-Diagnosis Relation
            for diagnosis in visit_diagnosis_relation[node.node_id]:
                if [node.node_id, diagnosis.node_id] not in adjacency_matrix:
                    adjacency_matrix.append([node.node_id, diagnosis.node_id])
                # Add Diagnosis-Service Relation
                for service in diagnosis_service_relation[diagnosis.node_id]:
                    if [diagnosis.node_id, service.node_id] not in adjacency_matrix:
                        adjacency_matrix.append([diagnosis.node_id, service.node_id])

        graph_id_index += 1


    if os.path.exists(
            os.path.join(base_directory, preprocessed_directory,
                         f'adjacency_matrix_ehr_data_{suffix}.txt')):
        os.remove(os.path.join(base_directory, preprocessed_directory,
                               f'adjacency_matrix_ehr_data_{suffix}.txt'))
    with open(os.path.join(base_directory, preprocessed_directory,
                           f'adjacency_matrix_ehr_data_{suffix}.txt'),
              'w+') as f:
        f.write('\n'.join([','.join([str(x) for x in line]) for line in adjacency_matrix]))

    if os.path.exists(
            os.path.join(base_directory, preprocessed_directory, f'nodes_ehr_data_{suffix}.txt')):
        os.remove(os.path.join(base_directory, preprocessed_directory, f'nodes_ehr_data_{suffix}.txt'))
    with open(os.path.join(base_directory, preprocessed_directory, f'nodes_ehr_data_{suffix}.txt'),
              'w+') as f:
        f.write('\n'.join([','.join([str(node.node_id), str(node.type)]) for node in all_nodes]))

    if os.path.exists(os.path.join(base_directory, preprocessed_directory,
                                   f'node_features_ehr_data_{suffix}.txt')):
        os.remove(
            os.path.join(base_directory, preprocessed_directory, f'node_features_ehr_data_{suffix}.txt'))
    with open(
            os.path.join(base_directory, preprocessed_directory, f'node_features_ehr_data_{suffix}.txt'),
            'w+') as f:
        f.write('\n'.join([','.join([str(feature) for feature in node.features]) for node in all_nodes]))

    if os.path.exists(
            os.path.join(base_directory, preprocessed_directory,
                         f'graph_indicators_ehr_data_{suffix}.txt')):
        os.remove(os.path.join(base_directory, preprocessed_directory,
                               f'graph_indicators_ehr_data_{suffix}.txt'))
    with open(os.path.join(base_directory, preprocessed_directory,
                           f'graph_indicators_ehr_data_{suffix}.txt'),
              'w+') as f:
        f.write('\n'.join([str(line) for line in graph_indicator]))

    if os.path.exists(
            os.path.join(base_directory, preprocessed_directory, f'graph_labels_ehr_data_{suffix}.txt')):
        os.remove(
            os.path.join(base_directory, preprocessed_directory, f'graph_labels_ehr_data_{suffix}.txt'))
    with open(
            os.path.join(base_directory, preprocessed_directory, f'graph_labels_ehr_data_{suffix}.txt'),
            'w+') as f:
        f.write('\n'.join([str(line) for line in graph_labels]))
