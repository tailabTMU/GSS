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


def read_fused_data(raw_data_dir, base_directory=None, preprocessed_directory='preprocessed', recreate=False,
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
            os.path.join(base_directory, preprocessed_directory, f'adjacency_matrix_fused_data_{suffix}.txt')):
        all_preprocessed_files_exist = False
    elif not os.path.exists(os.path.join(base_directory, preprocessed_directory, f'nodes_fused_data_{suffix}.txt')):
        all_preprocessed_files_exist = False
    elif not os.path.exists(
            os.path.join(base_directory, preprocessed_directory, f'node_features_fused_data_{suffix}.txt')):
        all_preprocessed_files_exist = False
    elif not os.path.exists(
            os.path.join(base_directory, preprocessed_directory, f'graph_indicators_fused_data_{suffix}.txt')):
        all_preprocessed_files_exist = False
    elif not os.path.exists(
            os.path.join(base_directory, preprocessed_directory, f'graph_labels_fused_data_{suffix}.txt')):
        all_preprocessed_files_exist = False
    if all_preprocessed_files_exist is False:
        create_files(raw_data_dir, base_directory, preprocessed_directory)

    print("Loading Files")

    batch = read_file(os.path.join(base_directory, preprocessed_directory), '',
                      f'graph_indicators_fused_data_{suffix}',
                      dtype=torch.long, to_tensor=True)
    node_features = read_file(os.path.join(base_directory, preprocessed_directory), '',
                              f'node_features_fused_data_{suffix}',
                              dtype=torch.long)
    nodes = read_file(os.path.join(base_directory, preprocessed_directory), '', f'nodes_fused_data_{suffix}')

    nodes = [[int(node[0]), node[1]] for node in nodes]
    # num_cpu_workers = 2
    num_cpu_workers = math.floor(cpu_count() / 2)

    # node_types = ['patient', 'event_timeline', 'event', 'behaviour', 'emotion', 'activity', 'ed_visit', 'diagnosis',
    #               'service']
    node_types = ['patient', 'event_timeline', 'event', 'concern', 'ed_visit', 'diagnosis', 'service']
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
                             f'processed_{node_type}_fused_data_{suffix}.pkl')) or all_preprocessed_files_exist is True:
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
                                     f'processed_{node_type}_fused_data_{suffix}.pkl'))
        else:
            node_type_features = joblib.load(
                os.path.join(base_directory, preprocessed_directory,
                             f'processed_{node_type}_fused_data_{suffix}.pkl'))
        node_features_dict[node_type] = node_type_features

    y = read_file(os.path.join(base_directory, preprocessed_directory), '', f'graph_labels_fused_data_{suffix}',
                  dtype=torch.long,
                  to_tensor=True)
    _, y = y.unique(sorted=True, return_inverse=True)

    edge_index = read_file(os.path.join(base_directory, preprocessed_directory), '',
                           f'adjacency_matrix_fused_data_{suffix}',
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

    level_two_node_types = node_types.copy()
    level_two_node_types.remove('patient')
    level_two_node_types.remove('event')
    level_two_node_types.remove('ed_visit')
    level_two_node_types.remove('diagnosis')
    level_two_node_types.remove('service')
    edge_indices_dict = {}
    for node_type in level_two_node_types:
        edge_indices_dict[f'patient_{node_type}'] = None
        edge_indices_dict[f'patient_{node_type}'] = [
            [patient_node_type_indices['patient'].index(e[0]), patient_node_type_indices[node_type].index(e[1])]
            for e in edge_index
            if (e[0] in patient_node_type_indices['patient'] and e[1] in patient_node_type_indices[node_type])
        ]
        edge_indices_dict[f'patient_{node_type}'] = process_edge_index(edge_indices_dict[f'patient_{node_type}'])
        if node_type == 'event_timeline':
            for relation in ['event_timeline-event', 'event_timeline-ed_visit', 'ed_visit-diagnosis',
                             'ed_visit-ed_visit', 'diagnosis-service']:
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
            # node_type = 'event'
            # edge_indices_dict[f'event_timeline_{node_type}'] = None
            # edge_indices_dict[f'event_timeline_{node_type}'] = [
            #     [patient_node_type_indices['event_timeline'].index(e[0]),
            #      patient_node_type_indices[node_type].index(e[1])]
            #     for e in edge_index
            #     if
            #     (e[0] in patient_node_type_indices['event_timeline'] and e[1] in patient_node_type_indices[node_type])
            # ]
            # edge_indices_dict[f'event_timeline_{node_type}'] = process_edge_index(
            #     edge_indices_dict[f'event_timeline_{node_type}'])


    # Get Patient Graph Label
    patient_y = y[(graph_id - 1)]
    data = HeteroData()

    for node_type in node_types:
        data[node_type].x = node_type_features_dict[node_type]

    for node_type in level_two_node_types:
        data['patient', f'patient_{node_type}', node_type].edge_index = edge_indices_dict[f'patient_{node_type}']
        data['patient', f'patient_{node_type}', node_type].edge_attr = np.array([])
        if node_type == 'event_timeline':
            for relation in ['event_timeline-event', 'event_timeline-ed_visit', 'ed_visit-diagnosis',
                             'ed_visit-ed_visit', 'diagnosis-service']:
                relation_types = relation.split('-')
                node_type_1 = relation_types[0]
                node_type_2 = relation_types[1]
                data[node_type_1, f'{node_type_1}_{node_type_2}', node_type_2].edge_index = edge_indices_dict[
                    f'{node_type_1}_{node_type_2}']
                data[node_type_1, f'{node_type_1}_{node_type_2}', node_type_2].edge_attr = np.array([])
            # node_type = 'event'
            # data['event_timeline', f'event_timeline_{node_type}', node_type].edge_index = edge_indices_dict[
            #     f'event_timeline_{node_type}']
            # data['event_timeline', f'event_timeline_{node_type}', node_type].edge_attr = np.array([])


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

    # path = osp.join(raw_data_dir, 'fused_questionnaires.csv')
    # ed_path = osp.join(raw_data_dir, 'fused_ed.csv')

    path = osp.join(raw_data_dir, 'fixed_questionnaires.csv')
    ed_path = osp.join(raw_data_dir, 'fixed_ed.csv')

    # columns = [
    #     'C_248',
    #     'C_92',
    #     'C_33',
    #     'C_301',
    #     'C_238',
    #     'C_142',
    #     'C_970',
    #     'C_343',
    #     'C_239',
    #     'C_974',
    #     'C_333',
    #     'C_99',
    #     'C_1132',
    #     'C_29',
    #     'C_141',
    #     'C_1135',
    #     'C_243',
    #     'C_84',
    #     # 'C_189',  # Siblings
    #     'C_276',
    #     'C_321',
    #     'C_246',
    #     'C_344',
    #     'C_247',
    #     # '180-days ED visit label'
    #     '180_days_ed_visit_label'
    # ]

    columns = [
        'C_974',
        'C_1057',
        'C_971',
        'C_1091',
        'C_969',
        'C_1067',
        'C_1092',
        'C_970',
        'C_1098',
        'C_1093',
        'C_1094',
        '180_days_ed_visit_label'
    ]

    # node_question_mappings = {
    #     'patient': [['C_33', 'C_84', 'C_321', 'C_247']],
    #     'behaviour': [['C_248'], ['C_301'], ['C_333'], ['C_1135'], ['C_246']],
    #     'activity': [['C_92'], ['C_99']],
    #     'emotion': [['C_1132'], ['C_276']]
    # }

    node_question_mappings = {
        'patient': [['C_971', 'C_1091', 'C_1092', 'C_1098', 'C_1093', 'C_1094']],
        'concern': [['C_1057'], ['C_1067']],
    }

    concern_keys = {
        'C_1057': {'val': 0, 'label': 'physical_appearance', 'from': 0, 'to': 1},
        'C_1067': {'val': 1, 'label': 'physically_hearting_self', 'from': 2, 'to': 3},
    }

    # TODO: Handle All Missing Values
    # Vague Missing Values (Missing a checkbox response.) -->  97
    # Valid Skips -->  99
    # Other Missing Values -->  98

    events = {
        # TODO: Handle Missing Values and Never
        'not_applicable': {'val': 0, 'nodes': {'C_974': [3]}},
        'never': {'val': 1, 'nodes': {'C_974': [2, 98, 99], 'C_969': [2, 98, 99], 'C_970': [2, 98, 99]}},
        # 'past_4': {'val': 2, 'nodes': {}},
        'past_12': {'val': 3, 'nodes': {'C_974': [1], 'C_969': [1], 'C_970': [1]}},
        # 'past_24': {'val': 4, 'nodes': {}},
        # 'past_36': {'val': 5, 'nodes': {}},
        # 'more_than_36': {'val': 6, 'nodes': {}},
    }

    event_timeline_values = {
        'not_applicable': 0,
        'never': 1,
        'past_4': 2,
        'past_12': 3,
        'past_24': 4,
        'past_36': 5,
        'more_than_36': 6,
    }

    event_keys = {
        'C_974': {'val': 0, 'label': 'expelled_from_school'},
        'C_969': {'val': 1, 'label': 'mh_talk_to_clinician'},
        'C_970': {'val': 2, 'label': 'mh_er_visit_parent'},
    }

    columns_without_missing_vals = ['C_1057', 'C_1067']

    # questionnaire_id_col_name = 'C_0'
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
        for node_type in node_question_mappings:
            nodes_dict[node_type] = []
            nodes = node_question_mappings[node_type]
            for items_in_node in nodes:
                if len(items_in_node) > 1:
                    node_val = [patient_data[q_id] if ((q_id in columns_without_missing_vals) or (
                            int(patient_data[q_id]) not in [97, 98, 99])) else -1 for q_id in items_in_node]
                else:
                    q_id = items_in_node[0]
                    if (q_id not in columns_without_missing_vals) and (int(patient_data[q_id]) in [97, 98, 99]):
                        # TODO: Handle Missing Values
                        node_val = 0
                    else:
                        node_val = patient_data[q_id]
                    if node_type == 'concern':
                        node_val = node_val + concern_keys[q_id]['from']
                    node_val = [node_val]
                if len(node_val) > 0:
                    node_obj = Node(node_type=node_type, node_features=node_val)
                    nodes_dict[node_type].append(node_obj)
                    all_nodes.append(node_obj)
                    graph_indicator.append(graph_id_index)

        # Handling Event Timeline
        timeline_event_relation = {}
        event_timeline_ids = {}
        for event_time in events:
            nodes = events[event_time]['nodes']
            node_type = 'event'
            nodes_val = []
            for qid in nodes:
                if int(patient_data[qid]) in nodes[qid]:
                    event_val = event_keys[qid]['val']
                    nodes_val.append([event_val])
            if len(nodes_val) > 0:
                # Add Timeline
                node_val = [event_timeline_values[event_time]]
                node_obj = Node(node_type='event_timeline', node_features=node_val)
                event_time_id = node_obj.node_id
                if 'event_timeline' not in nodes_dict:
                    nodes_dict['event_timeline'] = []
                if event_time_id not in timeline_event_relation:
                    timeline_event_relation[event_time_id] = []
                nodes_dict['event_timeline'].append(node_obj)
                event_timeline_ids[event_time] = node_obj
                all_nodes.append(node_obj)
                graph_indicator.append(graph_id_index)

                # Add Event
                if node_type not in nodes_dict:
                    nodes_dict[node_type] = []
                for single_node_value in nodes_val:
                    node_obj = Node(node_type=node_type, node_features=single_node_value)
                    nodes_dict[node_type].append(node_obj)
                    all_nodes.append(node_obj)
                    graph_indicator.append(graph_id_index)
                    timeline_event_relation[event_time_id].append(node_obj)

        patient_ed_data_days = list(patient_ed_data['Registration Date'].unique())
        questionnaire_date = parser.parse(patient_data[questionnaire_date_col])
        ed_timeline = {
            'past_4': [],
            'past_12': [],
            'past_24': [],
            'past_36': [],
            'more_than_36': [],
        }
        for date in patient_ed_data_days:
            ed_date = parser.parse(date)
            delta = relativedelta.relativedelta(questionnaire_date, ed_date)
            total_months = (delta.years * 12) + delta.months + (
                    delta.days / 30)  # Just to make it go over 12, 24, or 36
            if total_months <= 4:
                ed_timeline['past_4'].append(date)
            elif total_months <= 12:
                ed_timeline['past_12'].append(date)
            elif total_months <= 24:
                ed_timeline['past_24'].append(date)
            elif total_months <= 36:
                ed_timeline['past_36'].append(date)
            else:
                ed_timeline['more_than_36'].append(date)
        # Handling Event Timeline
        visit_visit_relation = {}
        visit_diagnosis_relation = {}
        diagnosis_service_relation = {}
        added_adjacency = []
        services = {}
        diagnosis = {}
        previous_visit = None
        for event_time in ed_timeline:
            nodes = ed_timeline[event_time]
            if len(nodes) > 0:
                if event_time in event_timeline_ids:
                    # Get Event Timeline Node
                    event_timeline_obj = event_timeline_ids[event_time]
                    event_time_id = event_timeline_obj.node_id
                else:
                    # Add Event Timeline Node
                    node_val = [event_timeline_values[event_time]]
                    event_timeline_obj = Node(node_type='event_timeline', node_features=node_val)
                    event_time_id = event_timeline_obj.node_id
                    if 'event_timeline' not in nodes_dict:
                        nodes_dict['event_timeline'] = []
                    if event_time_id not in timeline_event_relation:
                        timeline_event_relation[event_time_id] = []
                    nodes_dict['event_timeline'].append(event_timeline_obj)
                    event_timeline_ids[event_time] = event_timeline_obj
                    all_nodes.append(event_timeline_obj)
                    graph_indicator.append(graph_id_index)
                for visit_index, visit_date in enumerate(nodes, start=1):
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
                    timeline_event_relation[event_time_id].append(visit_node)

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

        patient_node = nodes_dict['patient'][0]
        # Patient to Event Timeline
        if len(nodes_dict['event_timeline']) > 0:
            for node in nodes_dict['event_timeline']:
                adjacency_matrix.append([patient_node.node_id, node.node_id])
                # Event Timeline to Event and ED Visit
                for event in timeline_event_relation[node.node_id]:
                    adjacency_matrix.append([node.node_id, event.node_id])
                    # Add ED Visit Subgraph
                    if event.type == 'ed_visit':
                        # Add Visit-Visit Relation
                        if event.node_id in visit_visit_relation:
                            for visit in visit_visit_relation[event.node_id]:
                                if [event.node_id, visit.node_id] not in adjacency_matrix:
                                    adjacency_matrix.append([event.node_id, visit.node_id])
                        # Add Visit-Diagnosis Relation
                        for diagnosis in visit_diagnosis_relation[event.node_id]:
                            if [event.node_id, diagnosis.node_id] not in adjacency_matrix:
                                adjacency_matrix.append([event.node_id, diagnosis.node_id])
                            # Add Diagnosis-Service Relation
                            for service in diagnosis_service_relation[diagnosis.node_id]:
                                if [diagnosis.node_id, service.node_id] not in adjacency_matrix:
                                    adjacency_matrix.append([diagnosis.node_id, service.node_id])

        # Patient to Concern
        if len(nodes_dict['concern']) > 0:
            for node in nodes_dict['concern']:
                adjacency_matrix.append([patient_node.node_id, node.node_id])
        graph_id_index += 1


    if os.path.exists(
            os.path.join(base_directory, preprocessed_directory,
                         f'adjacency_matrix_fused_data_{suffix}.txt')):
        os.remove(os.path.join(base_directory, preprocessed_directory,
                               f'adjacency_matrix_fused_data_{suffix}.txt'))
    with open(os.path.join(base_directory, preprocessed_directory,
                           f'adjacency_matrix_fused_data_{suffix}.txt'),
              'w+') as f:
        f.write('\n'.join([','.join([str(x) for x in line]) for line in adjacency_matrix]))

    if os.path.exists(
            os.path.join(base_directory, preprocessed_directory, f'nodes_fused_data_{suffix}.txt')):
        os.remove(os.path.join(base_directory, preprocessed_directory, f'nodes_fused_data_{suffix}.txt'))
    with open(os.path.join(base_directory, preprocessed_directory, f'nodes_fused_data_{suffix}.txt'),
              'w+') as f:
        f.write('\n'.join([','.join([str(node.node_id), str(node.type)]) for node in all_nodes]))

    if os.path.exists(os.path.join(base_directory, preprocessed_directory,
                                   f'node_features_fused_data_{suffix}.txt')):
        os.remove(
            os.path.join(base_directory, preprocessed_directory, f'node_features_fused_data_{suffix}.txt'))
    with open(
            os.path.join(base_directory, preprocessed_directory, f'node_features_fused_data_{suffix}.txt'),
            'w+') as f:
        f.write('\n'.join([','.join([str(feature) for feature in node.features]) for node in all_nodes]))

    if os.path.exists(
            os.path.join(base_directory, preprocessed_directory,
                         f'graph_indicators_fused_data_{suffix}.txt')):
        os.remove(os.path.join(base_directory, preprocessed_directory,
                               f'graph_indicators_fused_data_{suffix}.txt'))
    with open(os.path.join(base_directory, preprocessed_directory,
                           f'graph_indicators_fused_data_{suffix}.txt'),
              'w+') as f:
        f.write('\n'.join([str(line) for line in graph_indicator]))

    if os.path.exists(
            os.path.join(base_directory, preprocessed_directory, f'graph_labels_fused_data_{suffix}.txt')):
        os.remove(
            os.path.join(base_directory, preprocessed_directory, f'graph_labels_fused_data_{suffix}.txt'))
    with open(
            os.path.join(base_directory, preprocessed_directory, f'graph_labels_fused_data_{suffix}.txt'),
            'w+') as f:
        f.write('\n'.join([str(line) for line in graph_labels]))
