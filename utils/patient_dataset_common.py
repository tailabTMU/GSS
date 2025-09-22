import torch
import torch.nn.functional as F
import os.path as osp


class Node:
    counter = 0

    def __init__(self, node_type: str, node_features=None):
        if node_features is None:
            node_features = []
        self.node_id = Node.counter
        self.type = node_type
        self.features = [int(x) for x in node_features]
        self._increment()

    @staticmethod
    def _encode_features(features, needs_to_be_encoded: bool):
        features = torch.tensor(features).to(torch.long).squeeze()
        if features.dim() == 1:
            features = features.unsqueeze(-1)
        features = features - features.min(dim=0)[0]
        features = features.unbind(dim=-1)
        if needs_to_be_encoded:
            features = [F.one_hot(e, num_classes=-1) for e in features]
        return torch.cat(features, dim=-1).to(torch.float)

    @staticmethod
    def _increment():
        Node.counter += 1

    def __str__(self):
        return f"Node {self.node_id}, a {self.type} node, with {len(self.features)} {'feature' if len(self.features) == 1 else 'features'}"
    def __repr__(self) -> str:
        return f"Node{self.node_id}-{self.type}-len({len(self.features)})"


def process_edge_index(edge_index):
    starting_node = [edge[0] for edge in edge_index]
    ending_node = [edge[1] for edge in edge_index]
    edge_index = torch.tensor([starting_node,
                               ending_node], dtype=torch.long)
    # num_nodes = edge_index.max().item() + 1
    # edge_index, _ = remove_self_loops(edge_index)
    # edge_index, _ = coalesce(edge_index, None, num_nodes,
    #                          num_nodes)
    return edge_index


def process_features(features, is_categorical=False):
    if features.dim() == 1:
        features = features.unsqueeze(-1)
    if is_categorical:
        features = features - features.min(dim=0)[0]
        features = features.unbind(dim=-1)
        features = [F.one_hot(x, num_classes=-1) for x in features]
        features = torch.cat(features, dim=-1).to(torch.float)
    # Turn features to float
    features = features.to(torch.float)
    return features


def read_file(folder, prefix, name, to_number=None, dtype=None, to_tensor=False):
    path = osp.join(folder, '{}{}{}.txt'.format(prefix, '_' if prefix.strip() != '' else '', name))
    return read_txt_array(path, sep=',', dtype=dtype, to_tensor=to_tensor)


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None


def read_txt_array(path, sep=None, start=0, end=None, dtype=None, to_tensor=False):
    to_number = None
    if dtype is not None and torch.is_floating_point(torch.empty(0, dtype=dtype)):
        to_number = float
    elif dtype is not None:
        to_number = int
    with open(path, 'r') as f:
        src = f.read().split('\n')

    if to_number is not None:
        src = [[to_number(x) for x in line.split(sep)[start:end]] for line in src]
    else:
        src = [[x for x in line.split(sep)[start:end]] for line in src]

    if to_tensor:
        src = torch.tensor(src).to(dtype).squeeze()
    return src
