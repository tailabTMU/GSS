import torch
from torch.nn import Linear, Dropout
from torch_geometric.nn import BatchNorm
import torch.nn.functional as F
from torch_geometric import nn
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp



class GraphConvGNNFused(torch.nn.Module):
    def __init__(self, metadata, num_classes, device, hidden_channels, num_layers=3, dropout=0.5):
        super(GraphConvGNNFused, self).__init__()
        torch.manual_seed(12345)

        batch = torch.tensor([], dtype=torch.long)
        batch = batch.to(device)
        x = torch.tensor([])
        x = x.to(device)
        self.device = device
        self.initial_batch = batch
        self.initial_x = x

        self.dropout = 0.0
        if 0.0 < dropout < 1.0:
            self.dropout = dropout

        self.convs = torch.nn.ModuleList()

        node_types = []
        for layer_num in range(1, num_layers + 1):
            hetero_conv_dict = {}
            for edge_type in metadata['edges']:
                source = edge_type[0]
                dest = edge_type[2]
                source_size = metadata['num_features'][source]
                dest_size = metadata['num_features'][dest]
                node_types.append(source)
                node_types.append(dest)
                if layer_num > 1:
                    source_size = hidden_channels
                    dest_size = hidden_channels
                if edge_type[0] == edge_type[2]:
                    hetero_conv_dict[edge_type] = nn.GraphConv(source_size, hidden_channels)
                else:
                    hetero_conv_dict[edge_type] = nn.GraphConv((source_size, dest_size), hidden_channels)
            convs = nn.HeteroConv(hetero_conv_dict)
            self.convs.append(convs)

        node_types = list(set(node_types))
        self.batch_norm = torch.nn.ModuleDict({node_type: BatchNorm(hidden_channels) for node_type in node_types})
        linear_hidden = hidden_channels
        self.lin = Linear(linear_hidden, num_classes)

    def forward(self, x_dict, edge_index_dict, batch_dict):
        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            # Apply Relu
            x_dict = {key: F.relu(x_tensor) for key, x_tensor in x_dict.items()}

            x_dict = {key: self.batch_norm[key](x_tensor) for key, x_tensor in x_dict.items()}

        batch = self.initial_batch
        final_readout = self.initial_x
        for key, x_tensor in x_dict.items():
            batch = torch.cat((batch, batch_dict[key]), 0)
            final_readout = torch.cat((final_readout, x_tensor), 0)

        final_readout = gap(final_readout, batch)

        if self.dropout > 0.0:
            final_readout = F.dropout(final_readout, p=self.dropout, training=self.training)

        # 3. Apply a final classifier
        res = self.lin(final_readout)
        return res

    def __repr__(self) -> str:
        return 'GraphConvGNN'


class SelfDistillationGraphConvGNN(torch.nn.Module):
    def __init__(self, metadata, num_classes, device, hidden_channels, num_layers=3, dropout=0.5):
        super(SelfDistillationGraphConvGNN, self).__init__()
        torch.manual_seed(12345)

        batch = torch.tensor([], dtype=torch.long)
        batch = batch.to(device)
        x = torch.tensor([])
        x = x.to(device)
        self.device = device
        self.initial_batch = batch
        self.initial_x = x

        self.dropout = 0.0
        if 0.0 < dropout < 1.0:
            self.dropout = dropout

        self.convs = torch.nn.ModuleList()

        node_types = []
        for layer_num in range(1, num_layers + 1):
            hetero_conv_dict = {}
            for edge_type in metadata['edges']:
                source = edge_type[0]
                dest = edge_type[2]
                source_size = metadata['num_features'][source]
                dest_size = metadata['num_features'][dest]
                node_types.append(source)
                node_types.append(dest)
                if layer_num > 1:
                    source_size = hidden_channels
                    dest_size = hidden_channels
                if edge_type[0] == edge_type[2]:
                    hetero_conv_dict[edge_type] = nn.GraphConv(source_size, hidden_channels)
                else:
                    hetero_conv_dict[edge_type] = nn.GraphConv((source_size, dest_size), hidden_channels)
            convs = nn.HeteroConv(hetero_conv_dict)
            self.convs.append(convs)

        node_types = list(set(node_types))
        self.batch_norm = torch.nn.ModuleDict({node_type: BatchNorm(hidden_channels) for node_type in node_types})

        linear_hidden = hidden_channels

        self.fcs = torch.nn.ModuleList([Linear(linear_hidden, num_classes) for _ in range(num_layers)])

    def forward(self, x_dict, edge_index_dict, batch_dict):
        feature_list = []
        output_list = []

        for i, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            # Apply Relu
            x_dict = {key: F.relu(x_tensor) for key, x_tensor in x_dict.items()}

            x_dict = {key: self.batch_norm[key](x_tensor) for key, x_tensor in x_dict.items()}

            batch = self.initial_batch
            final_readout = self.initial_x
            for key, x_tensor in x_dict.items():
                batch = torch.cat((batch, batch_dict[key]), 0)
                final_readout = torch.cat((final_readout, x_tensor), 0)

            final_readout = gap(final_readout, batch)

            feature_list.append(final_readout)

        for idx, fc in enumerate(self.fcs):
            final_readout = feature_list[idx]
            if self.dropout > 0.0:
                final_readout = F.dropout(final_readout, p=self.dropout, training=self.training)
            output_list.append(fc(final_readout))

        return output_list, feature_list

    def __repr__(self) -> str:
        return 'SelfDistillationGraphConvGNN'
