import torch_geometric.nn as geom_nn
from torch import nn

gnn_layer_by_name = {
    "GCN": geom_nn.GCNConv,
    "GAT": geom_nn.GATConv,
    "GraphConv": geom_nn.GraphConv
}


class GNNModel(nn.Module):

    def __init__(
            self, c_in: int, c_hidden: int, c_out: int, num_layers: int = 2, layer_name: str = "GCN",
            dp_rate: float = 0.1, **kwargs):
        """
        GNN model using several possible GNN layers

        Args:
            c_in (int): Dimension of input features
            c_hidden (int): Dimension of hidden features
            c_out (int): Dimension of the output features. Usually number of classes in classification
            num_layers (int): Number of "hidden" graph layers
            layer_name (str): String of the graph layer to use (see gnn_layer_by_name dict)
            dp_rate (float): Dropout rate to apply throughout the network
            **kwargs: Additional arguments for the graph layer (e.g. number of heads for GAT)
        """

        super().__init__()
        gnn_layer = gnn_layer_by_name[layer_name]

        layers = []
        in_channels, out_channels = c_in, c_hidden
        # Build model
        for l_idx in range(num_layers - 1):
            layers += [
                gnn_layer(in_channels=in_channels,
                          out_channels=out_channels,
                          **kwargs),
                nn.ReLU(inplace=True),
                nn.Dropout(dp_rate)
            ]
            in_channels = c_hidden
        layers += [gnn_layer(in_channels=in_channels,
                             out_channels=c_out,
                             **kwargs)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        """

        Args:
            x : Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)

        """

        for l in self.layers:
            # For graph layers, we need to add the "edge_index" tensor as additional input
            # All PyTorch Geometric graph layer inherit the class "MessagePassing", hence
            # we can simply check the class type.
            if isinstance(l, geom_nn.MessagePassing):
                x = l(x, edge_index)
            else:
                x = l(x)
        return x
