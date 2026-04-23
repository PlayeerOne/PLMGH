import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GATConv, TransformerConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.utils import to_undirected, add_self_loops

import torch_geometric

class BaseGNNEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        depth: int,
        conv_layer,
        conv_kwargs: dict,
        dropout: float = 0.25,
        negative_slope: float = 0.2,
        activation: str = "ReLU",
        use_norm: bool = True,
        norm_type: str = "graph_norm",
        add_self_loop: bool = True,
        make_symmetrical: bool = True,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.depth = depth
        self.dropout = dropout
        self.use_norm = use_norm
        self.norm_type = norm_type
        self.add_self_loop = add_self_loop
        self.make_symmetrical = make_symmetrical
        self.conv_kwargs = conv_kwargs

        activations = {
            'ReLU': F.relu,
            'LeakyReLU': lambda x: F.leaky_relu(x, negative_slope=negative_slope),
            'Sigmoid': torch.sigmoid,
            'Tanh': torch.tanh,
            'ELU': F.elu,
            'GELU': F.gelu
        }
        if activation not in activations:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = activations[activation]

        # Build convolutional layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if use_norm else None
        #Add the GNN convs 
        if depth > 1 :
            for layer_idx in range(depth):
                # Determine dims for this layer
                in_size = in_dim if layer_idx == 0 else hidden_dim
                out_size = out_dim if layer_idx == depth - 1 else hidden_dim
    
                # Create conv and optional norm
                conv = conv_layer(in_size, out_size, **conv_kwargs)
                self.convs.append(conv)
                if use_norm:
                    #GraphNorm
                    if norm_type == "graph_norm" :
                        self.norms.append(GraphNorm(out_size))
                    #LayerNorm
                    elif norm_type == "layer_norm" :
                        self.norms.append(nn.LayerNorm(out_size))
                    else:
                        raise ValueError(f"Unknown norm_type `{self.norm_type}`")
        else :
            self.convs.append(conv_layer(in_dim, out_dim, **conv_kwargs))
            if use_norm:
                #GraphNorm
                if norm_type == "graph_norm" :
                    self.norms.append(GraphNorm(out_dim))
                #LayerNorm
                elif norm_type == "layer_norm" :
                    self.norms.append(nn.LayerNorm(out_dim))
                else:
                    raise ValueError(f"Unknown norm_type `{self.norm_type}`")

    def forward(self, data: torch_geometric.data.Data):
        x, edge_index = data.x, data.edge_index
        # Optionally enforce symmetric and self-loops
        if self.make_symmetrical:
            edge_index = to_undirected(edge_index)
        if self.add_self_loop:
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Pass through GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            # apply activation & dropout except on last layer
            if i < len(self.convs) - 1:
                if self.use_norm:
                    #LayerNorm
                    if self.norm_type == "layer_norm":
                        x = self.norms[i](x)
                    #GraphNorm
                    elif self.norm_type == "graph_norm":
                        x = self.norms[i](x, data.batch)
                    else:
                        raise ValueError(f"Unknown norm_type `{self.norm_type}`")
                x = self.activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x
        
class GCN_Encoder(BaseGNNEncoder):
    def __init__(self, 
                 in_dim, 
                 out_dim, 
                 hidden_dim, 
                 depth=2,
                 dropout=0.0, 
                 negative_slope: float = 0.2,
                 activation="ReLU", 
                 use_norm=True,
                 norm_type: str = "graph_norm"
                 ):
        
        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            depth=depth,
            conv_layer=GCNConv,
            conv_kwargs={
                "improved" : False,
                "cached" : False,
                "add_self_loops" : True, #We add self loops ourselves.
                "normalize" : True,
                "bias" : True
            },
            dropout=dropout,
            negative_slope = negative_slope,
            activation=activation,
            use_norm=use_norm,
            norm_type = norm_type,
            add_self_loop = True,
            make_symmetrical = True,
        )
        
class GAT_Encoder(BaseGNNEncoder):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        depth: int = 2,
        heads: int = 1,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        activation: str = "ReLU",
        use_norm: bool = True,
        norm_type: str = "graph_norm",
    ):
        # For GAT, conv_kwargs specifies heads and related params
        conv_kwargs = {
            'heads': heads,
            'negative_slope': negative_slope,
            "add_self_loops" : False, #Self loops we add self loops ourselves
            'bias': True,
            'residual': True,
            "concat" : True if heads >1 else False
        }
        super().__init__(
            in_dim = in_dim,
            hidden_dim = hidden_dim//heads,
            out_dim =out_dim,
            depth = depth,
            conv_layer = GATConv,
            conv_kwargs = conv_kwargs,
            dropout = dropout,
            negative_slope = negative_slope,
            activation = activation,
            use_norm = use_norm,
            norm_type = norm_type,
            add_self_loop = True,
            make_symmetrical = True,
        )

        # Build convolutional layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if use_norm else None
        #Add the GNN convs 
        if depth > 1 :
            for layer_idx in range(depth):
                # Determine dims for this layer
                in_size = in_dim if layer_idx == 0 else hidden_dim
                out_size = out_dim if layer_idx == depth - 1 else hidden_dim
                # Create conv and optional norm
                if self.conv_kwargs["concat"]:
                    conv = GATConv(in_size, out_size//heads, **conv_kwargs)
                else :conv = GATConv(in_size, out_size, **conv_kwargs)
                self.convs.append(conv)
                if use_norm:
                    #GraphNorm
                    if norm_type == "graph_norm":
                        self.norms.append(GraphNorm(out_size))
                    #LayerNorm
                    elif norm_type == "layer_norm":
                        self.norms.append(nn.LayerNorm(out_size))
        else :
            if self.conv_kwargs["concat"]:
                self.convs.append(GATConv(in_dim, out_dim//heads, **conv_kwargs))
            else : self.convs.append(GATConv(in_dim, out_dim, **conv_kwargs))
                
            if use_norm:
                #GraphNorm
                if norm_type == "graph_norm":
                    self.norms.append(GraphNorm(out_dim))
                #LayerNorm
                elif norm_type == "layer_norm":
                    self.norms.append(nn.LayerNorm(out_dim))


class GraphTransformer_Encoder(BaseGNNEncoder):
    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 hidden_dim: int,
                 depth: int = 2,
                 heads: int = 1,
                 dropout: float = 0.0,
                 negative_slope: float = 0.2,
                 activation: str = "ReLU",
                 use_norm: bool = True,
                 norm_type : str = "graph_norm",
                ):
        
        conv_kwargs = {
            'heads': heads,
            'bias': True,
            "concat" : True if heads >1 else False
        }
        super().__init__(
            in_dim = in_dim,
            hidden_dim = hidden_dim//heads,
            out_dim =out_dim,
            depth = depth,
            conv_layer = TransformerConv,
            conv_kwargs = conv_kwargs,
            dropout = dropout, 
            negative_slope = negative_slope,
            activation = activation,
            use_norm = use_norm,
            norm_type = norm_type,
            add_self_loop = True,
            make_symmetrical = True,
        )

        # Build convolutional layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList() if use_norm else None
        #Add the GNN convs 
        if depth > 1 :
            for layer_idx in range(depth):
                # Determine dims for this layer
                in_size = in_dim if layer_idx == 0 else hidden_dim
                out_size = out_dim if layer_idx == depth - 1 else hidden_dim
                # Create conv and optional norm
                if self.conv_kwargs["concat"]:
                    conv = TransformerConv(in_size, out_size//heads, **conv_kwargs)
                else :conv = TransformerConv(in_size, out_size, **conv_kwargs)
                self.convs.append(conv)
                if use_norm:
                    #GraphNorm
                    if norm_type == "graph_norm":
                        self.norms.append(GraphNorm(out_size))
                    #LayerNorm
                    elif norm_type == "layer_norm":
                        self.norms.append(nn.LayerNorm(out_size))
        else :
            if self.conv_kwargs["concat"]:
                self.convs.append(TransformerConv(in_dim, out_dim//heads, **conv_kwargs))
            else : self.convs.append(TransformerConv(in_dim, out_dim, **conv_kwargs))
                
            if use_norm:
                #GraphNorm
                if norm_type == "graph_norm":
                    self.norms.append(GraphNorm(out_dim))
                #LayerNorm
                elif norm_type == "layer_norm":
                    self.norms.append(nn.LayerNorm(out_dim))
        #Send to the appropriate device
        
        