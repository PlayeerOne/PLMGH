import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch_geometric.nn.aggr import AttentionalAggregation
from torch_geometric.nn.pool import global_mean_pool, global_max_pool, global_add_pool

from torch_geometric.data import Data


class Cls_Decoder(nn.Module):
    def __init__(self, in_dim, h_dim, out_dim=250, dropout=0.25, pooling_method='attentional'):
        super().__init__()

        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.dropout = dropout

        # Gate mechanism: attention mechanism to compute importance of each node's features
        self.gate = nn.Sequential(
            nn.Linear(in_dim, in_dim * 2),
            nn.LayerNorm(in_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim * 2, 1)
        )

        # Apply He initialization to the Linear layers
        init.kaiming_uniform_(self.gate[0].weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.gate[3].weight, mode='fan_in', nonlinearity='relu')

        # Select pooling function based on the provided method
        pooling_methods = {
            'attentional': AttentionalAggregation(gate_nn=self.gate),
            'mean': global_mean_pool,
            'max': global_max_pool,
            'sum': global_add_pool,  # Added sum pooling
        }

        # Ensure valid pooling method
        if pooling_method not in pooling_methods:
            raise ValueError(f"Unsupported pooling method: {pooling_method}. Available methods: 'attentional', 'mean', 'max', 'sum'")

        self.pool_fn = pooling_methods[pooling_method]
        
        # Pool normalization
        self.pool_norm = nn.LayerNorm(in_dim)
        
        # FINAL MLP for classification
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_dim, out_dim),
        )

        # Apply He initialization to the Linear layers
        init.kaiming_uniform_(self.mlp[0].weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_uniform_(self.mlp[4].weight, mode='fan_in')

    def forward(self, input_graph: Data):
        x = input_graph.x  # Node features
        batch_ids = input_graph.batch  # Batch indices for pooling

        # Pool Graph Features
        out = self.pool_norm(self.pool_fn(x, batch_ids)) 
        logits = self.mlp(out)
        return logits
