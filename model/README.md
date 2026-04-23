# Model Module

The `model/` package contains the Lightning-ready GNN classifier and its building blocks. The current architecture fuses multiple node-level modalities (node types, positional encodings, semantic transformer embeddings) and supports multiple GNN encoder backbones.

## Key Components

- **`gnn_classifier.py`**
  - `GNN_Classifier` is a `lightning.LightningModule` that wires together:
    1. `FeatureFusionLayer` (multi-modal node embeddings).
    2. A pluggable GNN encoder (GCN, GAT, GraphTransformer).
    3. `Cls_Decoder` for graph pooling + classification.
  - Handles class-imbalance weights, binary vs multi-class metrics, and OneCycleLR scheduling.

- **`fusion_layer.py`**
  - `FeatureFusionLayer` combines node-type, positional, and semantic features.
  - Fusion methods: `concat`, `sum`, or `gated`.

- **`gnn_encoder.py`**
  - Encoder wrappers for `GCN_Encoder`, `GAT_Encoder`, and `GraphTransformer_Encoder`.
  - Shared configuration: depth, hidden/out dims, normalization, activation, dropout.

- **`gnn_decoder.py`**
  - `Cls_Decoder` supports attentional pooling or mean/max/sum to build graph embeddings.

## Expected Graph Inputs

The classifier expects a `torch_geometric.data.Data` graph with node attributes:

- `node_type_idx` (long tensor) for node-type embeddings.
- `positional_feats` (float tensor) for positional encodings.
- `semantic_feats` (float tensor) for transformer embeddings.

These names are configurable in `GNN_Classifier` if you use different keys.

## Minimal Usage

```python
from model.gnn_classifier import GNN_Classifier
from torch_geometric.data import Data

model = GNN_Classifier(
    gnn_type="GAT",
    gnn_depth=4,
    gnn_heads=8,
    gnn_dim=1024,
    fusion_method="concat",
    node_type_dim=256,
    pos_features_dim=32,
    semantic_features_dim=768,
    num_classes=250,
)

# data = Data(
#     node_type_idx=..., positional_feats=..., semantic_feats=..., edge_index=...
# )
logits = model(data)
```

## Notes

- For binary classification (`num_classes=2`), the classifier uses threshold-aware metrics and can apply `pos_weight` for BCE.
- The module is optimized for Lightning training loops via `utils/training/`.
