# `utils/` Module

This folder contains shared preprocessing utilities, dataset tooling, metrics, and training helpers used across the BeyondTheTokens pipeline.

## Preprocessing (`utils/preprocessing/`)

The preprocessing stack builds AST graphs and enriches them with semantic and positional features before converting them to PyG `Data` objects.

- **`ast_extraction.py`**
  - `extract_ast(...)` builds a `networkx.DiGraph` from Tree-sitter ASTs.
  - `build_node_type_mapping_from_snippets(...)` generates stable node-type IDs.

- **`semantic_extraction.py`**
  - Pools pretrained transformer embeddings into per-node semantic features.

- **`positional_extraction.py`**
  - Computes Laplacian positional encodings and attaches them to nodes.

- **`feature_extraction.py`**
  - `extract_graph_features(...)` orchestrates AST building, semantic extraction, positional encoding, and PyG conversion in one call.

- **Dataset-specific entry points**
  - `java250_preprocessing.py` and `devign_preprocessing.py` build `.pt` graphs for each dataset and emit split CSVs.

### Example: Extract a batch of graphs

```python
from utils.preprocessing.feature_extraction import extract_graph_features

graphs = extract_graph_features(
    snippets=["def foo(): pass"],
    lang="python",
    model=plm,
    tokenizer=tok,
    extract_semantics=True,
    extract_positionals=True,
    positional_feat_dim=32,
)
```

## Training (`utils/training/`)

Training helpers wrap PyTorch Lightning with consistent callbacks, logging, and checkpointing.

- **`training.py`**
  - `train_model(...)` builds the model, trainer, and runs fit/test.

- **`trainer.py`**
  - `build_trainer(...)` and `create_callbacks(...)` standardize checkpoints and early stopping.

- **`hp_tuning.py`**
  - Optuna-based HPO utilities, including pruners and W&B logging hooks.

## Metrics (`utils/metrics/`)

Custom loss + metric helpers for multi-class or binary classification, including class-weighting and threshold-aware metrics.
