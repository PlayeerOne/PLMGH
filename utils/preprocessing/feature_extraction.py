# =============================================================================
# Graph Feature Extraction Pipeline
# -----------------------------------------------------------------------------
# This function takes raw code snippets, builds their ASTs (as NetworkX graphs),
# optionally attaches semantic and positional node features, and (optionally)
# converts the result to torch_geometric `Data` objects.
#
# High-level stages:
#   1) Normalize inputs (single string vs list of strings).
#   2) Build ASTs: one NetworkX DiGraph per snippet (parent → child edges).
#      - If `node_type_to_index` is provided, each node gets `node_type_attr_name`.
#   3) Semantic features (optional, batched over the input snippets):
#      - Uses a transformer `model` and `tokenizer` to produce representations
#        and pools them per node (mean/sum/max/first/last) into `semantic_attr_name`.
#   4) Positional features (optional, per-graph):
#      - Adds a fixed-size positional embedding per node under `positional_attr_name`.
#   5) PyG conversion (optional):
#      - Converts NetworkX graphs to torch_geometric `Data` via `from_networkx`.
#
# Returns:
#   - A single graph if the input was a single string.
#   - A list of graphs if the input was an iterable of strings.
#
# Notes & trade-offs:
#   - Semantic extraction is batched with `batch_size=len(graphs)`, which is simple
#     but can be memory-heavy for large batches. Adjust upstream if needed.
#   - If `semantics_device` is None, we infer CUDA if available, else CPU.
#   - `stride=None` is interpreted as 0 (no overlap).
#   - When `to_torch_geometric=True`, node attributes become PyG `Data` fields.
# =============================================================================

import torch
import networkx as nx
from utils.preprocessing.ast_extraction import extract_ast, get_nodetypes
from utils.preprocessing.semantic_extraction import extract_semantic_features
from utils.preprocessing.positional_extraction import extract_positional_features
from torch_geometric.utils import from_networkx


def extract_graph_features(
    snippets,                      # str or List[str]
    lang: str,
    model,
    tokenizer,
    node_type_to_index : dict = None,
    stride=32,                   # None -> 0 (no overlap)
    max_length=None,
    out_dtype=torch.float16,
    autocast_dtype=torch.float16,
    pooling="mean",
    to_torch_geometric = True,
    extract_semantics=True,
    extract_positionals=True,
    positional_feat_dim=32,
    semantic_attr_name="semantic_feats",
    positional_attr_name="positional_feats",
    node_type_attr_name = "node_type_idx",
    semantics_device="cuda",
    positional_device="cuda",
    ):
    """
    Build AST graphs and enrich them with semantic and positional features.

    Parameters
    ----------
    snippets : str or List[str]
        One or many source-code snippets.
    lang : str
        Language identifier understood by the AST extractor.
    model : transformers.PreTrainedModel
        PLM used to compute semantic features (only if `extract_semantics=True`).
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer paired with the model (semantic extraction).
    node_type_to_index : dict, optional
        Mapping {node_type: int}. If provided, each node gets `node_type_attr_name`.
    stride : int, default=32
        Token-window stride used during semantic extraction; `None` → 0.
    max_length : int, optional
        Max model sequence length (semantic extraction).
    out_dtype : torch.dtype, default=torch.float16
        Dtype for stored features.
    autocast_dtype : torch.dtype, default=torch.float16
        Dtype used for autocast during PLM forward pass (semantic extraction).
    pooling : {"mean","sum","max","first","last"}, default="mean"
        How to pool per-node representations across sub-tokens/windows.
    to_torch_geometric : bool, default=True
        If True, convert NetworkX graphs to torch_geometric `Data`.
    extract_semantics : bool, default=True
        If True, compute and attach semantic features under `semantic_attr_name`.
    extract_positionals : bool, default=True
        If True, attach positional features under `positional_attr_name`.
    positional_feat_dim : int, default=32
        Size of positional embedding per node.
    semantic_attr_name : str, default="semantic_feats"
        Node attribute name to store semantic features.
    positional_attr_name : str, default="positional_feats"
        Node attribute name to store positional features.
    node_type_attr_name : str, default="node_type_idx"
        Node attribute name to store node-type indices (if mapping provided).
    semantics_device : str or torch.device, default="cuda"
        Device for semantic model. If None, inferred as CUDA if available else CPU.
    positional_device : str or torch.device, default="cuda"
        Device used by positional feature computation.

    Returns
    -------
    Union[nx.Graph, torch_geometric.data.Data, List[...]]
        - Single graph/Data if input was a single string.
        - List of graphs/Data otherwise.

    Notes
    -----
    - This function assumes `extract_ast` returns a DiGraph with at least a "type"
      attribute per node; if `node_type_to_index` is passed, nodes also receive
      `node_type_attr_name`.
    - Using `batch_size=len(graphs)` for semantic extraction is convenient but may
      increase memory usage for many or long snippets.
    """
    # --- normalize inputs
    single_input = isinstance(snippets, str)
    instances = [snippets] if single_input else list(snippets)
    stride = 0 if stride is None else int(stride)

    # --- build ASTs
    # Each instance becomes a NetworkX DiGraph; if a node-type mapping is given,
    # nodes get an integer attribute under `node_type_attr_name`.
    instances_asts = [extract_ast(source=s, lang=lang, node_type_to_index = node_type_to_index, attribute_name = node_type_attr_name) for s in instances]

    graphs = instances_asts  # start from bare ASTs

    # --- semantic features (batched)
    if extract_semantics:
        pooling = pooling.lower()
        assert pooling in {"mean", "sum", "max", "first", "last"}
        # Choose device automatically if None; otherwise honor provided device.
        if semantics_device is None:
          semantics_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(semantics_device)
        
        # Compute semantic node features for all snippets in one batch.
        # NOTE: batch_size=len(graphs) is simple but can be RAM/VRAM heavy.
        graphs = extract_semantic_features(
            snippets=instances,                # use normalized list
            input_graphs=graphs,
            model=model,
            tokenizer=tokenizer,
            batch_size=len(graphs),            # preprocess in one go
            stride=stride,
            max_length=max_length,
            out_dtype=out_dtype,
            autocast_dtype=autocast_dtype,
            attr_name=semantic_attr_name,
            pooling=pooling,
        )
        # ensure list
        if isinstance(graphs, nx.Graph):
            graphs = [graphs]

    # --- positional features
    # Per-graph pass to add a fixed-size positional embedding to each node.
    if extract_positionals:
        out = []
        for g in graphs:
            g = extract_positional_features(
                input_graph=g,
                positional_feat_dim=positional_feat_dim,
                device=positional_device,
                attribute_name=positional_attr_name,
                out_dtype=out_dtype,
            )
            out.append(g)
        graphs = out

    # --- optional torch_geometric conversion
    # Converts each NetworkX graph into a `Data` object with node attributes mapped.
    if to_torch_geometric:
        out = []
        for g in graphs:
            d = from_networkx(g)
            # Enforce dtypes on the Data object (PyG may have re-tensored attributes)

            if hasattr(d, semantic_attr_name):
                setattr(d, semantic_attr_name, getattr(d, semantic_attr_name).to(out_dtype))

            if hasattr(d, positional_attr_name):
                setattr(d, positional_attr_name, getattr(d, positional_attr_name).to(out_dtype))

            # Embedding indices must be long
            if hasattr(d, node_type_attr_name):
                setattr(d, node_type_attr_name, getattr(d, node_type_attr_name).long())

            out.append(d)
        graphs = out
    
    # Keep API ergonomic: return a single item if input was a single string.
    return graphs[0] if single_input else graphs
