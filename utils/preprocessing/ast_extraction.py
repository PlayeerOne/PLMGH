# =============================================================================
# AST → Directed Graph utilities
# -----------------------------------------------------------------------------
# This module builds a NetworkX DiGraph from source code using `code_ast`,
# collects node types, and (optionally) assigns a stable integer index per
# node type. Nodes are uniquely identified by their "path" from the root,
# represented as a tuple of child indices (e.g., (), (0,), (0, 3), ...).
#
# Node attributes set by `get_ast`:
#   - "type":        parser-specific node type string
#   - "start_byte":  start byte offset in the source
#   - "end_byte":    end byte offset in the source
#   - "path":        list of child indices from root to this node
#
# Optional attribute set by `assign_nodetypeidx` or `extract_ast`:
#   - attribute_name (default "node_type_idx"): stable int id for the node type
#
# Design notes:
# - We keep edges parent → child (top-down). This is convenient for traversals
#   and respects the AST hierarchy.
# - Node keys are tuples (the immutable form of "path") to be hashable and
#   stable without extra id bookkeeping.
# - `get_nodetypes` can either:
#     (a) return a mapping {node_type: idx}, or
#     (b) write the index into each graph and return (graphs_with_idx, mapping).
# - All functions are pure with respect to the input source; only the graph
#   object is mutated when assigning attributes.
#
# Performance & memory:
# - `get_nodetypes(assign=True)` materializes the graphs iterable to allow two
#   passes (collect types, then assign). For large corpora, consider streaming
#   with assign=False to obtain the mapping first, then re-iterate to assign.
# - Recursion in `get_ast.traverse` assumes reasonably bounded AST depth.
#   Extremely deep trees may require refactoring to an explicit stack.
# =============================================================================
import warnings
import networkx as nx
import code_ast
from typing import Dict, Iterable, Optional, Union, Tuple, List
from tqdm import tqdm as _tqdm

def build_node_type_mapping_from_snippets(
    snippets_dict,
    snippet_key,
    lang: str,
    show_progress: bool = True,
) -> Dict[str, int]:
    
    node_types: set = set()

    # Deterministic iteration order (ids may be numeric strings or mixed)
    keys = list(snippets_dict.keys())
    try:
        # If keys are sortable as-is (e.g., all strings), this is fine.
        keys.sort()
    except Exception:
        # Fallback: leave as-is if not sortable.
        pass

    iterator = keys
    if show_progress and _tqdm is not None:
        iterator = _tqdm(iterator, desc="Extracting node type mapping")

    for snippet_id in iterator:
        entry = snippets_dict[snippet_id]
        snippet = entry.get(snippet_key, "")
        if not snippet:
            continue  # skip empty snippets safely

        g = extract_ast(source=snippet, lang=lang)
        # Collect node type strings
        node_types.update(nx.get_node_attributes(g, "type").values())
        del g  # free graph ASAP

    # Produce stable indices
    sorted_types = sorted(node_types)
    node_type_to_idx = {t: i for i, t in enumerate(sorted_types)}
    return node_type_to_idx

def extract_ast(
    source: str,
    lang: str,
    node_type_to_index: Optional[Dict[str, int]] = None,
    attribute_name: str = "node_type_idx",
) -> nx.DiGraph:
    """
    Build an AST graph for `source` in language `lang`, and optionally assign a
    stable integer index to each node's type.

    Parameters
    ----------
    source : str
        Raw source code to parse.
    lang : str
        Language identifier understood by `code_ast.ast`.
    node_type_to_index : Optional[Dict[str, int]]
        Mapping from node-type string to stable integer id. If provided (even
        if empty), indices will be assigned to nodes under `attribute_name`.
        If a node type is missing from the mapping, a KeyError is raised.
    attribute_name : str
        Node attribute name under which the integer index will be stored.

    Returns
    -------
    nx.DiGraph
        Directed graph of the AST with node attributes described above.

    Raises
    ------
    KeyError
        If `node_type_to_index` is provided and a node's type is not present.
    """
    # Parse and convert to a DiGraph
    G = get_ast(source, lang)
    # Optionally assign stable integer indices for node types
    if node_type_to_index is not None:  # allow empty dict
        G = assign_nodetypeidx(G, node_type_to_index, attribute_name=attribute_name)
    return G


def assign_nodetypeidx(
    graph: nx.Graph,
    node_type_to_index: Dict[str, int],
    attribute_name: str = "node_type_idx",
) -> nx.Graph:
    """
    Assign a stable integer index to each node based on its "type" attribute.

    Parameters
    ----------
    graph : nx.Graph
        Graph whose nodes have a "type" attribute (string).
    node_type_to_index : Dict[str, int]
        Mapping {node_type: idx}. All node types encountered must be keys here.
    attribute_name : str
        Node attribute key to store the integer index (default "node_type_idx").

    Returns
    -------
    nx.Graph
        The same graph instance with indices assigned in-place.

    Raises
    ------
    KeyError
        If a node type is not found in `node_type_to_index`.

    Notes
    -----
    If you need a soft fallback for unseen types (e.g., "OOV"), you can adapt
    this function to default to a reserved index instead of raising a KeyError.
    """
    # Iterate all nodes; look up the type string; store its index
    for nid in graph.nodes():
        node_type = graph.nodes[nid].get("type")
        if node_type not in node_type_to_index:
            raise KeyError(f"Node type '{node_type}' not found in mapping.")
        graph.nodes[nid][attribute_name] = node_type_to_index[node_type]
    return graph


def get_nodetypes(
    graphs: Iterable[nx.Graph],
    show_progress: bool = True,
    assign: bool = True,
    attribute_name: str = "type_idx",
) -> Union[Dict[str, int], Tuple[List[nx.Graph], Dict[str, int]]]:
    """
    Build a stable mapping {node_type: idx} (sorted).
    If assign=True, also write indices into each graph and return (graphs_with_idx, mapping).
    If assign=False, return mapping only.
    """
    # If we plan to assign, we must be able to iterate twice -> materialize once.
    graphs_list = list(graphs) if assign else graphs

    # Optional progress bar; tqdm works even if length is unknown
    iterator = graphs_list
    if show_progress and _tqdm is not None:
        # tqdm works best with lists (len known); works with iterables too.
        iterator = _tqdm(iterator, desc="Collecting node types")

    node_type_set: set = set()
    for G in iterator:
        for _, attrs in G.nodes(data=True):
            t = attrs.get("type")
            if t is not None:
                node_type_set.add(t)

    # Produce a stable (sorted) mapping to integers [0..N-1]
    sorted_types = sorted(node_type_set)
    node_type_to_idx = {nt: i for i, nt in enumerate(sorted_types)}

    if not assign:
      # Only return the mapping
        return node_type_to_idx

    # Assign indices into each graph under `attribute_name`; return both
    out_graphs: List[nx.Graph] = []
    for graph in graphs_list:  # safe: we materialized when assign=True
        out_graphs.append(assign_nodetypeidx(graph, node_type_to_idx, attribute_name=attribute_name))

    return out_graphs, node_type_to_idx

def get_ast(source: str, lang: str = "java") -> nx.DiGraph:
    """
    Parse `source` into an AST using `code_ast` and convert it into a DiGraph.

    Parameters
    ----------
    source : str
        Raw source code to parse.
    lang : str, default="java"
        Language identifier passed to `code_ast.ast`.

    Returns
    -------
    nx.DiGraph
        Directed graph where nodes are addressed by their path from the root
        (tuple of child indices). Edges are parent→child.

    Implementation details
    ----------------------
    - Suppresses FutureWarning from the underlying parser to avoid noisy logs.
    - Each node gets: "type", "start_byte", "end_byte", "path".
    - Uses a recursive DFS; the `path` list is copied when stored to avoid
      aliasing across nodes.

    Notes
    -----
    The start/end byte offsets refer to the raw byte positions in `source`.
    For multibyte encodings, they may not align with character indices.
    """
    # Silence FutureWarnings potentially emitted by the underlying parser
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        tree = code_ast.ast(source, lang=lang, syntax_error="ignore")
        root = tree.root_node()

    # Create a directed graph and populate via DFS
    G = nx.DiGraph()

    def traverse(node, path):
        # Use the immutable tuple of the path as node identifier
        node_id = tuple(path)

        # Store common AST metadata on the node
        G.add_node(
            node_id,
            type=node.type,
            start_byte=node.start_byte,
            end_byte=node.end_byte,
            path=path.copy(),  # keep a list form for convenience on the attribute
        )

        # Connect to parent if not root
        if path:
            parent_id = tuple(path[:-1])
            G.add_edge(parent_id, node_id)

        # Recurse on children with extended path
        for idx, child in enumerate(node.children):
            traverse(child, path + [idx])

    # Start DFS from the root with an empty path
    traverse(root, [])
    return G
