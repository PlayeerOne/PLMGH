# =============================================================================
# Positional Features via Graph Laplacian Eigenvectors
# -----------------------------------------------------------------------------
# This module computes "structural" / positional embeddings (PEs) for AST nodes
# using the eigenvectors of the (combinatorial) graph Laplacian. The first
# nontrivial k eigenvectors (i.e., skipping the constant mode) are used as
# k-dimensional coordinates for each node.
#
# Overview:
#   - extract_positional_features: end-to-end helper that computes PEs and
#     writes them as a node attribute into a NetworkX graph.
#   - compute_laplacian_pe / _gpu: return the PE matrix and a node indexing map.
#   - assign_positional_features: attaches per-node PE vectors to the graph.
#
# Notes & trade-offs:
#   - The GPU path converts the Laplacian to a dense CUDA tensor and uses
#     `torch.linalg.eigh`, which is O(N^3) in time and O(N^2) in memory; this
#     is fast for small–medium graphs but can be prohibitive for large N.
#   - The CPU path prefers sparse eigensolvers (scipy.sparse.linalg.eigsh) when
#     N is large; it falls back to dense eigh if sparse fails.
#   - Direction is ignored for spectral PEs (the Laplacian is built on the
#     undirected version of the input graph).
#   - The assigned attribute is `attribute_name` (default "positional_feats"),
#     a torch.Tensor of shape (k,) per node.
# =============================================================================

import torch 
import numpy as np
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def extract_positional_features(input_graph : nx.Graph, 
    positional_feat_dim : int = 16,
    device = "cuda", 
    attribute_name : str = "positional_feats", 
    out_dtype: torch.dtype = torch.float16
    ):
  """
  Compute and attach positional embeddings (PEs) to `input_graph` nodes.

  The PEs are the first `positional_feat_dim` nontrivial eigenvectors of the
  combinatorial Laplacian of the (undirected) graph. Each node receives a
  k-dimensional torch tensor under `attribute_name`.

  Parameters
  ----------
  input_graph : nx.Graph
      Input AST graph (directed or undirected). Direction is ignored.
  positional_feat_dim : int, default=16
      Number of nontrivial Laplacian eigenvectors to keep (k).
  device : str or torch.device, default="cuda"
      Execution device. If "cuda", uses dense GPU eigendecomposition;
      otherwise, CPU path with dense or sparse routines.
  attribute_name : str, default="positional_feats"
      Node attribute name to store the k-dim positional vector.
  out_dtype : torch.dtype, default=torch.float16
      Dtype used for the stored per-node PE tensors.

  Returns
  -------
  nx.Graph
      The same graph with node attribute `attribute_name` (torch tensor, shape (k,)).

  Notes
  -----
  - Output tensors are stored on CPU (PEs are moved to .cpu() on the GPU path).
  - For small graphs, both paths are typically fast; for large graphs, prefer
    the CPU sparse path by setting `device` to something other than "cuda".
  """
  # Output: networkx graph with positional features (tensors on CPU)
  #Output networkx graph with positional features
  #Output will be in cpu
  pe, node_order = compute_laplacian_pe(G = input_graph,
                                        k = positional_feat_dim,
                                        device = device)
  out_graph = assign_positional_features(
      G = input_graph,
      pe = pe,
      node_index = node_order,
      attr_name = attribute_name,
      out_dtype = out_dtype
      )
  return out_graph




def assign_positional_features(
    G: nx.Graph,
    pe: torch.Tensor,                    
    node_index,                  
    attr_name: str = "positional_feats",
    out_dtype: torch.dtype = torch.float32
):
  """
  Write positional embeddings into each node of G.

  Parameters
  ----------
  G : nx.Graph
      Graph whose nodes will receive PEs.
  pe : torch.Tensor
      Tensor of shape (N, k). Row i corresponds to node with index i.
  node_index : dict
      Mapping {node_id: i} that aligns graph nodes to PE rows.
  attr_name : str, default="positional_feats"
      Node attribute name to store the k-dim PE tensor.
  out_dtype : torch.dtype, default=torch.float32
      Declared dtype (not explicitly cast here). Kept for API symmetry.

  Returns
  -------
  nx.Graph
      Same graph with `attr_name` set on every node.

  Notes
  -----
  - Assumes `pe` is on CPU already. If not, ensure `.cpu()` before storage.
  - This function mutates `G` in place and also returns it for convenience.
  """
  pe_local = pe.detach().to(dtype=out_dtype, device=pe.device, copy=False)
  for nid in G.nodes():
      idx = node_index[nid]            
      G.nodes[nid][attr_name] = pe_local[idx]
  return G




def compute_laplacian_pe_gpu(G: nx.DiGraph, k: int = 16):
  """
  Compute the first k nontrivial Laplacian eigenvectors on GPU (dense).

  Steps:
    1) Convert to undirected graph and fix a stable node ordering.
    2) Build sparse adjacency and Laplacian (CPU).
    3) Materialize dense Laplacian and move to CUDA.
    4) Run symmetric eigendecomposition with torch.linalg.eigh.
    5) Drop the trivial eigenvector (constant mode), keep next k.
    6) Pad to k if graph is too small.

  Parameters
  ----------
  G : nx.DiGraph
      Input graph (direction ignored).
  k : int, default=16
      Number of nontrivial eigenvectors.

  Returns
  -------
  pe : torch.Tensor
      (N, k) positional embeddings on CPU.
  node_order : dict
      Mapping {node_id: i} for row alignment in `pe`.

  Trade-offs
  ----------
  - O(N^3) time and O(N^2) memory due to dense eigendecomposition. This is
    efficient for small–medium graphs but may be infeasible for large N.
  """
  # 1) Convert to undirected and order nodes
  U = nx.Graph(G)
  node_id_list = sorted(U.nodes())
  N = len(node_id_list)
  node_order = {node_id: i for i, node_id in enumerate(node_id_list)}

  # 2) Build sparse Laplacian L
  rows, cols, data = [], [], []
  for u, v in U.edges():
      i, j = node_order[u], node_order[v]
      rows += [i, j]
      cols += [j, i]
      data += [1.0, 1.0]
  A = sp.coo_matrix((data, (rows, cols)), shape=(N, N))
  degs = np.array(A.sum(axis=1)).flatten()
  L = sp.diags(degs) - A  # scipy sparse

  # 3) Convert to dense torch tensor on GPU
  L_dense = torch.from_numpy(L.toarray().astype(np.float32)).to("cuda")

  # 4) Compute eigenpairs on GPU
  try:
    eigvals, eigvecs = torch.linalg.eigh(L_dense)  # symmetric eigendecomposition
  except Exception:
      pe_cpu, node_order_cpu = compute_laplacian_pe(G, k=k, device="cpu")
      return pe_cpu, node_order_cpu
      
  # 5) Discard trivial eigenvector and select top k
  k_eff = min(k, N - 1)
  pe = eigvecs[:, 1:k_eff+1].cpu()

  # 6) Pad if needed
  if pe.shape[1] < k:
      pad = torch.zeros((N, k - k_eff), dtype=pe.dtype)
      pe = torch.cat([pe, pad], dim=1)

  #del L, L_dense, eigvals, eigvecs
  #torch.cuda.empty_cache()
  return pe, node_order

def compute_laplacian_pe(G: nx.DiGraph, k: int = 16, device = "cuda"):
  """
  Compute the first k nontrivial Laplacian eigenvectors for a given graph.

  If `device == "cuda"`, dispatches to the dense GPU routine; otherwise uses a
  CPU routine with sparse eigensolver for large graphs when possible.

  Args:
      G (nx.DiGraph): Directed graph (AST) where nodes already exist.
      k (int): Number of nontrivial eigenvectors to return. (Total spectral dim = k.)
      device (str or torch.device): Selector for GPU ("cuda") vs CPU path.

  Returns:
      pe_tensor (torch.FloatTensor): Shape (N, k), where N = number of nodes in G.
      node_order (dict): Dict {node_id: i}. Row i in `pe_tensor` corresponds to node_id.

  Implementation details:
      - Builds undirected adjacency from G to ignore direction.
      - Constructs Laplacian L = D - A (combinatorial).
      - Uses np.linalg.eigh for small N (<= 2000) or scipy.sparse.linalg.eigsh
        for larger graphs; falls back to dense eigh if sparse fails.
      - Drops the first (trivial) eigenvector and keeps the next k.
      - Pads with zeros if the graph is too small to provide k nontrivial modes.
  """
  if device == "cuda" :
      return compute_laplacian_pe_gpu(G = G, k = k)
  # 1) Convert to undirected, since spectral PE ignores edge direction
  U = nx.Graph(G)

  # 2) Fix a consistent ordering of all nodes
  node_id_list = sorted(U.nodes())
  N = len(node_id_list)
  node_order = {node_id: i for i, node_id in enumerate(node_id_list)}

  # 3) Build adjacency matrix A in COO format (shape N×N)
  rows, cols, data = [], [], []
  for u, v in U.edges():
      i, j = node_order[u], node_order[v]
      # Add both (i,j) and (j,i) to make A symmetric
      rows += [i, j]
      cols += [j, i]
      data += [1.0, 1.0]
  A = sp.coo_matrix((data, (rows, cols)), shape=(N, N))

  # 4) Compute degree matrix D and Laplacian L = D - A
  degs = np.array(A.sum(axis=1)).flatten()
  D = sp.diags(degs)
  L = D - A

  # 5) Cap effective k to at most N-1
  max_nontrivial = N - 1
  k_eff = min(k, max_nontrivial)

  # 6) Solve for the smallest (k+1) eigenpairs: L u = λ u
  try:
      if N <= 2000:
          L_dense = L.toarray()
          eigvals, eigvecs = np.linalg.eigh(L_dense)
          eigvecs = eigvecs[:, : (k_eff + 1)]
      else:
          eigvals, eigvecs = spla.eigsh(L, k=k_eff+1, which="SM", tol=1e-4)
  except Exception:
      # Fallback to dense if sparse solver fails
      L_dense = L.toarray()
      eigvals, eigvecs = np.linalg.eigh(L_dense)
      eigvecs = eigvecs[:, : (k_eff + 1)]

  # 7) Discard the trivial eigenvector at index 0 (constant mode), keep next k
  pe = eigvecs[:, 1 : k_eff + 1]  # shape: (N, k)

  # 8) Padding if necessary
  if pe.shape[1] < k:
      pad_width = k - k_eff
      pad = np.zeros((N, pad_width), dtype=pe.dtype)
      pe = np.hstack([pe, pad])

  # 9) Convert to torch.FloatTensor and return
  pe_tensor = torch.from_numpy(pe)  # (N, k)
  return pe_tensor, node_order
