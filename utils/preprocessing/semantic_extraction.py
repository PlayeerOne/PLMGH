# =============================================================================
# Semantic Feature Extraction for Code-as-Graphs
# -----------------------------------------------------------------------------
# This module encodes raw code snippets with a transformer (LLM/PLM), aligns
# AST nodes to token spans using byte offsets, pools token-level hidden states
# into per-node vectors, and writes them into NetworkX graphs.
#
# Pipeline sketch:
#   - tokenize_with_chunking: split long inputs into overlapping chunks
#   - extract_transformer_features: run the model once over the padded chunks
#   - stitch per-snippet token features (drop overlaps to avoid double-counting)
#   - map_nodes_to_token_spans_by_bytes: node byte spans → token index spans
#   - pool_semantic_features: pool token features [l:r) to one node vector
#   - assign_semantic_features: attach vectors to nodes under `attr_name`
#
# Key design choices & trade-offs:
#   - Offsets are computed with add_special_tokens=False to keep byte spans
#     aligned. Special tokens are only used later for padding in batches.
#   - Overlap handling: when stitching chunk features, the leading `stride`
#     tokens of later chunks are dropped (already covered by previous chunk).
#   - Mapping by byte offsets is robust to multi-byte UTF-8 characters via a
#     char→byte lookup table (char_to_byte_map).
#   - Pooling offers {"mean","sum","max","first","last"} to adapt to tasks.
#   - Memory: we batch chunks across all snippets at once. For very long inputs
#     or large batches, consider smaller `batch_size` upstream.
# =============================================================================

import bisect
import torch 
import networkx as nx
from collections import defaultdict
from typing import List, Union

from utils.preprocessing.utils import get_model_max_length


@torch.no_grad()
def extract_semantic_features(
    snippets: Union[str, List[str]],
    input_graphs: Union[nx.Graph, List[nx.Graph]],
    model,
    tokenizer,
    batch_size: int = 16,
    stride: int = 32,
    max_length: int = None,
    out_dtype: torch.dtype = torch.float16,
    autocast_dtype: torch.dtype = torch.float16,
    attr_name: str = "semantic_feats",
    pooling: str = "mean",
):
  """
  For each (snippet, graph) pair:
    1) encode snippet -> token-level hidden states (batched)
    2) map node->token spans
    3) pool and write node features to graph[attr_name]
  Returns: list[nx.Graph] (or single nx.Graph if single inputs were given)

  Parameters
  ----------
  snippets : str | List[str]
      Code text(s) to encode.
  input_graphs : nx.Graph | List[nx.Graph]
      Graph(s) whose nodes carry "start_byte" and "end_byte" attributes.
  model, tokenizer
      Hugging Face model/tokenizer pair used to produce hidden states.
  batch_size : int
      Number of snippets to process per batch in the transformer pass.
  stride : int
      Overlap (in tokens) between consecutive chunks for long inputs.
  max_length : int | None
      Maximum tokens per chunk. If None, inferred from model/tokenizer.
  out_dtype : torch.dtype
      Final dtype of stored node feature tensors.
  autocast_dtype : torch.dtype
      Mixed-precision dtype used inside the forward pass (e.g., bf16/fp16).
  attr_name : str
      Node attribute key to store the pooled features.
  pooling : {"mean","sum","max","first","last"}
      Aggregation across token span [l:r).

  Notes
  -----
  - Graphs are mutated in place (attributes written on nodes).
  - Returns the same object(s) for convenience.
  """
  # ---- normalize inputs
  single_input = isinstance(snippets, str)
  if single_input:
      snippets = [snippets]
  if isinstance(input_graphs, nx.Graph):
      input_graphs = [input_graphs]
  assert len(snippets) == len(input_graphs), "snippets and graphs must align 1:1"

  pooling = pooling.lower()
  assert pooling in {"mean", "sum", "max", "first", "last"}

  # ---- 1) encode snippets in batches (you already have this)
  # expected return: List[torch.Tensor] with length == len(snippets), each [T_i, H]
  feat_mats: List[torch.Tensor] = extract_hidden_features(
      snippets=snippets,
      model=model,
      tokenizer=tokenizer,
      batch_size=batch_size,
      stride=stride,
      max_length=max_length,
      out_dtype=out_dtype,
      autocast_dtype=autocast_dtype,
  )

  # ---- 2) attach to graphs
  out_graphs: List[nx.Graph] = []
  for snippet, G, token_feats in zip(snippets, input_graphs, feat_mats):
      # token_feats can be on CUDA; assign_semantic_features handles dtype+CPU writeback
      out_G = assign_semantic_features(
          input_graph=G,
          source_snippet=snippet,
          tokenizer=tokenizer,
          semantic_feats=token_feats,   # [T, H]
          attr_name=attr_name,
          pooling=pooling,
          out_dtype=out_dtype,
      )
      out_graphs.append(out_G)

  return out_graphs[0] if single_input else out_graphs

@torch.no_grad()
def assign_semantic_features(
    input_graph: nx.Graph,
    source_snippet : str,
    tokenizer,
    semantic_feats,                  # [T, H] (CPU or CUDA)
    attr_name = "semantic_feats",    # node attribute name to write (e.g., "x")
    pooling = "mean",                # "mean"|"sum"|"max"|"first"|"last"
    out_dtype = torch.float32,
):
  """
  Attach pooled token embeddings to graph nodes.

  Steps:
    - Map nodes to token spans using byte offsets.
    - Pool token features for each span according to `pooling`.
    - Store pooled vectors under `attr_name` on each node.

  Assumptions
  -----------
  - `input_graph` nodes expose integer "start_byte" and "end_byte".
  - `semantic_feats` corresponds to the full (stitched) token sequence
    returned by `extract_hidden_features` for `source_snippet`.
  """
  #Map graph nodes to tokens
  node_to_span = map_nodes_to_token_spans_by_bytes(
        source = source_snippet,
        G = input_graph,
        tokenizer = tokenizer
        )

  node_feat_mat, node_id_to_idx = pool_semantic_features(
  G = input_graph,
  node_to_span = node_to_span,                   
  semantic_feats = semantic_feats,                 
  attr_name = attr_name,    
  pooling = pooling,             
  out_dtype = out_dtype
  )

  # Write back to graph node attributes (store CPU tensors for from_networkx later)
  for node_id in node_to_span:
      idx = node_id_to_idx[node_id]
      input_graph.nodes[node_id][attr_name] = node_feat_mat[idx]

  return input_graph

def map_nodes_to_token_spans_by_bytes(
    source: str,
    G: nx.DiGraph,
    tokenizer,
    snap_if_empty: bool = True,     # try nearest token if no overlap
    prefer_left: bool = True        # pick left neighbor on ties
) -> dict:
  """
  Map each AST node (start_byte,end_byte) to token span [l, r) using byte offsets.
  Returns {node_id: (l, r)} or None if truly unmappable.

  Details
  -------
  - Uses tokenizer offsets at the *character* level, then converts them to
    byte-level spans via `char_to_byte_map(source)`. This ensures correctness
    with UTF-8 multibyte characters.
  - Nodes with no overlap to any token (e.g., degenerate ranges) can be
    snapped to the nearest token if `snap_if_empty=True`.
  - The token list is treated as half-open intervals; r is exclusive.
  """
  tok_spans = get_token_byte_spans(source, tokenizer)   # [(bs, be)], no truncation
  if not tok_spans:
      return {nid: None for nid in G.nodes()}

  starts = [s for (s, _) in tok_spans]   # non-decreasing
  ends   = [e for (_, e) in tok_spans]
  T = len(tok_spans)

  node_to_span = {}

  for node_id, data in G.nodes(data=True):
      n_start = int(data.get("start_byte", 0))
      n_end   = int(data.get("end_byte", 0))
      if n_end < n_start:
          n_start, n_end = n_end, n_start  # normalize

      # find first token whose end > n_start
      left = bisect.bisect_right(ends, n_start - 1)
      # skip forward until actual overlap (defensive)
      while left < T and not (starts[left] < n_end and ends[left] > n_start):
          left += 1

      mapped = None
      if left < T:
          right = left
          while right < T and (starts[right] < n_end and ends[right] > n_start):
              right += 1
          if right > left:
              mapped = (left, right)

      # optional: snap zero/empty spans to nearest token
      if mapped is None and snap_if_empty:
          j = bisect.bisect_left(starts, n_start)
          if j <= 0:
              snap = 0
          elif j >= T:
              snap = T - 1
          else:
              left_dist  = n_start - starts[j-1]
              right_dist = starts[j] - n_start
              snap = (j-1) if (left_dist <= right_dist) else j
              if not prefer_left:
                  snap = j if (right_dist <= left_dist) else (j-1)
          mapped = (snap, min(snap + 1, T))

      node_to_span[node_id] = mapped

  return node_to_span

def get_token_byte_spans(source: str, tokenizer):
  """
  Tokenize `source` without adding special tokens, retrieve character‐level offsets,
  then convert those to byte‐level spans with char_to_byte_map.
  This version avoids the “>512 tokens” warning by temporarily raising model_max_length.

  Returns
  -------
  List[Tuple[int,int]]
      A list of (byte_start, byte_end) per token, covering the entire tokenized text.

  Notes
  -----
  - `return_overflowing_tokens=True` produces multiple chunks if needed, but we
    only read offsets from the first encoding here (for full-text span mapping).
  - Offsets are character positions in the original string; we translate them
    to UTF-8 byte positions for robust alignment with node byte ranges.
  """
  # 1) Tokenize normally (no truncation; we want ALL offsets)
  encoding = tokenizer(
      source,
      return_offsets_mapping=True,
      add_special_tokens=False,
      truncation=False,
      return_overflowing_tokens=True,
  )
  # 2) Convert char offsets to byte offsets
  offsets = getattr(encoding.encodings[0], "offsets", None) or encoding["offset_mapping"]
  c2b = char_to_byte_map(source)

  token_byte_spans = []
  for (char_s, char_e) in offsets:
      byte_s = c2b[char_s]
      byte_e = c2b[char_e]
      token_byte_spans.append((byte_s, byte_e))
  return token_byte_spans

def char_to_byte_map(source: str):
  """
  O(n) map: i -> byte offset of source[:i] under UTF-8.
  Returns list of length len(source)+1, with char_to_byte[0] = 0.

  Rationale
  ---------
  - Python strings are Unicode; tokenizer offsets are in character indices.
  - AST node locations are tracked in *byte* indices. We bridge the two by
    precomputing cumulative byte lengths of the UTF-8 encoding.
  """
  # Build a simple char→byte lookup table
  m = [0] * (len(source) + 1)
  off = 0
  for i, ch in enumerate(source, start=1):
      off += len(ch.encode("utf-8"))
      m[i] = off
  return m


@torch.no_grad()
def pool_semantic_features(
    G: nx.Graph,
    node_to_span,                    # {node_id: (start, end)}  (end exclusive)
    semantic_feats,                  # [T, H] (CPU or CUDA)
    attr_name = "semantic_feats",    # node attribute name to write (e.g., "x")
    pooling = "mean",                # "mean"|"sum"|"max"|"first"|"last"
    out_dtype = torch.float32,
):
  """
  Pool token features across each node's token span.

  Parameters
  ----------
  G : nx.Graph
      Input graph (for node iteration order; not mutated here).
  node_to_span : dict
      {node_id: (l, r)} with r exclusive, in token indices.
  semantic_feats : torch.Tensor
      [T, H] token-level hidden states. Device can be CPU or CUDA.
  attr_name : str
      (Unused here; kept for API symmetry with assign_*.)
  pooling : str
      Reduction strategy over token axis.
  out_dtype : torch.dtype
      Output dtype for the stacked node feature matrix.

  Returns
  -------
  node_feat_mat : torch.Tensor
      [N_nodes, H] pooled features stacked in node order of `node_to_span`.
  node_id_to_idx : dict
      Maps node_id to row index in `node_feat_mat`.
  """
  pooling = pooling.lower()
  assert pooling in {"mean", "sum", "max", "first", "last"}

  # Helper to pool a slice safely
  def _pool_slice(feat_slice: torch.Tensor):
      if pooling == "mean":
          return feat_slice.mean(dim=0)
      elif pooling == "sum":
          return feat_slice.sum(dim=0)
      elif pooling == "max":
          return feat_slice.max(dim=0).values
      elif pooling == "first":
          return feat_slice[0]
      elif pooling == "last":
          return feat_slice[-1]

  out_rows = []
  node_id_to_idx = {}
  for node_id in node_to_span:
      # get span
      span = node_to_span[node_id]
      #print(span, node_id)
      s, e = span
      #Get feature vec and pool
      feat_slice = semantic_feats[s:e]
      vec = _pool_slice(feat_slice)
      #store
      node_id_to_idx[node_id] = len(out_rows)
      out_rows.append(vec)

  # Stack on-device, convert to dtype
  node_feat_mat = torch.stack(out_rows, dim=0).to(dtype=out_dtype)

  return node_feat_mat, node_id_to_idx

@torch.no_grad()
def extract_hidden_features(
    snippets,
    model,
    tokenizer,
    batch_size = 32,
    stride=32,
    max_length=None,
    out_dtype=torch.float16,
    autocast_dtype=torch.bfloat16,
):
  """
  Encode multiple snippets and return per-snippet token feature matrices.

  Steps:
    - Partition into mini-batches of raw snippets.
    - For each batch, call `get_batch_hidden_features`, which handles
      chunking, model forward, and stitching.
    - Concatenate results across batches to preserve input order.

  Returns
  -------
  List[torch.Tensor]
      One [T_i, H] matrix per input snippet (dtype `out_dtype`).
  """
  feature_matrices = []
  for i in range(0, len(snippets), batch_size):
    batch_snippets = snippets[i:i+batch_size]
    batch_feature_matrices = get_batch_hidden_features(
        batch_snippets=batch_snippets,
        model=model ,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        out_dtype=out_dtype,
        autocast_dtype=autocast_dtype,
    )

    feature_matrices.extend(batch_feature_matrices)
  return feature_matrices

@torch.no_grad()
def get_batch_hidden_features(
    batch_snippets,
    model,
    tokenizer,
    max_length=None,
    stride=32,
    out_dtype=torch.float16,
    autocast_dtype=torch.bfloat16,
):
  """
  Encode a mini-batch of raw code snippets into per-snippet token feature matrices.

  This helper performs chunked tokenization for long inputs, executes a single
  transformer forward pass over the padded batch of chunks, and then stitches
  chunk-level hidden states back into a contiguous sequence per snippet by
  dropping the overlapping `stride` tokens from subsequent chunks.

  Parameters
  ----------
  batch_snippets : str or List[str]
      One code string or a list of code strings for this mini-batch.
  model : transformers.PreTrainedModel
      Hugging Face model used to produce token hidden states. The function
      assumes the model is already on the desired device.
  tokenizer : transformers.PreTrainedTokenizer
      Tokenizer paired with `model`.
  max_length : int or None, default=None
      Maximum tokens per chunk. If None, resolved via `get_model_max_length(model, tokenizer)`.
  stride : int, default=32
      Overlap (in tokens) between consecutive chunks for long inputs. During
      stitching, the first `stride` tokens of each subsequent chunk are dropped.
  out_dtype : torch.dtype, default=torch.float16
      Dtype of the returned per-snippet matrices.
  autocast_dtype : torch.dtype, default=torch.bfloat16
      Mixed-precision dtype used inside the transformer forward pass (propagated
      to `extract_transformer_features`).

  Returns
  -------
  List[torch.Tensor]
      A list of length `len(instances)`, where each element is a tensor of
      shape `[T_i, H]` (token length × hidden size) on CPU with dtype
      `out_dtype`. If a snippet yields no chunks, an empty tensor of shape
      `(0, H)` is returned for that snippet (with `H` taken from the model
      config or inferred from outputs).

  Notes
  -----
  - Tokenization is performed by `tokenize_with_chunking`, which sets
    `add_special_tokens=False` to preserve clean offsets; padding is added
    only when forming the chunk batch for model input.
  - The model is run once over all chunks via `extract_transformer_features`,
    which pads to a common length and returns a CPU tensor `[B, L, H]`.
  - Stitching keeps the entire first chunk and removes the overlapping prefix
    (`stride` tokens) from later chunks to avoid double-counting.
  - Hidden size `H` is read from the returned tensor if available, otherwise
    falls back to `model.config.hidden_size` (default 768).
  """
  # normalize inputs
  if isinstance(batch_snippets, str):
      instances = [batch_snippets]
  else:
      instances = batch_snippets

  # resolve max length
  if max_length is None:
      max_length = get_model_max_length(model, tokenizer)

  # 1) tokenize into chunks
  chunks = tokenize_with_chunking(instances, tokenizer, max_length=max_length, stride=stride)

  # 2) run model once over all chunks
  hidden_states = extract_transformer_features(chunks, tokenizer, model, autocast_dtype=autocast_dtype)
  H = hidden_states.size(-1) if hidden_states.numel() else getattr(model.config, "hidden_size", 768)

  # 3) group & stitch per snippet (do it on device, D2H once)
  per_snippet = defaultdict(list)
  for i, ch in enumerate(chunks):
      real_len = len(ch["input_ids"])
      per_snippet[ch["snippet_id"]].append((ch["chunk_index"], real_len, hidden_states[i, :real_len, :]))

  feature_matrices = []
  for sid in range(len(instances)):
      parts = per_snippet.get(sid, [])
      if not parts:
          feature_matrices.append(torch.zeros((0, H), dtype=out_dtype))
          continue
      parts.sort(key=lambda x: x[0])
      stitched = []
      for j, (_, real_len, emb) in enumerate(parts):
          if j == 0:
              stitched.append(emb)
          else:
              drop = min(stride, real_len)
              stitched.append(emb[drop:])
      mat = torch.cat(stitched, dim=0).to(dtype=out_dtype)  # single copy to CPU
      feature_matrices.append(mat)

  return feature_matrices

@torch.no_grad()
def extract_transformer_features(chunks, tokenizer, model, autocast_dtype=torch.bfloat16):
  """
  Run the transformer once on a padded batch of chunked inputs and return
  token-level hidden states (on CPU).

  Parameters
  ----------
  chunks : List[dict]
      Output of `tokenize_with_chunking` (fields: input_ids, snippet_id, ...).
  tokenizer : PreTrainedTokenizer
      Provides pad token id (set to eos/unk if missing).
  model : PreTrainedModel
      HF model; if encoder-decoder, we use the encoder output.
  autocast_dtype : torch.dtype
      Mixed-precision dtype for forward pass.

  Returns
  -------
  torch.Tensor
      [B, L, H] hidden states for each chunk, padded to common length L,
      moved to CPU for downstream stitching.
  """
  device = next(model.parameters()).device if hasattr(model, "parameters") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
  if tokenizer.pad_token_id is None:
      tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
  pad_id = tokenizer.pad_token_id

  # Pad all chunks to a single batch
  ids_list = [torch.tensor(c["input_ids"], dtype=torch.long) for c in chunks]
  if len(ids_list) == 0:
      return torch.empty((0, 0, getattr(model.config, "hidden_size", 768)))
  L = max(t.size(0) for t in ids_list)
  B = len(ids_list)

  batch_ids  = torch.full((B, L), pad_id, dtype=torch.long)
  attn_mask  = torch.zeros((B, L), dtype=torch.bool)
  for i, ids in enumerate(ids_list):
      batch_ids[i, :ids.size(0)] = ids
      attn_mask[i, :ids.size(0)] = True

  batch_ids  = batch_ids.to(device, non_blocking=True)
  attn_mask  = attn_mask.to(device, non_blocking=True)

  model.eval()
  with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=autocast_dtype):
      if getattr(model.config, "is_encoder_decoder", False):
          hs = model.encoder(input_ids=batch_ids, attention_mask=attn_mask, return_dict=True).last_hidden_state
      else:
          hs = model(input_ids=batch_ids, attention_mask=attn_mask, return_dict=True).last_hidden_state
  return hs.cpu()


def tokenize_with_chunking(code_snippets, tokenizer, max_length = None, stride = 32):
  """
  Tokenize one or more code snippets using a sliding-window chunking strategy.

  Each snippet is split into overlapping chunks of token IDs if it exceeds the model's max length.
  Returns a list of chunk dictionaries containing:
    - "snippet_id": index of the snippet in the original input list
    - "chunk_index": index of this chunk for that snippet
    - "input_ids": list of token IDs for the chunk
    - "offset_mapping": list of (char_start, char_end) tuples for each token

  Args:
      code_snippets (str or List[str]): Single code string or list of code strings to tokenize.
      tokenizer: Huggingface tokenizer instance (e.g., CodeBERT tokenizer).
      max_length (int, optional): Maximum number of tokens per chunk. Defaults to tokenizer.model_max_length.
      stride (int): Number of tokens to overlap between consecutive chunks.

  Returns:
      List[dict]: List of chunk metadata dictionaries.

  Notes
  -----
  - We use add_special_tokens=False to keep raw offsets clean; special tokens
    are only introduced during batch padding for model input.
  - Overlap `stride` ensures continuity between chunks; later, we drop the
    overlapping prefix when stitching hidden states back together.
  """
  if max_length is None :
      max_length = tokenizer.model_max_length

  all_chunks = []
  if isinstance(code_snippets, str):
      instances = [code_snippets]
  else :
      instances = code_snippets

  for idx, code in enumerate(instances):
      encoded = tokenizer(
          code,
          return_offsets_mapping=True,
          add_special_tokens=False,  # no special tokens yet for clean offsets
          max_length=max_length,
          truncation=True,
          stride=stride,
          return_overflowing_tokens=True
      )

      for chunk_i, (input_ids, offsets) in enumerate(zip(encoded["input_ids"], encoded["offset_mapping"])):
          all_chunks.append({
              "snippet_id": idx,
              "chunk_index": chunk_i,
              "input_ids": input_ids,
              "offset_mapping": offsets
          })

  return all_chunks
