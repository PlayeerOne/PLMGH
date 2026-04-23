import torch
import torch.nn as nn
from torch_geometric.data import Data

class FeatureFusionLayer(nn.Module):
    """
    Fuse (optional) node-type, positional, and semantic features into a single node embedding.
    fusion_method: "concat" | "sum" | "gated"
    """
    def __init__(
        self,
        node_type_key="node_type_idx",
        use_node_type=True,
        pos_features_key="positional_feats",
        use_pos_features=True,
        semantic_features_key="semantic_feats",
        use_semantic_features=True,
        semantic_features_dim=768,
        pos_features_dim=32,
        node_type_dim=256,
        n_node_types=250,
        h_dim=512,
        out_dim=256,
        dropout=0.25,
        fusion_method="concat",
        padding_idx=None,
    ):
      super().__init__()

      # flags/keys
      self.node_type_key = node_type_key
      self.use_node_type = use_node_type

      self.pos_features_key = pos_features_key
      self.use_pos_features = use_pos_features

      self.semantic_features_key = semantic_features_key
      self.use_semantic_features = use_semantic_features

      self.node_type_dim = node_type_dim
      proj_dim = node_type_dim  # unify projected dims

      self.modalities = []

      # node-type embedding
      if self.use_node_type:
          self.node_type_emb = nn.Embedding(n_node_types, node_type_dim, padding_idx=padding_idx)
          nn.init.normal_(self.node_type_emb.weight, mean=0.0, std=0.02)
          if padding_idx is not None:
              with torch.no_grad():
                  self.node_type_emb.weight[padding_idx].zero_()
          self.modalities.append("node_type")

      # semantic projection
      if self.use_semantic_features:
          self.semantic_proj = nn.Sequential(
              nn.LayerNorm(semantic_features_dim),
              nn.Linear(semantic_features_dim, proj_dim),
              nn.GELU(),
          )
          nn.init.xavier_uniform_(self.semantic_proj[1].weight)
          self.modalities.append("semantic")

      # positional projection
      if self.use_pos_features:
          self.pos_proj = nn.Sequential(
              nn.LayerNorm(pos_features_dim),
              nn.Linear(pos_features_dim, proj_dim),
              nn.GELU(),
          )
          nn.init.xavier_uniform_(self.pos_proj[1].weight)
          self.modalities.append("positional")

      if fusion_method not in {"concat", "sum", "gated"}:
          raise ValueError("fusion_method must be one of {'concat','sum','gated'}")
      self.fusion_method = fusion_method

      m = len(self.modalities)
      mlp_in = m * proj_dim if fusion_method == "concat" else proj_dim

      self.mlp_out = nn.Sequential(
          nn.Linear(mlp_in, h_dim),
          nn.GELU(),
          nn.Dropout(dropout),
          nn.Linear(h_dim, out_dim),
          nn.LayerNorm(out_dim),
          nn.Dropout(dropout),
      )
      for mod in self.mlp_out:
          if isinstance(mod, nn.Linear):
              nn.init.xavier_uniform_(mod.weight)
              if mod.bias is not None:
                  nn.init.zeros_(mod.bias)

      if fusion_method == "gated":
          # one scalar gate per modality, conditioned on its own projected feature
          self.gates = nn.ModuleDict({name: nn.Linear(proj_dim, 1) for name in self.modalities})
          for g in self.gates.values():
              nn.init.xavier_uniform_(g.weight)
              if g.bias is not None:
                  nn.init.zeros_(g.bias)

    @staticmethod
    def _fetch_attr(data: Data, key: str):
        if hasattr(data, key):
            return getattr(data, key)
        try:  # PyG sometimes supports dict-like access
            return data[key]
        except Exception as e:
            raise KeyError(f"Graph is missing expected attribute '{key}'") from e

    def forward(self, g: Data, output_key: str = "x"):
        pieces = []
        if self.use_node_type:
            node_types = self._fetch_attr(g, self.node_type_key).long()
            nt = self.node_type_emb(node_types)
            pieces.append(("node_type", nt))

        if self.use_semantic_features:
            sem = self._fetch_attr(g, self.semantic_features_key)
            sem = self.semantic_proj(sem)
            pieces.append(("semantic", sem))

        if self.use_pos_features:
            pos = self._fetch_attr(g, self.pos_features_key)
            pos = self.pos_proj(pos)
            pieces.append(("positional", pos))

        if not pieces:
            raise RuntimeError("No features enabled for fusion.")

        if self.fusion_method == "concat":
            fused = torch.cat([t for _, t in pieces], dim=-1)
        elif self.fusion_method == "sum":
            fused = torch.stack([t for _, t in pieces], dim=0).sum(dim=0)
        else:  # gated
            gated = []
            for name, t in pieces:
                alpha = torch.sigmoid(self.gates[name](t))  # [N,1]
                gated.append(alpha * t)
            fused = torch.stack(gated, dim=0).sum(dim=0)

        out = self.mlp_out(fused)
        setattr(g, output_key, out)
        return g
