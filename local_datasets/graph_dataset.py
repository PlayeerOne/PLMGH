# unified_dataset_pyg.py
import torch
import os, gc
from typing import List, Dict, Optional

from torch_geometric.data import Dataset, Data
from utils.preprocessing.utils import load_pyg_graph


class GraphDatasetBase(Dataset):
    """
    Base dataset using three resolvers:
      - resolve_id_to_path():        {sample_id -> path_to_graph_pt}
      - resolve_id_to_str_label():   {sample_id -> label_string}
      - resolve_str_label_to_idx():  {label_string -> class_index}

    RAM cache:
      - Call `use_memory_cache()` to toggle ON/OFF (also usable from __init__ via in_memory=True).
      - When ON, get() serves a cloned copy from an in-RAM dict (pre-transform).
      - When OFF, get() loads from disk on demand.
    """

    def __init__(self, root: str, transform=None, *, in_memory: bool = False):
      super().__init__(root=root, transform=transform)
      #normpath this for me 
      self.root = os.path.normpath(os.path.abspath(root))
      # subclass-provided mappings
      self.id_to_path: Dict[str, str] = self.resolve_id_to_path()
      self.id_to_str_label: Dict[str, str] = self.resolve_id_to_str_label()
      self.str_label_to_idx: Dict[str, int] = self.resolve_str_label_to_idx()

      # stable IDs (intersection of available paths & labels)
      self.ids: List[str] = sorted(set(self.id_to_path) & set(self.id_to_str_label))

      # label map (dense 0..C-1)
      max_idx = max(self.str_label_to_idx.values()) if self.str_label_to_idx else -1
      self.idx_to_str: List[Optional[str]] = [None] * (max_idx + 1)
      for s, i in self.str_label_to_idx.items():
          if i >= len(self.idx_to_str):
              self.idx_to_str.extend([None] * (i - len(self.idx_to_str) + 1))
          self.idx_to_str[i] = s
      if any(v is None for v in self.idx_to_str):
          raise ValueError("Label indices must be dense from 0..C-1 with no gaps.")

      # RAM cache state
      self.in_memory: bool = False
      self._cache: Dict[str, Data] = {}

      if in_memory:
          self.load_in_memory()  # toggles ON

    # ---------- abstract resolvers (implement in subclass) ----------
    def resolve_id_to_path(self) -> Dict[str, str]:
      raise NotImplementedError

    def resolve_id_to_str_label(self) -> Dict[str, str]:
      raise NotImplementedError

    def resolve_str_label_to_idx(self) -> Dict[str, int]:
      raise NotImplementedError

    # ---------- label helpers ----------
    def str_to_index(self, label: str) -> int:
      return self.str_label_to_idx[label]

    def index_to_str(self, idx: int) -> str:
      return self.idx_to_str[idx]

    # ---------- core loading ----------
    def _make_raw_data(self, sid: str) -> Data:
      """Load one raw graph (CPU, pre-transform), attach label/id."""
      path = self.id_to_path[sid]
      label_str = self.id_to_str_label[sid]
      data: Data = load_pyg_graph(path)
      data.y = torch.tensor([self.str_to_index(label_str)], dtype=torch.long)
      data.graph_id = sid
      return data

    # ---------- RAM cache toggle ----------
    def load_in_memory(self):
      """
      Toggle RAM caching:
        - If OFF -> ON: load all samples into RAM cache.
      """
      if not self.in_memory:
          # Turn ON: fill cache
          for sid in self.ids:
              self._cache[sid] = self._make_raw_data(sid)
          self.in_memory = True

    def unload_memory(self):
      # Turn OFF: drop cache
      self._cache.clear()
      self.in_memory = False
      gc.collect()

    # ---------- PyG hooks ----------
    def len(self) -> int:
        return len(self.ids)
        
    def __len__(self):
      return self.len()

    def get(self, idx: int) -> Data:
      sid = self.ids[idx]
      if self.in_memory:
          # Serve a clone so transforms don't mutate the cached copy
          return self._cache[sid].clone()
      return self._make_raw_data(sid)