import torch 
import os, gc, pickle
import numpy as np
from torch.utils.data import Dataset 


class MLPDataset(Dataset): 

  def __init__(self, root = None):
    super().__init__()
    self.root = os.path.normpath(os.path.abspath(root)) 
    
    with open(self.root, "rb") as f:
      data = pickle.load(f)
    self.data = data
    self.ids = [key for key, _ in self.data.items()]
    self.ids.sort()

    self.id_to_str_label = self.resolve_id_to_str_label()
    self.str_label_to_idx = self.resolve_str_label_to_idx()
    self.id_to_label_idx = self.resolve_id_to_label_idx()

  def get_features(self):
    x = [self.data[x_id] for x_id in self.ids]
    x = torch.stack(x, dim = 0)
    return x 
  
  def get_labels(self):
    y = [self.id_to_label_idx[x_id] for x_id in self.ids] #List of indices
    return y 

  # ---------- abstract resolvers (implement in subclass) ----------
  def resolve_id_to_str_label(self):
    raise NotImplementedError

  def resolve_str_label_to_idx(self):
    raise NotImplementedError
  
  def resolve_id_to_label_idx(self):
    raise NotImplementedError

  # ---------- helpers ----------
  def str_to_index(self, label: str):
    return self.str_label_to_idx[label]

  def index_to_str(self, idx: int):
      # Reverse lookup
      for key, value in self.str_label_to_idx.items():
          if value == idx:
              return key
      raise KeyError(f"Label idx {idx} not found.")


  def len(self) -> int:
      return len(self.ids)
      
  def __len__(self):
    return self.len()

  def get(self, idx: int):
      x_id = self.ids[idx]
      x = self.data[x_id]
      y = self.id_to_label_idx[x_id]
      return x, torch.tensor(y, dtype=torch.long)

  def __getitem__(self, index):
     return self.get(index)