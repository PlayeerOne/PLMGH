import os 
import pandas as pd 
from pathlib import Path

import torch
from utils.preprocessing.utils import gather_dataset_files

from local_datasets.graph_dataset import GraphDatasetBase
from local_datasets.mlp_dataset import MLPDataset
from local_datasets.text_dataset_base import TextDatasetBase

class Java250GraphDataset(GraphDatasetBase):
  def __init__(self, root : str, csv_path : str, **kwargs):
    #Root that contains the dataset
    self.root = os.path.abspath(os.path.normpath(root))
    #Path for csv file
    self.csv_path = os.path.abspath(os.path.normpath(csv_path))
    self.csv = pd.read_csv(self.csv_path, header=None)
    #list of paths
    root_paths = gather_dataset_files(self.root, suffix = "pt")
    id_to_path_root = {Path(instance_path).stem :  instance_path for instance_path in root_paths}
    self.paths = [ id_to_path_root[instance_id] for instance_id in self.csv[0].tolist() ]
    #init the parent attributes
    super().__init__(root=root, **kwargs)

  
  def resolve_id_to_path(self):
    """Return {sample_id: absolute_path_to_graph_pt}."""
    id_to_path = {Path(instance_path).stem :  instance_path for instance_path in self.paths}
    return id_to_path

  def resolve_id_to_str_label(self):
    """Return {sample_id: label_string}."""
    id_to_str_label = {Path(instance_path).stem :  Path(instance_path).parent.name for instance_path in self.paths}
    return id_to_str_label

  def resolve_str_label_to_idx(self):
    """Return {label_string: class_index}."""
    string_labels = set()
    
    for instance_path in self.paths:
      label = Path(instance_path).parent.name
      string_labels.add(label)

    string_labels = list(string_labels)
    #Sort for consistency
    string_labels.sort()
    str_label_to_idx = {str_label : idx for idx, str_label in enumerate(string_labels)}  
    return str_label_to_idx

class Java250MLPDataset(MLPDataset):
  def __init__(self, features_root : str, dataset_root : str):
    self.features_root = os.path.abspath(os.path.normpath(features_root))
    self.dataset_root = os.path.abspath(os.path.normpath(dataset_root))

    self.files = self.find_files_with_extension_os_only()

    super().__init__(root=self.features_root)

  def find_files_with_extension_os_only(self):
    found_files = []
    # from the top directory.
    for root, dirs, files in os.walk(self.dataset_root):
        for file in files:
            # Check if the file's extension matches the target extension
            if file.endswith(".java"):
                # Construct the full path by joining the directory root and the filename
                full_path = os.path.join(root, file)
                # Append the absolute path to the list
                found_files.append(os.path.abspath(full_path))
    return found_files


  def resolve_id_to_str_label(self):
    id_to_str_label = {
      os.path.basename(path).split(".")[0] : os.path.dirname(path).split(os.sep)[-1] for path in self.files
      }
    return id_to_str_label

  def resolve_str_label_to_idx(self):
    labels = [os.path.dirname(path).split(os.sep)[-1] for path in self.files]
    labels = set(labels)
    labels = list(labels)
    labels.sort()
    str_label_to_idx = {str_label : idx for idx, str_label in enumerate(labels)}

    return str_label_to_idx
  
  def resolve_id_to_label_idx(self):
    id_to_label_idx = {id : self.str_label_to_idx[label] for id, label in self.id_to_str_label.items()}
    return id_to_label_idx


class Java250TextDataset(TextDatasetBase):
  def __init__(self, root : str, csv_path : str, tokenizer, model = None, max_length = None, cache_hidden = True, features_pickle = None,
               features_dtype = torch.float16 , build_if_missing = True, build_batch_size = 2, build_progress= True):
    
    self.root = os.path.abspath(os.path.normpath(root))
    self.csv_path = os.path.abspath(os.path.normpath(csv_path))
    self.csv = pd.read_csv(self.csv_path, header=None)
    root_paths = gather_dataset_files(root, suffix = "java")
    id_to_path_root = {Path(instance_path).stem :  instance_path for instance_path in root_paths}
    self.paths = [ id_to_path_root[instance_id] for instance_id in self.csv[0].tolist() ]

    super().__init__(root = self.root, tokenizer = tokenizer, model = model,
                     max_length = max_length, cache_hidden = cache_hidden, 
                     features_pickle = features_pickle, features_dtype = features_dtype, 
                     build_if_missing = build_if_missing, build_batch_size = build_batch_size,
                     build_progress = build_progress)


  # ---------------- abstract resolvers ----------------

  def resolve_id_to_text(self):
    id_to_path = {Path(instance_path).stem :  Path(instance_path) for instance_path in self.paths}
    return { sid : Path(p).read_text(encoding="utf-8") for sid, p in id_to_path.items()}

  def resolve_id_to_str_label(self) :
    """Return {sample_id: label_string}."""
    id_to_str_label = {Path(instance_path).stem :  Path(instance_path).parent.name for instance_path in self.paths}
    return id_to_str_label

  def resolve_str_label_to_idx(self) :
    """Return {label_string: class_index}."""
    string_labels = set()
    
    for instance_path in self.paths:
      label = Path(instance_path).parent.name
      string_labels.add(label)

    string_labels = list(string_labels)
    #Sort for consistency
    string_labels.sort()
    str_label_to_idx = {str_label : idx for idx, str_label in enumerate(string_labels)}  
    return str_label_to_idx


  # Optional; only needed if you want "path" in items
  def resolve_id_to_path(self) :
    id_to_path = {Path(instance_path).stem :  instance_path for instance_path in self.paths}
    return id_to_path

