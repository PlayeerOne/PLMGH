import os 

import torch
import pandas as pd 
from datasets import load_dataset

from pathlib import Path

from utils.preprocessing.utils import gather_dataset_files, get_tokenizer

from local_datasets.graph_dataset import GraphDatasetBase
from local_datasets.mlp_dataset import MLPDataset
from local_datasets.text_dataset_base import TextDatasetBase



class DevignGraphDataset(GraphDatasetBase):
  def __init__(self, root : str, csv_path : str, **kwargs):
    #Root that contains the dataset
    self.root = os.path.abspath(os.path.normpath(root))
    #Path for csv file
    self.csv_path = os.path.abspath(os.path.normpath(csv_path))
    self.csv = pd.read_csv(self.csv_path)
    #init the parent attributes
    super().__init__(root=root, **kwargs)

  
  def resolve_id_to_path(self):
    """Return {sample_id: absolute_path_to_graph_pt}."""
    ids = self.csv.id.tolist()
    id_to_path = { instance_id : os.path.join(self.root, f"{instance_id}.pt")  for instance_id in ids}
    return id_to_path

  def resolve_id_to_str_label(self):
    """Return {sample_id: label_string}."""
    id_to_str_label = {}
    for x in self.csv.iloc :
      instance_id = int(x.id)
      instance_label_str = str(x.label)
      id_to_str_label[instance_id] = instance_label_str 
    #id_to_str_label = {Path(instance_path).stem :  Path(instance_path).parent.name for instance_path in self.paths}
    return id_to_str_label

  def resolve_str_label_to_idx(self):
    """Return {label_string: class_index}."""
    #Binary classification
    return {
      "True" : 1,
      "False" : 0
    }



class DevignMLPDataset(MLPDataset):
  def __init__(self, features_root : str, dataset_root = "DetectVul/devign", split_key = "train", label_key = "target", id_key = "id"):
    self.features_root = os.path.abspath(os.path.normpath(features_root))
    self.dataset_root = dataset_root

    self.split_key = split_key
    self.label_key = label_key
    self.id_key = id_key

    super().__init__(root=self.features_root)

  def resolve_id_to_str_label(self):
    dataset = load_dataset(self.dataset_root)
    output = {}
    
    split_dataset = dataset[self.split_key]
    split_id_to_str_label = { int(row[self.id_key]) : str(row[self.label_key]) for row in split_dataset }
    output = {**output, **split_id_to_str_label}
    return output

  def resolve_str_label_to_idx(self):
    return {
      "True" : 1,
      "False" : 0
    }
  def resolve_id_to_label_idx(self):
    id_to_label_idx = {id : self.str_label_to_idx[label] for id, label in self.id_to_str_label.items()}
    return id_to_label_idx



class DevignTextDataset(TextDatasetBase):
  def __init__(self, model_name,hugginface_path = "DetectVul/devign", split_key = "train", label_key = "target", key_of_interest = "func",
               id_key = "id", features_dtype = torch.float16, max_length = None ):
    

    self.hugginface_path = hugginface_path
    self.split_key = split_key
    self.label_key = label_key
    self.id_key = id_key
    self.key_of_interest = key_of_interest
    self.max_length = max_length

    dataset = load_dataset(hugginface_path)
    self.split_dataset = dataset[self.split_key]
    
    super().__init__(model_name = model_name, max_length = max_length)
    
  # ---------------- abstract resolvers ----------------

  def resolve_id_to_text(self):
    return { int(row[self.id_key]) : row[self.key_of_interest] for row in self.split_dataset }

  def resolve_id_to_str_label(self) :
    """Return {sample_id: label_string}."""
    return { int(row[self.id_key]) : str(row[self.label_key]) for row in self.split_dataset }

  def resolve_str_label_to_idx(self) :
    """Return {label_string: class_index}."""
    return {
      "True" : 1,
      "False" : 0
    }


  # Not Needed because the dataset comes from HF
  def resolve_id_to_path(self) :
    return {}

