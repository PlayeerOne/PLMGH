import os
import math
import torch 
import pandas as pd
import networkx as nx

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

from utils.preprocessing.ast_extraction import extract_ast, build_node_type_mapping_from_snippets
from utils.preprocessing.feature_extraction import extract_graph_features
from utils.preprocessing.utils import get_semantic_extractor

def preprocess_devign( 
  out_dir,
  hugginface_path = "DetectVul/devign",
  key_of_interest: str = "func",
  split_keys: list[str] = ["train", "validation", "test"],
  label_key: str = "target",
  id_key: str = "id",
  lang: str = "c",
  model_name = None,
  max_length = None,
  stride = 32,
  batch_size = 32,
  out_dtype = torch.float16,
  autocast_dtype=torch.float16,
  pooling="mean",
  extract_semantics=True,
  extract_positionals=True,
  positional_feat_dim=32,
  semantic_attr_name="semantic_feats",
  positional_attr_name="positional_feats",
  node_type_attr_name = "node_type_idx",
  semantics_device="cuda",
  positional_device="cuda",
  ):

  # 1. load the dataset from huggingface.
  dataset = load_dataset(hugginface_path)

  # 2. Gather all instances across specified splits 
  snippets_dict = {}  #"<instance_id>": {"label": "<class_name>", "snippet": "<source_code>", "split": "train" | "valid" | "test", "out_path"}
  split_ids = {key : [] for key in split_keys}
  out_dir = os.path.abspath(os.path.normpath(out_dir)) #We also prepare the paths were the preprocessed data goes
  for split in split_keys:
    for row in dataset[split]:
      #Get the snippet info
      snippet = row[key_of_interest]
      instance_id = row[id_key]
      label = row[label_key]
      out_path = os.path.abspath(os.path.join(out_dir, f"{instance_id}.pt"))
      snippets_dict[instance_id] = {
        "label" :str(label),
        "snippet" : snippet,
        "split": split,
        "out_path" : out_path
      }
      split_ids[split].append(instance_id)
  snippet_ids = [snippet_id for snippet_id in snippets_dict]
  print(f"Found {len(snippets_dict)} instances under {hugginface_path}'")
  
  #Get node type to index mapping
  node_type_to_idx = build_node_type_mapping_from_snippets(
    snippets_dict = snippets_dict,
    snippet_key = "snippet",
    lang =  lang,
    show_progress = True
  )

  #Handle PLM for semantic extraction
  semantic_dict = get_semantic_extractor(model_name = model_name, device = semantics_device, autocast_dtype = autocast_dtype)
  model = semantic_dict["model"]
  tokenizer = semantic_dict["tokenizer"]
  semantics_device = semantic_dict["device"]

  for idx in tqdm( range(0, len(snippet_ids), batch_size), desc = "preprocessing the Devign Dataset" ):
    batch_ids = snippet_ids[idx : idx + batch_size]
    batch_snippets = [snippets_dict[snippet_id]["snippet"] for snippet_id in batch_ids]

    #Process batch of graphs    
    graphs = extract_graph_features(
      batch_snippets,
      lang = lang,
      model = model,
      tokenizer = tokenizer,
      node_type_to_index  = node_type_to_idx,
      stride = stride,
      max_length = max_length,
      out_dtype = out_dtype,
      autocast_dtype = autocast_dtype,
      pooling = pooling,
      to_torch_geometric = True,
      extract_semantics = extract_semantics,
      extract_positionals = extract_positionals,
      positional_feat_dim = positional_feat_dim,
      semantic_attr_name = semantic_attr_name,
      positional_attr_name = positional_attr_name,
      node_type_attr_name = node_type_attr_name,
      semantics_device = semantics_device,
      positional_device = positional_device
      )
    #Now we write the torch_geometric graphs
    for idx, instance_id in enumerate(batch_ids) :
      instance_path = snippets_dict[instance_id]["out_path"]
      Path(os.path.dirname(instance_path)).mkdir(parents=True, exist_ok=True)
      graph = graphs[idx]
      torch.save(graph, instance_path)

  split_dir = os.path.abspath(os.path.join(out_dir, "split"))
  Path(split_dir).mkdir(parents=True, exist_ok=True)

  for name, ids in split_ids.items():
      rows = [{"id": sid, "label": snippets_dict[sid]["label"]} for sid in ids]
      df = pd.DataFrame(rows, columns=["id", "label"])
      df.to_csv(os.path.join(split_dir, f"{name}_ids_split.csv"), index=False)
  




