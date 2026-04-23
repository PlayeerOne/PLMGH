import os 
import math
import torch
import pandas as pd
import networkx as nx
from tqdm import tqdm
from pathlib import Path
from utils.preprocessing.utils import gather_dataset_files, get_semantic_extractor
from utils.preprocessing.ast_extraction import extract_ast, build_node_type_mapping_from_snippets
from utils.preprocessing.feature_extraction import extract_graph_features
from transformers import AutoTokenizer, AutoModel


def preprocess_java250(
  dataset_path, 
  out_dir,
  lang: str = "java",
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

  #Get the path of the dataset
  dataset_path = os.path.abspath(os.path.normpath(dataset_path))
  out_dir = os.path.abspath(os.path.normpath(out_dir))
  #Read the files
  snippets_dict = get_snippet_dicts_java250(dataset_path) #"<instance_id>": {"label": "<class_name>", "snippet": "<source_code>", "split": "train" | "valid" | "test"}
  #We figure out the path to each graph -> out_dir/label/instance_id.pt and add it in the subdicts
  for inst_id, rec in snippets_dict.items():
      label = rec["label"]
      inst_dir = os.path.join(out_dir, label)
      # Compute absolute .pt file path
      abs_path = os.path.abspath(os.path.join(inst_dir, f"{inst_id}.pt"))
      rec["out_path"] = abs_path

  snippet_ids = [snippet_id for snippet_id in snippets_dict]
  print(f"Found {len(snippet_ids)} Java files under '{dataset_path}'")

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

  for idx in tqdm( range(0, len(snippet_ids), batch_size), desc = "preprocessing the Java 250 Dataset" ):
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

  #Add train_ids_split.csv, valid_ids_split.csv, test_ids_split.csv in out_dir/split
  split_dir = os.path.abspath(os.path.join(out_dir, "split"))
  os.makedirs(split_dir, exist_ok=True)

  split_ids = {"train": [], "valid": [], "test": []}
  for inst_id, rec in snippets_dict.items():
      s = str(rec.get("split", "train")).lower()
      if s.startswith("val"):  # handle "val"/"valid"/"validation"
          s = "valid"
      if s not in split_ids:
          s = "train"
      split_ids[s].append(inst_id)

  for name, ids in split_ids.items():
      csv_path = os.path.join(split_dir, f"{name}_ids_split.csv")
      with open(csv_path, "w", encoding="utf-8") as f:
          for _id in ids:
              f.write(f"{_id}\n")

  print("Preprocessing Done !")

  

def get_snippet_dicts_java250(path: str):
    """
    Builds a nested dict for Java250 where each entry looks like:
        {
          "<instance_id>": {
              "label": "<class_name>",
              "snippet": "<source_code>",
              "split": "train" | "valid" | "test"
          },
          ...
        }
    """
    triplets = gather_java250_files(path)
    paths = [instance_path for _, _, instance_path in triplets]
    ids = [Path(p).stem for p in paths]
    labels = [Path(p).parent.name for p in paths]

    # get split mapping { "train": [...ids...], "valid": [...], "test": [...] }
    split_to_ids = get_idx_split_java250(
        dataset_path= path
    )

    # build reverse lookup: {id: split_name}
    id_to_split = {}
    for split_name, split_ids in split_to_ids.items():
        for sid in split_ids:
            id_to_split[sid] = split_name

    snippets_dict = {}

    for i, id_ in enumerate(ids):
        # read the source code
        with open(paths[i], "r", encoding="utf-8", errors="ignore") as f:
            code = f.read()

        # determine split, fallback to "unspecified" if not found
        split = id_to_split.get(id_, "unspecified")

        snippets_dict[id_] = {
            "label": labels[i],
            "snippet": code,
            "split": split,
        }

    return snippets_dict

def get_idx_split_java250(dataset_path: str):
    """
    Reads train, validation, and test indices from CSV split files in the dataset.

    Expects:
      data/Project_CodeNet/java250/
        └── split/
            ├── train.csv
            ├── valid.csv
            ├── test.csv
            ├──train_ids_split.csv
            ├──valid_ids_split.csv
            └──test_ids_split.csv

    Each CSV contains a single column of integer indices (zero-based), one per line.
    The ids_split files contain instance IDs instead

    Returns:
        { "train": [...], "valid": [...], "test": [...] }
    """
    split_folder = os.path.join(dataset_path, "split")
    if not os.path.isdir(split_folder):
        raise FileNotFoundError(f"Split folder not found at: {split_folder}")

    train_ids_csv = os.path.join(split_folder, "train_ids_split.csv")
    valid_ids_csv = os.path.join(split_folder, "valid_ids_split.csv")
    test_ids_csv = os.path.join(split_folder, "test_ids_split.csv")

    # If the split by IDs files exist, load them
    if os.path.isfile(train_ids_csv) and os.path.isfile(valid_ids_csv) and os.path.isfile(test_ids_csv):
        print("Loading splits by instance IDs from CSV files.")
        train_ids = pd.read_csv(train_ids_csv, header=None).iloc[:, 0].tolist()
        valid_ids = pd.read_csv(valid_ids_csv, header=None).iloc[:, 0].tolist()
        test_ids = pd.read_csv(test_ids_csv, header=None).iloc[:, 0].tolist()
    else: #Otherwise create them using the idx method (which tends to be udeterministic)
        print("split files by instance IDs not found. Generating splits using the indices instead.")
        
        #Only do this if you cant fine the ids .csv files (train_ids_split.csv, valid_ids_split.csv, test_ids_split.csv)
        train_idx = pd.read_csv(os.path.join(split_folder, "train.csv"), header=None).iloc[:, 0].tolist()
        valid_idx = pd.read_csv(os.path.join(split_folder, "valid.csv"), header=None).iloc[:, 0].tolist()
        test_idx = pd.read_csv(os.path.join(split_folder, "test.csv"), header=None).iloc[:, 0].tolist()
        
        triplets = gather_java250_files(dataset_path)
        train_ids = [triplets[idx][0] for idx in train_idx] 
        valid_ids = [triplets[idx][0] for idx in valid_idx]
        test_ids = [triplets[idx][0] for idx in test_idx]
        
        # Save the generated splits into CSV files for future use
        pd.DataFrame(train_ids).to_csv(train_ids_csv, header=None, index=False)
        pd.DataFrame(valid_ids).to_csv(valid_ids_csv, header=None, index=False)
        pd.DataFrame(test_ids).to_csv(test_ids_csv, header=None, index=False)
    #Otherwise load those ids splits here
    
    return {"train": train_ids, "valid": valid_ids, "test": test_ids}

def gather_java250_files(path):
    path = os.path.abspath(os.path.normpath(path))
    files = gather_dataset_files(path = path, suffix ="java")
    #Relative paths
    rel_paths = [ os.path.relpath(instance_path, path )for instance_path in files ]
    #Extract the labels for each instance
    labels = [os.path.splitext(rel_p)[0].split(os.sep)[0] for rel_p in rel_paths ]
    ids = [os.path.splitext(rel_p)[0].split(os.sep)[1] for rel_p in rel_paths ]
    
    return [(instance_id, instance_label, instance_path)  
            for instance_id, instance_label, instance_path in 
            zip(ids, labels, files)]


