from transformers import AutoTokenizer, AutoModel
import torch 
import os
        

def get_model_max_length(model, tokenizer):
  """
  Infer a reasonable `max_length` for chunking from the model config.

  Checks common attributes in order:
    - config.max_position_embeddings
    - config.n_positions (GPT-2 style)
    - config.seq_length (misc models)

  Returns
  -------
  int
      Usable max length after subtracting special tokens.

  Warning
  -------
  If none of the known attributes exist, `max_length` will be left undefined,
  which would raise an error here. Extend this function if you use an
  unusual model type.
  """
  if hasattr(model.config, 'max_position_embeddings') and model.config.max_position_embeddings < 1e12:
        max_length = model.config.max_position_embeddings
  elif hasattr(model.config, 'n_positions') and model.config.n_positions < 1e12: # For GPT2-like models
        max_length = model.config.n_positions
  elif hasattr(model.config, 'seq_length') and model.config.seq_length < 1e12: # For some other models
        max_length = model.config.seq_length

  max_length = max_length - tokenizer.num_special_tokens_to_add()
  return max_length


def crawl_directory(path: str, suffix: str = None) -> list[str]:
    """
    Recursively crawl a directory and return a list of absolute file paths,
    optionally filtered by file extension.

    Args:
        path (str): Root directory to crawl.
        suffix (str or None): File extension (without leading dot) to filter by.
            Example: "java" matches files ending in ".java". If None, returns all files.

    Returns:
        list[str]: List of absolute file paths that match the suffix.
    """
    abs_root = os.path.abspath(os.path.normpath(path))
    results = []

    for root, _, files in os.walk(abs_root):
        for fname in files:
            if suffix is None or fname.lower().endswith(f".{suffix.lower()}"):
                results.append(os.path.join(root, fname))

    return results 


def gather_dataset_files(path, suffix = None) : 
    path = os.path.abspath(os.path.normpath(path))
    data_files = [file for file in crawl_directory(path, suffix) ]
    return data_files 

def load_pyg_graph(path): 
    path = os.path.abspath(os.path.normpath(path))
    # 1) Make sure PyG’s Data class is in scope:
    from torch_geometric.data import Data, EdgeAttr 
    data = torch.load(path, map_location="cpu", weights_only=False)
    return data

def get_semantic_extractor(model_name = None, device = None, autocast_dtype = torch.float16):
    #Instanciate model from huggingFace if asked for a model.
    if model_name is not None:
        # 1. Load tokenizer & model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        try :
            model = AutoModel.from_pretrained(model_name, dtype = autocast_dtype)
        except Exception as e :
            model = AutoModel.from_pretrained(model_name)
        # 2. Move model to device
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
    else: 
        model = None
        tokenizer = None
        device = None
        
    return { "model" : model, 
            "tokenizer" : tokenizer,
            "device" : device
           }



def get_tokenizer(model_name = None, max_length = None):
    tok_model = get_semantic_extractor(model_name = model_name, device = "cpu")

    model, tok = tok_model["model"], tok_model["tokenizer"]
    fetched_max_length = get_model_max_length(model = model, tokenizer = tok)
    
    if max_length is None :
      max_length = fetched_max_length
    else : 
      max_length = min(max_length, fetched_max_length)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, truncation=True, max_length=max_length )
    return { "tokenizer" : tokenizer, 
            "max_length" : max_length,
           }
    
    
