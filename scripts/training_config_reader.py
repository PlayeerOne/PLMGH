import os
from BeyondTheTokens.local_datasets.java250 import Java250GraphDataset
from BeyondTheTokens.local_datasets.devign import DevignGraphDataset
from local_datasets.data_module import GraphDataModuleCls

def _get(d, key):
    if key not in d:
        raise KeyError(f"Missing required key: {key}")
    return d[key]

def get_data_module_config(config):
    data_config = _get(config, "data_module").copy()
    return data_config

def get_data_module(config):
    data_config = get_data_module_config(config)

    # === required keys check ===
    required = [
      "dataset", "root", "train_csv", "valid_csv", "test_csv",
      "batch_size", "num_workers", "pin_memory", "persistent_workers", "in_memory"
    ]
    missing = [k for k in required if k not in data_config]
    if missing:
        raise KeyError(f"[data_module] missing keys: {missing}")
    
    dataset_name = data_config["dataset"]
    root = os.path.normpath(os.path.abspath(data_config["root"]))
    
    train_csv = data_config["train_csv"]
    valid_csv = data_config["valid_csv"]
    test_csv  = data_config["test_csv"]
    in_memory = data_config["in_memory"]
    
    # === choose dataset ===
    if dataset_name == "Devign":
      train_ds = DevignGraphDataset(root=root, csv_path=train_csv, in_memory=in_memory)
      val_ds   = DevignGraphDataset(root=root, csv_path=valid_csv, in_memory=in_memory)
      test_ds  = DevignGraphDataset(root=root, csv_path=test_csv,  in_memory=in_memory)
    elif dataset_name == "Java250":
      train_ds = Java250GraphDataset(root=root, csv_path=train_csv, in_memory=in_memory)
      val_ds   = Java250GraphDataset(root=root, csv_path=valid_csv, in_memory=in_memory)
      test_ds  = Java250GraphDataset(root=root, csv_path=test_csv,  in_memory=in_memory)
    else:
      raise ValueError(f"Unknown dataset: {dataset_name}")
    
    data_kwargs = dict(
      train_ds=train_ds,
      val_ds=val_ds,
      test_ds=test_ds,
      batch_size=data_config["batch_size"],
      num_workers=data_config["num_workers"],
      pin_memory=data_config["pin_memory"],
      persistent_workers=data_config["persistent_workers"],
    )

    data_module = GraphDataModuleCls(**data_kwargs)
    return data_module

def get_wandb_config(config):
    wb = _get(config, "wandb").copy()
    return wb

def get_trainer_config(config):
    tr = _get(config, "trainer").copy()
    return tr

def get_early_stop_config(config):
    es = _get(config, "early")
    return {
      "early_monitor": es.get("early_monitor"),
      "early_mode": es.get("early_mode"),
      "early_patience": int(es.get("early_patience")),
    }

def get_checkpoint_config(config):
    ck = _get(config, "ckpt")
    return {
      "ckpt_monitor": ck.get("ckpt_monitor"),
      "ckpt_mode": ck.get("ckpt_mode"),
      "ckpt_save_top_k": int(ck.get("ckpt_save_top_k")),
    }

def get_model_name(config):
    return {"model_name" : _get(config, "model_name")} 

def get_n_classes(config):
    out = _get(config, "model")
    return {"num_classes" : out["num_classes"],
           "model_type" : out["model_type"]} 

def get_model_params(config):
    params = {"model_params" : _get(config, "model_params")}
    params["model_params"]["dropout"] = float(params["model_params"]["dropout"])
    params["model_params"]["weight_decay"] = float(params["model_params"]["weight_decay"])
    params["model_params"]["lr"] = float(params["model_params"]["lr"])
    params["model_params"]["pct_start"] = float(params["model_params"]["pct_start"])
    return params

def get_n_seeds(config):
    return {"n_seeds" : _get(config, "n_seeds")}

def get_training_config(config):
    training_kwargs = {"data_module" : get_data_module(config)}
    
    training_kwargs.update(get_wandb_config(config)) #WandB
    training_kwargs.update(get_trainer_config(config)) #Trainer
    training_kwargs.update(get_model_params(config)) #Model params
    training_kwargs.update(get_model_name(config)) #Model name
    training_kwargs.update(get_n_classes(config)) #Model
    training_kwargs.update(get_n_seeds(config)) #Number of seeds
    
    # callbacks
    training_kwargs.update(get_early_stop_config(config))
    training_kwargs.update(get_checkpoint_config(config))
    
    return training_kwargs