#Set path 
import os, sys
ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
PARENT = os.path.dirname(ROOT)                                        
for p in (PARENT, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
        
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
import torch
import argparse
from lightning.pytorch import seed_everything
from utils.training.training import train_model
from scripts.config_parser import build_kwargs_from_file
from scripts.training_config_reader import get_training_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    base_kwargs = build_kwargs_from_file(args.config, config_reader=get_training_config)
    
    #Get n_seeds and pop from kwargs
    n_seeds   = int(base_kwargs.pop("n_seeds", 1))
    base_seed = 42
    #Get model_name and pop from kwargs
    base_name = str(base_kwargs.pop("model_name"))
    results = []
    for i in range(n_seeds):
        seed_i = base_seed + i
        seed_everything(seed_i, workers=True)
        if n_seeds == 1 :
            model_name = base_name
        else :
            model_name = f"{base_name}_{i}"
            
        #Rebuild kwargs to avoid stale RNG/splits inside the datamodule.
        kwargs = build_kwargs_from_file(args.config, config_reader=get_training_config)
        kwargs.pop("n_seeds", None)
        kwargs["model_name"] = model_name
        
        out = train_model(**kwargs)
        results.append(out)
        #Free up memory
        del kwargs
        gc.collect()
    try:
        for r in results:
            if r is None: 
                continue
            name = r.get("run_name", "<run>")
            best = r.get("best_ckpt", "")
            print(f"[done] {name}  best_ckpt={best}")
            tm = r.get("test_metrics")
            if isinstance(tm, list) and tm:
                print("  test_metrics:", tm[0])
    except Exception:
        pass
        
    for r in results:
        if not r:
            continue
        name = r.get("run_name", "<run>")
        best = r.get("best_ckpt", "")
        wpth = r.get("weights_path", "")
        print(f"[done] {name}")
        print(f"  best_ckpt   : {best}")
        print(f"  weights(.pt): {wpth}")
        tm = r.get("test_metrics")
        if isinstance(tm, list) and tm:
            print("  test_metrics:", tm[0])
            
if __name__ == "__main__":
    main()
    