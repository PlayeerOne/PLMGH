#Set path 
import os, sys
ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /content/BeyondTheTokens
PARENT = os.path.dirname(ROOT)                                        # /content
for p in (PARENT, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
        # scripts/run_hpo_from_config.py
import argparse
from utils.training.hp_tuning import run_hptuning
from scripts.config_parser import build_kwargs_from_file
from scripts.hpo_config_reader import get_graph_hpo_config
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    kwargs = build_kwargs_from_file(args.config, config_reader=get_graph_hpo_config)
    run_hptuning(**kwargs)

if __name__ == "__main__":
    main()
