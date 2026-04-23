# Set path 
import os, sys
ROOT= os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARENT = os.path.dirname(ROOT)             
for p in (PARENT, ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)
        
# scripts/run_hpo_from_config.py
import argparse
import yaml

        
from scripts.preprocessing_config_reader import (
    read_preprocessing_config,
    run_preprocessing,
    ConfigError,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-config", "--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    try:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # If your YAML nests this under a key, support it transparently.
        cfg = cfg.get("preprocessing", cfg)

        pconf = read_preprocessing_config(cfg)
        run_preprocessing(pconf)
        print("Preprocessing completed.")
    except (OSError, yaml.YAMLError) as e:
        print(f"Failed to read config: {e}", file=sys.stderr)
        sys.exit(1)
    except ConfigError as e:
        print(f"Invalid config: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Error during preprocessing: {e}", file=sys.stderr)
        sys.exit(3)

if __name__ == "__main__":
    main()
