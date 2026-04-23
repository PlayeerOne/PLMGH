# config_parser.py
import os
import io
import yaml
from typing import Any, Callable, Optional

class ConfigParserError(Exception):
    pass

def _expand_path(p: str) -> str:
    return os.path.normpath(os.path.abspath(os.path.expanduser(os.path.expandvars(p))))

def _expand_env_in_obj(obj: Any) -> Any:
    """
    Recursively expand ${VARS} and ~ in strings inside nested dicts/lists.
    Non-strings are returned untouched.
    """
    if isinstance(obj, dict):
        return {k: _expand_env_in_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_in_obj(v) for v in obj]
    if isinstance(obj, str):
        # expand env and user-home; do NOT resolve to abspath here for non-path strings
        return os.path.expandvars(os.path.expanduser(obj))
    return obj

def load_yaml_config(config_path: str) -> dict:
    """
    Load YAML safely, with good error messages, and expand env vars in values.
    """
    if not isinstance(config_path, str):
        raise ConfigParserError("config_path must be a string")

    path = _expand_path(config_path)
    if not os.path.exists(path):
        raise ConfigParserError(f"Config file not found: {path}")

    try:
        with io.open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigParserError(f"YAML parse error in {path}:\n{e}") from e

    if cfg is None:
        raise ConfigParserError(f"Empty YAML file: {path}")

    # Expand env vars inside the loaded structure
    cfg = _expand_env_in_obj(cfg)

    return cfg

def build_kwargs_from_file(
    config_path: str,
    *,
    config_reader: Callable[[dict], dict],
) -> dict:
    """
    High-level helper:
    1) load YAML
    2) call your config_reader builder
    3) return kwargs for experiment
    """
    cfg = load_yaml_config(config_path)
    try:
        kwargs = config_reader(cfg)
    except Exception as e:
        # Re-wrap to surface the config file path in the error
        raise ConfigParserError(f"Error while building kwargs from {config_path}: {e}") from e
    return kwargs
