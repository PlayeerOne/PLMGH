from typing import Any, Callable, Dict, Tuple
from dataclasses import dataclass, field

from BeyondTheTokens.utils.preprocessing.java250_preprocessing import preprocess_java250
from BeyondTheTokens.utils.preprocessing.devign_preprocessing import preprocess_devign


class ConfigError(Exception):
    pass


@dataclass(frozen=True)
class DatasetSpec:
    func: Callable[..., Any]
    required: Tuple[str, ...]
    optional: Dict[str, Any] = field(default_factory=dict)


SPECS: Dict[str, DatasetSpec] = {
    "Java250": DatasetSpec(
        func=preprocess_java250,
        required=(
            "dataset_path", "out_dir", "model_name", "max_length",
            "stride", "batch_size", "pooling",
            "extract_semantics", "extract_positionals", "positional_feat_dim",
        ),
        optional={
            # put dataset-specific defaults here if you have any
            # e.g. "semantics_device": "cuda", "positional_device": "cuda"
        },
    ),
    "Devign": DatasetSpec(
        func=preprocess_devign,
        required=(
            "out_dir", "model_name", "max_length",
            "stride", "batch_size", "pooling",
            "extract_semantics", "extract_positionals", "positional_feat_dim",
        ),
        optional={},
    ),
}


def read_preprocessing_config(config: Dict[str, Any]) -> Dict[str, Any]:
    ds = config.get("dataset")
    if ds not in SPECS:
        raise ConfigError(f"Unknown or missing dataset: {ds!r}")

    spec = SPECS[ds]
    allowed = set(spec.required) | set(spec.optional)

    missing = [k for k in spec.required if k not in config]
    extra = [k for k in config.keys() if k not in allowed | {"dataset"}]
    
    if len(missing) or len(extra):
        msgs = []
        if missing:
            msgs.append(f"missing={missing}")
        if extra:
            msgs.append(f"unknown={extra}")
        raise ConfigError(f"Invalid config for {ds}: " + "; ".join(msgs))
    # build kwargs: optional defaults overridden by provided values
    kwargs = {**spec.optional, **{k: config[k] for k in allowed if k in config}}
    
    return {
        "preprocessing_func": spec.func,
        **kwargs,
    }


def run_preprocessing(config: Dict[str, Any]):
    func = config.pop("preprocessing_func")
    return func(**config)
