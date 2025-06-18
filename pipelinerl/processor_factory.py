"""Simple cache for AutoProcessor instances."""
from typing import Dict
from transformers import AutoProcessor

_processors: Dict[str, AutoProcessor] = {}

def get_processor(model_name: str) -> AutoProcessor:
    """Get or create an AutoProcessor for the given model."""
    if model_name not in _processors:
        _processors[model_name] = AutoProcessor.from_pretrained(model_name)
    return _processors[model_name]

def clear_cache() -> None:
    """Clear all cached processors."""
    _processors.clear()