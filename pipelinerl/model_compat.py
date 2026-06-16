import json
from functools import lru_cache
from pathlib import Path


_QWEN35_NAME_HINTS = ("Qwen3.5", "qwen3.5", "qwen3_5")


@lru_cache(maxsize=64)
def _read_local_model_signature(model_path: str) -> tuple[str | None, tuple[str, ...], str | None]:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        return None, (), None

    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None, (), None

    architectures = tuple(str(arch) for arch in (config.get("architectures") or []))
    text_config = config.get("text_config")
    text_model_type = text_config.get("model_type") if isinstance(text_config, dict) else None
    return config.get("model_type"), architectures, text_model_type


def is_qwen35_multimodal_model(model_path: str | Path | None) -> bool:
    if model_path is None:
        return False

    model_path_str = str(model_path)
    if any(hint in model_path_str for hint in _QWEN35_NAME_HINTS):
        return True

    model_type, architectures, text_model_type = _read_local_model_signature(model_path_str)
    if model_type == "qwen3_5":
        return True
    if text_model_type == "qwen3_5_text":
        return True
    return "Qwen3_5ForConditionalGeneration" in architectures
