import logging
from concurrent.futures import Future
from pathlib import Path
from typing import Iterable, Sequence

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    from huggingface_hub.utils import HfHubHTTPError
except ImportError as exc:  # pragma: no cover - dependency provided by transformers
    HfApi = None  # type: ignore
    create_repo = None  # type: ignore
    upload_folder = None  # type: ignore
    HfHubHTTPError = Exception  # type: ignore
    _import_error = exc
else:
    _import_error = None

log = logging.getLogger(__name__)


def _get_config_value(cfg, key: str, default=None):
    """Safely read a key from DictConfig or mapping-like objects."""
    if cfg is None:
        return default
    try:
        return cfg.get(key, default)  # type: ignore[attr-defined]
    except AttributeError:
        return getattr(cfg, key, default)


def format_revision(prefix: str | None, step: int) -> str:
    base = (prefix or "checkpoint").strip()
    if not base:
        base = "checkpoint"
    return f"{base}-step-{step:06d}"


def push_checkpoint_to_hub(
    cfg,
    checkpoint_dir: Path,
    step: int,
    *,
    extra_ignore: Sequence[str] | None = None,
) -> Future | None:
    """Upload a checkpoint directory to the Hugging Face Hub.

    Returns a Future when uploads run asynchronously, otherwise None.
    """
    if not bool(_get_config_value(cfg, "push_to_hub", False)):
        return None

    if _import_error:
        log.error("huggingface_hub is required for push_to_hub but could not be imported: %s", _import_error)
        return None

    hub_model_id = _get_config_value(cfg, "hub_model_id")
    if not hub_model_id:
        log.warning("push_to_hub enabled but hub_model_id is not set; skipping upload")
        return None

    if not checkpoint_dir.exists():
        log.warning("Checkpoint directory %s does not exist; skipping upload", checkpoint_dir)
        return None

    private = bool(_get_config_value(cfg, "hub_private", True))
    branch_prefix = _get_config_value(cfg, "hub_model_revision") or "checkpoint"
    branch_parent = _get_config_value(cfg, "hub_base_revision", "main")

    ignore_patterns: list[str] = []
    if extra_ignore:
        ignore_patterns.extend(extra_ignore)
    config_ignores: Iterable[str] | None = _get_config_value(cfg, "hub_ignore_patterns")
    if config_ignores:
        ignore_patterns.extend(config_ignores)

    # Deduplicate while preserving order
    seen: set[str] = set()
    ignore_patterns = [pattern for pattern in ignore_patterns if not (pattern in seen or seen.add(pattern))]

    api = HfApi()
    create_repo(
        repo_id=hub_model_id,
        private=private,
        exist_ok=True,
        repo_type="model",
    )

    revision = format_revision(branch_prefix, step)
    create_branch_kwargs = dict(
        repo_id=hub_model_id,
        branch=revision,
        repo_type="model",
        exist_ok=True,
    )
    if branch_parent:
        create_branch_kwargs["revision"] = branch_parent

    try:
        api.create_branch(**create_branch_kwargs)
    except HfHubHTTPError as err:
        # `exist_ok=True` still raises if revision/branch mismatch; surface once.
        if err.response is not None and err.response.status_code not in (409, 422):
            raise
        log.debug("Hub branch %s already exists on %s", revision, hub_model_id)

    upload_result = upload_folder(
        repo_id=hub_model_id,
        folder_path=str(checkpoint_dir),
        repo_type="model",
        revision=revision,
        commit_message=f"Add checkpoint {revision}",
        ignore_patterns=ignore_patterns or None,
        run_as_future=True,
    )

    future: Future | None = None
    if isinstance(upload_result, Future):
        future = upload_result
    elif hasattr(upload_result, "future"):
        future = getattr(upload_result, "future")
    else:
        log.warning(
            "Unexpected return type %s from upload_folder; upload will run synchronously",
            type(upload_result),
        )
        future = None

    if isinstance(future, Future):
        setattr(future, "_hf_revision", revision)
    log.info("Started Hub upload for %s@%s from %s", hub_model_id, revision, checkpoint_dir)
    return future
