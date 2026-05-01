"""
Dataset loader for the AI-CoScientist (aicsi) domain.

Expected dataset format: a HuggingFace dataset with a "messages" column where
  - messages[0]["content"] is the scientific query
  - messages[1]["content"] is the oracle (ground-truth) response

Produces problem dicts with keys:
  - "dataset": source dataset name
  - "task":    the query (messages[0]["content"])
  - "answer":  the oracle response (messages[1]["content"])
  - "title":   optional short title (first line of query, used as PAPER_TITLE in grader prompt)
"""

import logging
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


def _extract_title(query: str, max_chars: int = 120) -> str:
    """Use the first non-empty line of the query as a short title for the grader prompt."""
    for line in query.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:max_chars]
    return ""


def process_aicsi_problem(dataset, dataset_name: str):
    """Yield problem dicts from a dataset with a 'messages' column."""
    for row in dataset:
        messages = row["messages"]
        if len(messages) < 2:
            logger.warning(f"[{dataset_name}] Skipping row with fewer than 2 messages")
            continue
        msg0 = messages[0]
        msg1 = messages[1]
        query = msg0["content"] if isinstance(msg0, dict) else str(msg0)
        oracle = msg1["content"] if isinstance(msg1, dict) else str(msg1)
        if not query.strip() or not oracle.strip():
            continue
        yield {
            "dataset": dataset_name,
            "task": query,
            "answer": oracle,
            "title": _extract_title(query),
        }


def add_ids(samples: List[Dict]) -> List[Dict]:
    for i, s in enumerate(samples):
        s.setdefault("id", i)
    return samples


def load_datasets(
    dataset_names: List[str | Dict[str, Any]] | Dict[str, Any] | str | None,
    seed: int | None = None,
) -> List[Dict]:
    """
    Load aicsi datasets from HuggingFace Hub or local specs.

    Each entry in dataset_names can be:
      - A dict with "hub_id", optional "split" (default "train"), optional "config"
      - A plain "org/repo" string (loaded with split="train")

    Returns a flat list of problem dicts.
    """
    if dataset_names is None:
        return []

    if isinstance(dataset_names, (DictConfig, ListConfig)):
        dataset_names = OmegaConf.to_container(dataset_names, resolve=True)

    if isinstance(dataset_names, dict):
        dataset_names = [dataset_names]
    elif isinstance(dataset_names, str):
        dataset_names = [dataset_names]
    elif not isinstance(dataset_names, list):
        dataset_names = list(dataset_names)

    all_samples: List[Dict] = []

    for spec in dataset_names:
        if isinstance(spec, dict):
            hub_id = spec.get("hub_id")
            if not hub_id:
                raise ValueError("Dataset spec must include a 'hub_id' field.")
            config = spec.get("config")
            split = spec.get("split", "train")
            trust_remote_code = spec.get("trust_remote_code", True)

            load_args: Tuple[Any, ...] = (hub_id,)
            if config is not None:
                load_args += (config,)
            dataset = load_dataset(*load_args, split=split, trust_remote_code=trust_remote_code)

            if "messages" in dataset.column_names:
                samples = [s for s in process_aicsi_problem(dataset, hub_id.split("/")[-1]) if s is not None]
            else:
                # Fallback: treat each row as a pre-processed problem dict
                samples = [dict(row) for row in dataset]
                for s in samples:
                    s.setdefault("dataset", hub_id)

            logger.info(
                f"Loading hub dataset {hub_id}"
                + (f"/{config}" if config else "")
                + f" split={split}: {len(samples)} samples"
            )
            all_samples += add_ids(samples)

        elif isinstance(spec, str) and "/" in spec:
            dataset = load_dataset(spec, split="train", trust_remote_code=True)
            if "messages" in dataset.column_names:
                samples = [s for s in process_aicsi_problem(dataset, spec.split("/")[-1]) if s is not None]
            else:
                samples = [dict(row) for row in dataset]
                for s in samples:
                    s.setdefault("dataset", spec)
            logger.info(f"Loading hub dataset {spec} split=train: {len(samples)} samples")
            all_samples += add_ids(samples)

        else:
            logger.warning(f"Unrecognized dataset spec, skipping: {spec!r}")

    if seed is not None:
        import random
        rng = random.Random(seed)
        rng.shuffle(all_samples)

    return all_samples
