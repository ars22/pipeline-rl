"""
Dataset loader for the AI-CoScientist rubric (aicsi_rubric) domain.

Expected dataset format: ACSci/v3-train with columns:
  - query:           the scientific query
  - rubric_criteria: JSON string containing a list of criterion dicts:
                       [{"criterion_id": int, "aspect": str, "criterion": str, ...}, ...]
  - title:           paper title (used for logging / context)

Produces problem dicts with keys:
  - "dataset":  source dataset name
  - "task":     the query text
  - "rubric":   list of criterion dicts (parsed from rubric_criteria JSON)
  - "title":    paper title
  - "id":       sequential integer id
"""

import json
import logging
from typing import Any, Dict, List

from datasets import load_dataset
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger(__name__)


def _parse_rubric(rubric_criteria: Any) -> list:
    """Parse rubric_criteria into a list of criterion dicts."""
    if isinstance(rubric_criteria, list):
        return rubric_criteria
    if isinstance(rubric_criteria, str):
        try:
            parsed = json.loads(rubric_criteria)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse rubric_criteria JSON: {e}")
    return []


def process_rubric_problem(dataset, dataset_name: str):
    """Yield problem dicts from a dataset with query and rubric_criteria columns."""
    for row in dataset:
        query = row.get("query", "")
        rubric_raw = row.get("rubric_criteria", "")
        title = row.get("title", "")

        if not query.strip():
            continue

        rubric = _parse_rubric(rubric_raw)
        if not rubric:
            logger.warning(f"[{dataset_name}] Skipping row with empty rubric")
            continue

        yield {
            "dataset": dataset_name,
            "task": query,
            "rubric": rubric,
            "title": title,
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
    Load aicsi_rubric datasets from HuggingFace Hub.

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

            load_args = (hub_id,)
            if config is not None:
                load_args += (config,)
            dataset = load_dataset(*load_args, split=split, trust_remote_code=trust_remote_code)

            if "query" in dataset.column_names and "rubric_criteria" in dataset.column_names:
                samples = [s for s in process_rubric_problem(dataset, hub_id.split("/")[-1]) if s is not None]
            else:
                logger.warning(
                    f"[{hub_id}] Missing 'query' or 'rubric_criteria' columns; "
                    f"found: {dataset.column_names}"
                )
                samples = []

            logger.info(
                f"Loading hub dataset {hub_id}"
                + (f"/{config}" if config else "")
                + f" split={split}: {len(samples)} samples"
            )
            all_samples += add_ids(samples)

        elif isinstance(spec, str) and "/" in spec:
            dataset = load_dataset(spec, split="train", trust_remote_code=True)
            if "query" in dataset.column_names and "rubric_criteria" in dataset.column_names:
                samples = [s for s in process_rubric_problem(dataset, spec.split("/")[-1]) if s is not None]
            else:
                samples = []
            logger.info(f"Loading hub dataset {spec} split=train: {len(samples)} samples")
            all_samples += add_ids(samples)

        else:
            logger.warning(f"Unrecognized dataset spec, skipping: {spec!r}")

    if seed is not None:
        import random
        rng = random.Random(seed)
        rng.shuffle(all_samples)

    return all_samples
