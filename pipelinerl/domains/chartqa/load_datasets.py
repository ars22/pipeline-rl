import logging
from typing import List
import datasets
from datasets import load_dataset

logger = logging.getLogger(__name__)


def process_chartqa(dataset, dataset_name: str):
    """Process ChartQA dataset into standardized format."""
    logger.info(f"Starting to process {dataset_name} dataset")
    processed_count = 0
    skipped_count = 0
    
    for idx, item in enumerate(dataset):
        # Log first few items for debugging
        if idx < 3:
            logger.info(f"Item {idx} keys: {list(item.keys())}")
            logger.info(f"Item {idx} sample: {dict(list(item.items())[:3])}")
        
        # ChartQA uses 'query' instead of 'question' and 'label' instead of 'answer'
        if "image" not in item or "query" not in item or "label" not in item:
            skipped_count += 1
            if idx < 10:  # Log first 10 skipped items
                logger.warning(f"Skipping item {idx} - missing required fields. Available keys: {list(item.keys())}")
            continue
            
        # ChartQA has chart images and questions about them
        try:
            answer_str = str(item["label"][0]) if isinstance(item["label"], list) else str(item["label"])
            sample = {
                "dataset": dataset_name,
                "image": item["image"],  # PIL Image object
                "question": item["query"],  # Use 'query' field
                "answer": answer_str,  # Use first label if list
                "human_or_machine": item.get("human_or_machine", 0),
            }
            
            # Log first few processed samples
            if processed_count < 3:
                logger.info(f"Processed sample {processed_count}: question='{sample['question'][:50]}...', answer='{sample['answer']}'")
                logger.info(f"Image type: {type(sample['image'])}, size: {sample['image'].size if hasattr(sample['image'], 'size') else 'unknown'}")
            
            processed_count += 1
            yield sample
            
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            logger.error(f"Item data: {item}")
            skipped_count += 1
    
    logger.info(f"Finished processing {dataset_name}: {processed_count} processed, {skipped_count} skipped")


def add_ids(dataset: list[dict]):
    """Add sequential IDs to dataset items."""
    for i, entry in enumerate(dataset):
        entry["id"] = i
    return dataset


def load_problems(dataset_names: List[str] | str | None) -> List[dict]:
    """Load ChartQA datasets and return list of problems."""
    logger.info(f"load_problems called with dataset_names: {dataset_names}")
    
    if dataset_names is None:
        logger.warning("dataset_names is None, returning empty list")
        return []

    if isinstance(dataset_names, str):
        dataset_names = [dataset_names]
        logger.info(f"Converted string to list: {dataset_names}")
    
    datasets_list = []
    
    if "chartqa_train" in dataset_names:
        logger.info("Loading ChartQA train dataset from HuggingFace...")
        try:
            dataset = load_dataset("HuggingFaceM4/ChartQA", split="train", trust_remote_code=True)
            logger.info(f"Loaded raw train dataset with {len(dataset)} items")
            samples = [s for s in process_chartqa(dataset, "chartqa_train") if s is not None]
            logger.info(f"Processed ChartQA train dataset: {len(samples)} samples")
            datasets_list += add_ids(samples)
        except Exception as e:
            logger.error(f"Error loading ChartQA train dataset: {e}")
            raise
    
    if "chartqa_test" in dataset_names:
        logger.info("Loading ChartQA test dataset from HuggingFace...")
        try:
            dataset = load_dataset("HuggingFaceM4/ChartQA", split="test", trust_remote_code=True)
            logger.info(f"Loaded raw test dataset with {len(dataset)} items")
            samples = [s for s in process_chartqa(dataset, "chartqa_test") if s is not None]
            logger.info(f"Processed ChartQA test dataset: {len(samples)} samples")
            datasets_list += add_ids(samples)
        except Exception as e:
            logger.error(f"Error loading ChartQA test dataset: {e}")
            raise
    
    if "chartqa_val" in dataset_names:
        logger.info("Loading ChartQA val dataset from HuggingFace...")
        try:
            dataset = load_dataset("HuggingFaceM4/ChartQA", split="val", trust_remote_code=True)
            logger.info(f"Loaded raw val dataset with {len(dataset)} items")
            samples = [s for s in process_chartqa(dataset, "chartqa_val") if s is not None]
            logger.info(f"Processed ChartQA val dataset: {len(samples)} samples")
            datasets_list += add_ids(samples)
        except Exception as e:
            logger.error(f"Error loading ChartQA val dataset: {e}")
            raise

    logger.info(f"Total datasets loaded: {len(datasets_list)} samples")
    
    if len(datasets_list) == 0:
        logger.error("No ChartQA datasets loaded - this will raise an error")
        raise ValueError("No ChartQA datasets loaded")

    return datasets_list