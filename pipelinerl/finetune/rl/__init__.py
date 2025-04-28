import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Literal
from pydantic import BaseModel, Field

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import BatchEncoding, PreTrainedModel

from .utils import (
    calculate_advantage,
    calculate_rewards_with_implicit_kl,
    sum_sum,
    mean_sum,
    replace_dataset_column,
)

# FIXME: remove a warnings, but might be worth investigating
os.environ["TOKENIZERS_PARALLELISM"] = "false"


logger = logging.getLogger(__name__)

RL_DATA_COLUMNS = [
    "reward",
    "example_weight",
    "rewards",
    "advantages",
    "old_logprobs",
    "ref_logprobs",
]


class RLConfig(BaseModel):
    algo: str = Field(default="grpo", description="Algorithm to use for RL", choices=["grpo", "reinforce"])
    use_advantages: bool = Field(
        default=True,
        description="Use advantages instead of rewards to compute the loss",
    )
    epsilon: float = Field(default=0.2, description="Clip parameter for the ration of log probs")
    reward_minus_kl_coef: float = Field(
        default=0.0,
        # https://arxiv.org/abs/2402.14740
        description="Implicit KL coefficient similar to the RLOO paper",
    )
    kl_coef: float = Field(
        default=0.1,
        description="KL penalty coefficient with reference policy",
    )
    final_kl_coef: float = Field(
        default=0.1,
        description="Final KL penalty coefficient value",
    )
    entropy_bonus: float = Field(
        default=0.0,
        description="Entropy bonus coefficient",
    )
    final_entropy_bonus: float = Field(
        default=0.0,
        description="Final entropy bonus value",
    ) 
    relu_log_p_weights: bool = Field(
        default=False,
        description="ReLU the weights before updating the model",
    )
    clamp_log_ratio_ref_new_value: float = Field(
        default=10,
        description="Clamp the log ratio ref new value",
    )
    aggregate_loss: Literal["mean", "sum"] = Field(
        default="sum",
        description="How to aggregate the loss within a batch (when batch size is 1, there is no difference)",
    )
    overlong_filtering: bool = Field(
        default=False,
        description="Filter out sequence that do not have eos_token_id"
    )
    temperature: float = Field(
        default=1.0,
        description="Temperature for the training log probs",
    )

def make_rl_data_callback(args, current_dir, rl_config, model):
    if rl_config:
        populate_rl_data_ = partial(
            populate_rl_data,
            config=rl_config,
        )
    else:
        populate_rl_data_ = None
    return populate_rl_data_


def linear_decay_coef(current_step: int, max_step: int, initial_coef: float, final_coef: float) -> float:
    """
    Linearly decay the coefficient from initial to final value over the course of training.

    Args:
        current_step (int): Current step in the training
        max_step (int): Maximum number of steps in the training
        initial_coef (float): Initial coefficient value
        final_coef (float): Final coefficient value
    
    Returns:
        float: Linearly decayed coefficient value
    
    """
    return initial_coef + (final_coef - initial_coef) * current_step / max_step


def rl_step(
    model: PreTrainedModel,
    batch: dict,
    current_step: int,
    max_step: int,
    config: RLConfig
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Perform a single RL step on the model using the given batch and config.
    Handles both packed and unpacked sequences.

    Args:
        model (PreTrainedModel): The model to train
        batch (dict): Batch of data containing rewards, advantages, masks, input_ids etc.
        current_step (int): Current training step
        max_step (int): Maximum number of training steps
        config (RLConfig): Configuration for the RL training

    Returns:
        tuple[torch.Tensor, dict[str, float]]: Loss tensor and metrics dictionary
    """
    # pre-compute masks
    masks = batch["labels"] != -100
    masks_shifted = masks[:, 1:]

    # if we have position_ids, we are packing
    is_packed = "position_ids" in batch
    if is_packed:
        position_ids = batch["position_ids"][0]
        # sequence boundary computation
        sequence_starts = torch.where(position_ids == 0)[0]
        seq_boundaries = torch.cat([sequence_starts, torch.tensor([position_ids.shape[0]], device=position_ids.device)])
        num_sequences = len(sequence_starts)

        # ensure we have valid sequence boundaries
        assert num_sequences > 0, "No sequences found in packed batch"
        assert seq_boundaries[-1] == position_ids.shape[0], "Sequence boundaries don't match input length"

        # pre-compute segment boundaries
        segments = list(zip(seq_boundaries[:-1], seq_boundaries[1:]))
    else:
        num_sequences = masks.shape[0]
        segments = None

    model_inputs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": batch["labels"],
    }
    if is_packed:
        model_inputs["position_ids"] = batch["position_ids"]

    outputs = model(**model_inputs)

    # compute log probs and entropy
    logits = outputs.logits[:, :-1, :]
    logits = logits / config.temperature
    logprobs = F.log_softmax(logits, dim=-1)
    probs = F.softmax(logits, dim=-1)
    entropy = -(probs * logprobs).sum(dim=-1)
    del logits, probs

    # get log probs for actual tokens
    new_logprobs = torch.gather(
        logprobs,
        dim=2,
        index=batch["input_ids"][:, 1:].unsqueeze(2)
    ).squeeze(2)
    assert torch.isfinite(new_logprobs).all(), f"new_logprobs is not finite: {new_logprobs}"
    del logprobs

    # get shifted values and compute ratios
    rewards = batch.pop("rewards")[:, 1:]
    advantages = batch.pop("advantages")[:, 1:]
    ref_logprobs = batch["ref_logprobs"][:, 1:]
    old_logprobs = batch["old_logprobs"][:, 1:]
    examples_weights = batch["example_weight"][:, 1:]
    assert new_logprobs.shape == ref_logprobs.shape

    log_ratio_new_old = new_logprobs - old_logprobs
    ratio_new_old = torch.exp(log_ratio_new_old)
    log_ratio_ref_new = ref_logprobs - new_logprobs
    assert torch.isfinite(log_ratio_ref_new).all(), f"log_ratio_ref_new is not finite: {log_ratio_ref_new}"

    # compute weights and KL divergence
    log_p_weights = advantages if config.use_advantages else rewards
    if config.relu_log_p_weights:
        log_p_weights = torch.clamp(log_p_weights, min=0)

    # Create mask for non-zero advantages (using a small epsilon)
    non_zero_advantage_mask = torch.abs(advantages) > 1e-6
    # Combine with the original mask (labels != -100)
    combined_mask = masks_shifted & non_zero_advantage_mask

    clamp_log_ratio_ref_new_indicators = torch.abs(log_ratio_ref_new) > config.clamp_log_ratio_ref_new_value


    log_ratio_ref_new_clamp = torch.clamp(
        log_ratio_ref_new,
        min=-config.clamp_log_ratio_ref_new_value,
        max=config.clamp_log_ratio_ref_new_value,
    )

    approx_kl = torch.exp(log_ratio_ref_new_clamp) - log_ratio_ref_new_clamp - 1  # Schulman KL approx

    assert torch.isfinite(approx_kl).all(), f"approx_kl is not finite: {approx_kl}"
    entropy_bonus_coef = linear_decay_coef(current_step, max_step, config.entropy_bonus, config.final_entropy_bonus)
    kl_coef = linear_decay_coef(current_step, max_step, config.kl_coef, config.final_kl_coef)

    # compute algorithm-specific losses
    match config.algo:
        case "grpo":
            surr1 = ratio_new_old * log_p_weights
            clamped_ratio = torch.clamp(ratio_new_old, 1 - config.epsilon, 1 + config.epsilon)
            clamp_log_ratio_new_old_indicators = (clamped_ratio != ratio_new_old)
            surr2 = clamped_ratio * log_p_weights
            policy_loss = torch.min(surr1, surr2)
        case "reinforce":
            surr1 = torch.zeros_like(ratio_new_old)
            surr2 = torch.zeros_like(ratio_new_old)
            clamp_log_ratio_new_old_indicators = torch.zeros_like(ratio_new_old)
            policy_loss = new_logprobs * log_p_weights
        case _:
            raise ValueError(f"Unknown algorithm {config.algo}")

    # combine loss components
    loss = policy_loss - kl_coef * approx_kl + entropy_bonus_coef * entropy  # 1 x (BxL) x 1
    assert loss.shape == examples_weights.shape, f"Loss shape {loss.shape} does not match example weights shape {examples_weights.shape}"
    loss = loss * examples_weights  # 1 x (BxL) x 1

    if config.aggregate_loss == "mean":
        # Use combined_mask for aggregation
        final_loss = -mean_sum(loss, combined_mask, segments)
    elif config.aggregate_loss == "sum":
        # Use combined_mask for aggregation
        final_loss = -sum_sum(loss, combined_mask, segments)
    else:
        raise ValueError(f"{config.aggregate_loss} is not defined")

    # ensure loss is valid
    assert torch.isfinite(final_loss), f"Non-finite loss detected: {final_loss}"

    # Track the number of tokens masked due to zero advantage
    num_zero_advantage_tokens = sum_sum(~non_zero_advantage_mask, masks_shifted, segments).item()

    # All the stats are average then summed. They will be normalized by the number of sequences at the end of the step
    # Use combined_mask for relevant statistics aggregation
    stats = {
        "loss": final_loss.item(),
        "max_loss": final_loss.item(), # This might not be meaningful anymore if loss components are masked
        "min_loss": final_loss.item(), # This might not be meaningful anymore if loss components are masked
        "reward": mean_sum(rewards, masks_shifted, segments).item(), # Keep original mask for reward stats
        "max_reward": rewards[masks_shifted].max().item(),
        "min_reward": rewards[masks_shifted].min().item(),
        "entropy": mean_sum(entropy, combined_mask, segments).item(),
        "old_logprobs": mean_sum(old_logprobs, combined_mask, segments).item(),
        "new_logprobs": mean_sum(new_logprobs, combined_mask, segments).item(),
        "ref_logprobs": mean_sum(ref_logprobs, combined_mask, segments).item(),
        "advantage": mean_sum(advantages, combined_mask, segments).item(),
        "max_advantage": advantages[combined_mask].max().item() if combined_mask.any() else 0.0, # Use combined mask for filtering
        "min_advantage": advantages[combined_mask].min().item() if combined_mask.any() else 0.0, # Use combined mask for filtering
        "kl": mean_sum(approx_kl, combined_mask, segments).item(),
        "max_kl": approx_kl[combined_mask].max().item() if combined_mask.any() else 0.0, # Use combined mask for filtering
        "min_kl": approx_kl[combined_mask].min().item() if combined_mask.any() else 0.0, # Use combined mask for filtering
        "policy_loss": mean_sum(policy_loss, combined_mask, segments).item(),
        "surr1": mean_sum(surr1, combined_mask, segments).item(),
        "surr2": mean_sum(surr2, combined_mask, segments).item(),
        "ratio_new_old": mean_sum(ratio_new_old, combined_mask, segments).item(),
        "ratio_ref_new": mean_sum(torch.exp(log_ratio_ref_new), combined_mask, segments).item(),
        "ratio_ref_old": mean_sum(torch.exp(ref_logprobs - old_logprobs), combined_mask, segments).item(),
        "clamp_log_ratio_ref_new_indicator": mean_sum(clamp_log_ratio_ref_new_indicators, combined_mask, segments).item(),
        "clamp_log_ratio_new_old_indicator": mean_sum(clamp_log_ratio_new_old_indicators, combined_mask, segments).item(),
        "num_nans": torch.isnan(loss).sum().item(),
        "example_weight": mean_sum(examples_weights, combined_mask, segments).item(),
        "kl_coef": num_sequences * kl_coef,
        "entropy_bonus_coef": num_sequences * entropy_bonus_coef,
        "num_zero_advantage_tokens": num_zero_advantage_tokens, # Add new stat
    }

    return final_loss, stats


def update_rewards_and_advantages(dataset: Dataset, config: RLConfig) -> Dataset:
    """
    Updates the advantages column in the given dataset based on reward statistics.

    Args:
        dataset (Dataset): The input dataset containing rewards and placeholder advantages.

    Returns:
        Dataset: The updated dataset with the updated advantages column.

    """
    df = dataset.to_pandas()

    if config.reward_minus_kl_coef > 0:
        logger.info("Updating Reward with Implicit KL")
        calculate_rewards_with_implicit_kl_ = partial(
            calculate_rewards_with_implicit_kl, reward_minus_kl_coef=config.reward_minus_kl_coef
        )
        df["rewards"] = df.apply(calculate_rewards_with_implicit_kl_, axis=1)
        df["reward"] = df["rewards"].apply(lambda x: np.mean(x))

    # Group by group_id and compute mean and std of reward
    grouped = df.groupby("group_id")["reward"].agg(["mean", "std", "count"]).reset_index()

    # Rename columns for clarity
    grouped.columns = ["group_id", "reward_mean", "reward_std", "count"]

    # Merge the computed statistics back to the original dataset
    df_with_stats = pd.merge(df, grouped, on="group_id", how="left")

    df_with_stats["advantages"] = df_with_stats.apply(calculate_advantage, axis=1)

    # replace advantages entry
    dataset = replace_dataset_column(dataset, "advantages", df_with_stats["advantages"].tolist())

    # Convert back to a Hugging Face Dataset
    return dataset


def assign_example_weights(dataset: Dataset, eos_token_id: int, config: RLConfig) -> Dataset:
    """
    Assigns example weights to the dataset based on the reward column.

    Args:
        dataset (Dataset): The input dataset containing rewards.
        config (RLConfig): Configuration object containing RL training parameters

    Returns:
        Dataset: The updated dataset with the assigned example weights.
    """
    df = dataset.to_pandas()

    def compute_weight(row):
        if eos_token_id not in row["input_ids"] and config.overlong_filtering:
            return [0.0] * len(row["example_weight"])
        else:
            return [1.0] * len(row["example_weight"])

    df["example_weight"] = df.apply(compute_weight, axis=1)
    dataset = replace_dataset_column(dataset, "example_weight", df["example_weight"].tolist())
    return dataset


def populate_rl_data(dataset: Dataset, eos_token_id: int, config: RLConfig) -> Dataset:
    """
    Populates a dataset with reinforcement learning specific data columns.

    Args:
        dataset (Dataset): The input dataset to populate with RL data
        columns (list[str]): List of column names to include in the dataset
        collate_fn (Callable): Function to collate/batch the data
        config (RLConfig): Configuration object containing RL training parameters

    Returns:
        Dataset: The dataset populated with RL-specific columns including rewards and advantages
    """

    logger.debug("Populate RL Data")

    dataset = update_rewards_and_advantages(dataset, config)
    dataset = assign_example_weights(dataset, eos_token_id, config)

    logger.debug("Finish Populate RL Data")
    return dataset


def prepare_rl_fields(
    encoding: BatchEncoding,
    reward: float,
    old_logprobs: list[float],
    ref_logprobs: list[float],
) -> BatchEncoding:
    """
    Convert reward per agent step to reward per token and add returns and advantages placeholders
    """
    target_tokens = [token for token in encoding["labels"] if token != -100]
    assert len(target_tokens) == len(
        old_logprobs
    ), f"Target tokens: {len(target_tokens)}, old logprobs: {len(old_logprobs)}"

    encoding["rewards"] = [reward] * len(encoding["labels"])
    encoding["advantages"] = [0.0] * len(encoding["labels"])  # place holder
    encoding["old_logprobs"] = [0] * (len(encoding["labels"]) - len(old_logprobs)) + old_logprobs
    encoding["ref_logprobs"] = [0] * (len(encoding["labels"]) - len(ref_logprobs)) + ref_logprobs
    encoding["example_weight"] = [0] * len(encoding["labels"]) # place holder
    return encoding
