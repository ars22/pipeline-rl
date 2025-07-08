from dataclasses import dataclass
from typing import Any, Dict, List, Literal, TypeAlias, Union

import torch
from pydantic import BaseModel, ConfigDict, field_validator

ModelClass: TypeAlias = Literal["causal-language-modeling", "seq2seq-language-modeling", "vision2seq-language-modeling"]


class DataPartArgs(BaseModel):
    path: str
    files: list[str] = ["*.jsonl"]
    weight: float = 1.0
    model_config = ConfigDict(frozen=True)


class DataArgs(BaseModel):
    data_parts_train: list[DataPartArgs]
    data_parts_valid: list[DataPartArgs] | None = None
    data_parts_dev: list[DataPartArgs] | None = None
    model_config = ConfigDict(frozen=True)


@dataclass
class TrainingMetrics:
    epoch: int = 0
    passes: int = 0
    completed_steps: int = 0
    samples: int = 0
    tokens: int = 0
    samples_too_old_to_queue: int = 0
    samples_too_old_to_train: int = 0
    last_broadcasted_version: int = 0
    train_loss: float = 1e9
    eval_loss: float = 1e9
    dev_loss: float = 1e9
    grad_norm: float = 0.0
    best_eval_loss: float = 1e9
    best_completed_steps: int = 0
    lr: float = 0.0
    max_batch_len: int = 0
    min_batch_len: int = int(1e9)
    time_waiting_for_data: float = 0.0


class PipelineBatchEncoding(BaseModel):
    """Pydantic model for batch encoding with automatic tensor conversion."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # All fields are tensors after validation
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    labels: torch.LongTensor
    position_ids: torch.LongTensor | None = None  # Required when seq_packing=True
    
    rewards: torch.FloatTensor
    advantages: torch.FloatTensor
    ref_logprobs: torch.FloatTensor
    old_logprobs: torch.FloatTensor
    group_tokens: torch.FloatTensor
    overflow: torch.FloatTensor
    
    model_version: int
    sentinel: bool = False
    is_packed: bool = False 
    
    # Visual feature fields (optional, for multimodal models)
    pixel_values: torch.FloatTensor | None = None
    image_grid_thw: List[List[int]] | None = None
    
    @field_validator('input_ids', 'attention_mask', 'labels', 'position_ids', mode='before')
    @classmethod
    def convert_to_long_tensor(cls, v: List[int] | torch.Tensor | None) -> torch.LongTensor | None:
        """Convert lists to long tensors."""
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return v.long()
        return torch.tensor(v, dtype=torch.long)
    
    @field_validator('rewards', 'advantages', 'ref_logprobs', 'old_logprobs', 'group_tokens', 'overflow', 'pixel_values', mode='before')
    @classmethod
    def convert_to_float_tensor(cls, v: List[float] | torch.Tensor | None) -> torch.FloatTensor | None:
        """Convert lists to float tensors."""
        if v is None:
            return None
        if isinstance(v, torch.Tensor):
            return v.float()
        return torch.tensor(v, dtype=torch.float)
    
    def to_device(self, device: Union[str, torch.device]) -> 'PipelineBatchEncoding':
        """Move all tensors to the specified device and return updated instance."""
        for field_name in self.model_fields:
            field_value = getattr(self, field_name)
            if isinstance(field_value, torch.Tensor):
                setattr(self, field_name, field_value.to(device))
        return self
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], **defaults) -> 'PipelineBatchEncoding':
        """Create from dictionary, filling in missing required fields with defaults."""
        # Merge defaults with data
        merged = {**defaults, **data}
        
        # Extract only known fields for the model
        model_fields = {}
        extra_fields = {}
        
        for key, value in merged.items():
            if key in cls.model_fields:
                model_fields[key] = value
            else:
                extra_fields[key] = value
        
        # Create instance with model fields
        instance = cls(**model_fields)
        
        # Add extra fields
        for key, value in extra_fields.items():
            instance.model_extra[key] = value
            
        return instance
