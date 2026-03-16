import time
from typing import Literal

from pydantic import BaseModel

TRAINER_TOPIC = "weight_update_request"


class ParameterInfo(BaseModel):
    name: str
    shape: list[int]
    dtype: str


class WeightUpdateRequest(BaseModel):
    kind: Literal["weight_update_request"] = "weight_update_request"
    version: int
    parameters_info: list[ParameterInfo]
    timestamp: float = time.time()


class WeightUpdateSuccess(BaseModel):
    kind: Literal["weight_update_success"] = "weight_update_success"
    version: int
    timestamp: float = time.time()


class SamplesProcessed(BaseModel):
    kind: Literal["samples_processed"] = "samples_processed"
    samples_processed: int
    timestamp: float = time.time()


TrainerMessage = WeightUpdateRequest | WeightUpdateSuccess | SamplesProcessed
