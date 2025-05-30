
from pydantic import BaseModel

from pipelinerl.finetune.data import MASKED_TOKEN_ID
from tapeagents.core import LLMCall, TrainingText
from tapeagents.llms.trainable import TrainableLLM


class RolloutResult(BaseModel):
    training_texts: list[TrainingText]
    metrics: dict[str, float]
    latency: float
    # optional so fields that it can be filled later after RolloutResult is created
    model_version: int | None = None
    dataset_name: str | None = None
    group_id: str | None = None


def make_training_text(llm: TrainableLLM, llm_call: LLMCall) -> TrainingText:
    # TODO: integrate this to TapeAgents
    training_text = llm.make_training_text(llm_call.prompt, llm_call.output)
    if not llm_call.logprobs:
        raise ValueError("Logprobs are required to make training data for RL")
    input_ids = [lp.token_id for lp in llm_call.logprobs]
    labels = [lp.token_id for lp in llm_call.logprobs if lp.generated]
    # Apply masking to input tokens that aren't generated
    labels = [MASKED_TOKEN_ID] * (len(input_ids) - len(labels)) + labels
    training_text.input_ids = input_ids
    training_text.labels = labels
    training_text.logprobs = [lp.logprob for lp in llm_call.logprobs if lp.generated]
    return training_text