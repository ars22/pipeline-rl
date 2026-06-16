import logging
import os
from typing import Any

import requests
import transformers
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

PIPELINERL_LLM_TOKEN = "PIPELINERL_LLM_TOKEN"


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, DictConfig):
        plain = OmegaConf.to_container(value, resolve=True)
        return dict(plain) if isinstance(plain, dict) else {}
    if isinstance(value, dict):
        return dict(value)
    return dict(value)


class Prompt(BaseModel):
    messages: list[dict] = Field(default_factory=list)
    tools: list[dict] | None = None
    token_ids: list[int] = Field(default_factory=list)


class LLMOutput(BaseModel):
    content: str | None = None
    tool_calls: list[Any] | None = None


class TokenLogprob(BaseModel):
    logprob: float
    token_id: int
    generated: int = 1


class LLMCall(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    prompt: Prompt
    output: LLMOutput
    prompt_length_tokens: int = -1
    output_length_tokens: int = -1
    cached: bool = False
    logprobs: list[TokenLogprob] = Field(default_factory=list)


class TrainableLLM(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    base_url: str
    model_name: str
    tokenizer_name: str | None = None
    parameters: dict[str, Any] | DictConfig = Field(default_factory=dict)
    stream: bool = False
    collect_logprobs: bool = False
    use_cache: bool = False
    observe_llm_calls: bool = False
    api_token: str = Field(default="", exclude=True)
    tokenizer: transformers.PreTrainedTokenizerBase | None = Field(default=None, exclude=True)

    def model_post_init(self, __context: Any) -> None:
        self.api_token = os.getenv(PIPELINERL_LLM_TOKEN, "") or os.getenv("OPENAI_API_KEY", "")
        self.parameters = _to_plain_dict(self.parameters)

    def load_tokenizer(self) -> transformers.PreTrainedTokenizerBase:
        if self.tokenizer is not None:
            return self.tokenizer

        tokenizer_name = self.tokenizer_name or self.model_name
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id
            else:
                tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.tokenizer = tokenizer
        return tokenizer

    def log_output(self, prompt: Prompt, output: LLMOutput, count_tokens: bool = True) -> LLMCall:
        llm_call = LLMCall(prompt=prompt, output=output)
        if not count_tokens:
            return llm_call

        tokenizer = self.load_tokenizer()
        prompt_text = tokenizer.apply_chat_template(
            prompt.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        full_messages = prompt.messages + [{"role": "assistant", "content": output.content or ""}]
        full_text = tokenizer.apply_chat_template(full_messages, tokenize=False)
        llm_call.prompt_length_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        llm_call.output_length_tokens = len(tokenizer.encode(full_text[len(prompt_text):], add_special_tokens=False))
        return llm_call

    def get_batch_logprobs_token_ids(
        self, prompt_token_ids: list[list[int]], completion_token_ids: list[list[int]]
    ) -> list[dict[str, Any]]:
        self.load_tokenizer()

        headers = {"Content-Type": "application/json"}
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"

        generation_args = {
            "model": self.model_name,
            "prompt": [pids + cids for pids, cids in zip(prompt_token_ids, completion_token_ids)],
            "temperature": 0.0,
            "max_tokens": 0,
            "logprobs": 0,
            "echo": True,
            "include_stop_str_in_output": True,
            "skip_special_tokens": False,
            "n": 1,
            "stream": False,
        }
        response = requests.post(
            f"{self.base_url}/v1/completions",
            json=generation_args,
            headers=headers,
            verify=False,
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()

        all_logprobs: list[dict[str, Any]] = []
        for idx, completion_ids in enumerate(completion_token_ids):
            logprobs = []
            prompt_logprobs = payload["choices"][idx]["prompt_logprobs"][-len(completion_ids):]
            for logprob_entry in prompt_logprobs:
                if not logprob_entry:
                    continue
                for token_id, token_info in logprob_entry.items():
                    token_info = dict(token_info)
                    token_info.update({"generated": 0, "token_id": token_id})
                    logprobs.append(token_info)
            all_logprobs.append({"content": logprobs})
        return all_logprobs
