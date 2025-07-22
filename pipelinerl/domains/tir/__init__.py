"""TIR (Tool Integrated Reasoning) domain for PipelineRL."""

from .agent import TIRMathAgent, solve_task, AnswerAction
from .rollouts import generate_tir_rollout
from .datasets import load_datasets

__all__ = [
    "TIRMathAgent",
    "solve_task",
    "AnswerAction", 
    "generate_tir_rollout",
    "load_datasets"
]