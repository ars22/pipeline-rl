from itertools import takewhile
from typing import Generator, Literal, Union

from pydantic import Field

from tapeagents.agent import Agent
from tapeagents.core import (
    Action,
    AgentStep,
    FinalObservation,
    LLMOutputParsingFailureAction,
    Observation,
    PartialStep,
    SetNextNode,
    Step,
    StopStep,
    Tape,
    TapeMetadata,
    Thought,
)
from tapeagents.dialog_tape import DialogContext, UserStep
from tapeagents.llms import LLMStream
from tapeagents.nodes import StandardNode
from tapeagents.utils import FatalError
from tapeagents.steps import ActionExecutionFailure, ReasoningThought
from tapeagents.tools.browser import (
    ClickBIDAction,
    ClickCoordinatesAction,
    GoBackAction,
    GoForwardAction,
    HoverAction,
    InputTextAction,
    PageDownAction,
    PageObservation,
    PageUpAction,
    PressAction,
    SelectOptionAction,
)
from tapeagents.tools.simple_browser import PageObservation


################### STEPS ###################


class WebTask(Observation):
    kind: Literal["task"] = "task"
    task: str


class ReflectionThought(Thought):
    """
    Review the current state of the page and previous steps to find the best possible next action to accomplish the task:
    1. Produce reasoning thougt explaining the observed state, think about which blocks could be relevant to the given task and its current state, note relevant BIDs and coordinates.
    2. Produce list of things to do to accomplish the task. Mention all the fields that need to be filled, buttons that need to be clicked, etc.
    3. Reflect if you last action succes, check if it produced the expected effect on the page.
    4. Reflect on you last actions to check if you are being stuck with repeating the same action. If you repeatedly failed to perform the same action with the page element, propose a new way of interacting.
    5. Describe the next action to be performed and expected effect on the page.
    """

    kind: Literal["reflection_thought"] = "reflection_thought"
    page_state: str = Field(
        description="description of the current page state, think about which blocks could be relevant to the given task and its current state, note relevant BIDs and coordinates"
    )
    last_action_kind: str = Field(description="describe the kind of the last action")
    last_action_achieved_effect: str = Field(
        description="describe the the changes in the page state achieved by the last action that you can observe"
    )
    last_action_expected_effect: str = Field(
        description="describe the expected changes in the page state that should be achieved by the last action"
    )
    last_action_success: bool = Field(
        description="success of the last action, True if it produced the expected effect on the page"
    )
    last_action_repetitions: str = Field(
        description="check if you are being stuck with repeating the same action and describe here"
    )
    are_we_stuck: bool = Field(description="if you are being stuck with repeating the same action")
    task_solved: bool = Field(description="if the main task given by used is solved")
    todo_list: list[str] = Field(
        description="detailed list of things to do to accomplish the task, down to the fields level"
    )
    next_action: str = Field(description="describe the next action to be performed and expected effect on the page")
    page_line: str = Field(
        description="verbatim quote of the line of the page that contains required element. Empty if next action is not related to specific page element"
    )


class FinalAnswerAction(StopStep):
    """
    Action that provides the final answer to the user after completing the task.
    Should be produced when the agent has finished the task.
    When the task has question about numerical value, the answer should contain only one number!
    """

    kind: Literal["final_answer_action"] = "final_answer_action"
    text: str = Field(description="final answer to the user")


WebTapeStep = Union[
    UserStep,
    WebTask,
    PageObservation,
    FinalObservation,
    ActionExecutionFailure,
    LLMOutputParsingFailureAction,
    # thoughts
    ReasoningThought,
    ReflectionThought,
    # browser actions
    ClickBIDAction,
    ClickCoordinatesAction,
    SelectOptionAction,
    HoverAction,
    InputTextAction,
    PressAction,
    PageDownAction,
    PageUpAction,
    # TabFocusAction,
    # NewTabAction,
    # CloseTabAction,
    GoBackAction,
    GoForwardAction,
    # GotoPageAction,
    FinalAnswerAction,
    SetNextNode,
]


class WebTapeMetadata(TapeMetadata):
    task_name: str = Field(default_factory=str)
    seed: int = Field(default_factory=int)
    other: dict = Field(default_factory=dict)


class WebTape(Tape[DialogContext, WebTapeStep]):
    metadata: WebTapeMetadata = Field(default_factory=WebTapeMetadata)
    context: DialogContext = DialogContext(tools=[])


WebAgentStep = (
    # thoughts
    ReasoningThought,
    ReflectionThought,
    # browser actions
    ClickBIDAction,
    ClickCoordinatesAction,
    SelectOptionAction,
    HoverAction,
    InputTextAction,
    PressAction,
    # TabFocusAction,
    # NewTabAction,
    # CloseTabAction,
    GoBackAction,
    GoForwardAction,
    # GotoPageAction,
    FinalAnswerAction,
)


WebAgentAction = (
    # browser actions
    ClickBIDAction,
    ClickCoordinatesAction,
    SelectOptionAction,
    HoverAction,
    InputTextAction,
    PressAction,
    # TabFocusAction,
    # NewTabAction,
    # CloseTabAction,
    GoBackAction,
    GoForwardAction,
    # GotoPageAction,
    FinalAnswerAction,
)


################### UTILS ###################


class StepRepeatMonitor:
    def __init__(self, last_step=None, repeat_count=0, max_repeats=4):
        self.last_step = last_step
        self.repeat_count = repeat_count
        self.max_repeats = max_repeats

    def should_stop(self, step) -> bool:
        if step == self.last_step:
            self.repeat_count += 1
            return self.repeat_count > self.max_repeats
        self.repeat_count = 0
        self.last_step = step
        return False


################### NODES ###################

class WebNode(StandardNode):
    max_retries: int = (
        3  # max number of times to retry the current node if the last step is an LLMOutputParsingFailureAction
    )
    current_retries: int = 0  # current number of retries for the current node
    max_same_action: int = 4  # max number of times to repeat the same action
    max_chars_page_observation: int = 2000  # max number of characters to keep in PageObservation["text"]

    def tape_to_messages(self, tape: WebTape, steps_description: str) -> list[dict]:
        """
        Converts a Tape object and steps description into a list of messages for LLM conversation.

        Modifications from the original StandardNode:
        - If the last n steps are LLMOutputParsingFailureAction, put the guidance before the error steps
          and add a new guidance message that asks to retry.
        - Do not render older LLMOutputParsingFailureAction steps that got solved.
        - Always render WebTask observations as they contain the instructions
        - Truncate PageObservation steps up to `max_chars_page_observation` chars (instead of 100 by default).

        Args:
            tape (Tape): A Tape object containing conversation steps.
            steps_description (str): A description of the conversation steps.

        Returns:
            list[dict]: A list of dictionaries representing the conversation messages.
                       Each dictionary contains 'role' and 'content' keys.
                       Roles can be 'system', 'user', or 'assistant'.
                       The system prompt is always the first message.
                       If steps_description is provided, it's added as a user message.
                       Messages from tape are added with roles based on step type.
                       If guidance exists, it's added as the final user message.
        """
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if steps_description:
            messages.append({"role": "user", "content": steps_description})
        # ignore the last n consecutive parsing error steps for now, will add them later
        last_parsing_error_steps: list[LLMOutputParsingFailureAction] = []
        for step in reversed(tape.steps):
            if isinstance(step, LLMOutputParsingFailureAction):
                last_parsing_error_steps.append(step)
            else:
                break
        # put back the last n parsing error steps in the original order
        last_parsing_error_steps.reverse()
        # remove all the parsing error steps
        cleaned_steps = [step for step in tape.steps if not isinstance(step, LLMOutputParsingFailureAction)]
        # get the positions of all observations to trim the old ones
        page_observation_idx = [i for i, step in enumerate(cleaned_steps) if isinstance(step, PageObservation)]
        for i, step in enumerate(cleaned_steps):
            role = "assistant" if isinstance(step, AgentStep) else "user"
            if isinstance(step, PageObservation):
                if i not in page_observation_idx[-self.trim_obs_except_last_n :]:
                    # skip old page observations
                    continue
                else:
                    view = step.short_view(max_chars=self.max_chars_page_observation)
                    # view = f"Page Observation: ```json\n{view}\n```"
            elif isinstance(step, WebTask):
                view = f"Task: {step.task}"
            elif isinstance(step, UserStep):
                view = step.content
            elif isinstance(step, ReasoningThought):
                view = step.reasoning
            else:
                view = step.llm_view()
            messages.append({"role": role, "content": view})
        # add the initial guidance message
        if self.guidance:
            messages.append({"role": "user", "content": self.guidance})
        # add the last n parsing error steps and the new guidance message that asks to retry
        for step in last_parsing_error_steps:
            messages.append({"role": "assistant", "content": step.llm_view()})
            messages.append(
                {
                    "role": "user",
                    "content": "You made a mistake. Look at your generation (in 'llm_output') and the error message (in 'error') and please try again.",
                }
            )
        return messages

    def generate_steps(
        self, agent: Agent, tape: WebTape, llm_stream: LLMStream
    ) -> Generator[Step | PartialStep, None, None]:
        """
        Generates a sequence of steps based on the LLM stream output.

        This method processes the output from a language model stream and converts it into a series of steps.
        It handles the parsing of completions and post-processing of steps.

        Modifications from the original StandardNode:
        - If the last step is an LLMOutputParsingFailureAction, retry the current node
          **up to max_retries times**.
        - If the last action is repeated more than **max_same_action times**, stop the task.

        Args:
            agent (Any): The agent instance that will execute the steps.
            tape (Tape): The tape object containing the execution context and history.
            llm_stream (LLMStream): The stream of language model outputs to process.

        Yields:
            Union[Step, PartialStep]: Individual steps generated from the LLM stream output.

        Raises:
            FatalError: If no completions are generated from the LLM stream.

        Note:
            - If the node has a next_node defined and the final step is not a StopStep,
              it will yield a SetNextNode step to continue the execution flow.
        """
        # get the last action and the number of repetitions
        previous_actions = [step for step in tape.steps if isinstance(step, Action)]
        last_action = previous_actions[-1] if previous_actions else None
        n_last_actions = len(list(takewhile(lambda x: x == last_action, reversed(previous_actions))))
        action_monitor = StepRepeatMonitor(last_step=last_action, repeat_count=n_last_actions, max_repeats=self.max_same_action)

        new_steps = []
        for event in llm_stream:
            if not event.output:
                continue
            if event.output.content:
                new_steps += list(self.parse_completion(event.output.content))
            if event.output.tool_calls and self.use_function_calls:
                new_steps += [self.tool_call_to_step(agent, tool_call) for tool_call in event.output.tool_calls]

            for i, step in enumerate(new_steps):
                step = self.postprocess_step(tape, new_steps[:i], step)
                yield step
                # check that the new step is not a repetition of the last action up to max_same_action times
                if isinstance(step, Action) and action_monitor.should_stop(step):
                    raise FatalError(f"Max same action reached! {step}")
                # if the last step was an LLMOutputParsingFailureAction, retry the current node up to max_retries times
                if isinstance(step, LLMOutputParsingFailureAction):
                    if self.current_retries < self.max_retries:
                        retry_step = SetNextNode(next_node=self.name)
                        yield retry_step
                        self.current_retries += 1
                        new_steps.append(retry_step)
                        break
                    else:
                        raise FatalError(f"Max retries reached for node {self.name}!")
                else:
                    self.current_retries = 0
        if not new_steps:
            raise FatalError("No completions!")
        if (
            self.next_node
            and not isinstance(new_steps[-1], StopStep)
            and not any(isinstance(step, SetNextNode) for step in new_steps)
        ):
            yield SetNextNode(next_node=self.next_node)

