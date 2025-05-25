import argparse

import llm
import numpy as np
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

import tales
from tales.agent import register
from tales.token import get_token_counter
from tales.utils import (
    format_messages_to_markdown,
    is_recoverable_error,
    merge_messages,
    messages2conversation,
)
 # Please see llm.py for a more comprehensive example of how to structure a conversation-style agent
# This custom agent example is kept minimal for the sake of simplicity

class LLMCustomAgent(tales.Agent):

    def __init__(self, *args, **kwargs):
        # Still need to get the LLM endpoint for the llm package:
        self.llm = kwargs["llm"]
        self.model = llm.get_model(self.llm)
        self.token_counter = get_token_counter(self.model)

        # You should set this based on your own model
        self.allows_system_prompt = False

        self.seed = kwargs["seed"]
        self.rng = np.random.RandomState(self.seed)

        self.history = []
        self.context_limit = kwargs["context_limit"]
        if self.context_limit is not None:
            assert self.context_limit > 0, "--context-limit must be greater than 0."

        self.act_temp = kwargs["act_temp"]
        self.conversation = kwargs["conversation"]

    @property
    def uid(self):
        return (
            f"LLMCustomAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_t{self.act_temp}"
            f"_conv{self.conversation}"
        )

    @property
    def params(self):
        return {
            "agent_type": "custom",
            "llm": self.llm,
            "seed": self.seed,
            "context_limit": self.context_limit,
            "act_temp": self.act_temp,
            "conversation": self.conversation,
        }

    @retry(
        retry=retry_if_exception(is_recoverable_error),
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(100),
    )
    def build_messages(self, last_obs):
        # Construct the input to the LLM
        all_messages = []
        for obs, action in self.history:
            all_messages.append({'content':obs})
            all_messages.append({'content':action})
        all_messages.append({'content':last_obs})
        return all_messages

    def act(self, obs, reward, done, infos):
        messages = self.build_messages(f"{obs}\n")

        # For the custom agent in this example, we are not using a conversation so we make sure that it is set to false
        assert not self.conversation, "Conversation mode is not supported in this custom agent."

        llm_kwargs = {
            "temperature": self.act_temp,
            "max_tokens": 100,  # Text actions are short phrases.
            "seed": self.seed,
            "stream": False,
        }
        # Construct raw input:
        raw_input = []
        for message in messages:
            raw_input.append(message['content'])

        response = self.model.prompt("".join(raw_input), **llm_kwargs)

        action = response.text().strip().replace("\n", "")
        self.history.append((f"{obs}\n", f"{action}\n"))

        # Compute usage statistics
        stats = {
            "prompt": messages,
            "response": response.text(),
            "nb_tokens": self.token_counter(messages=messages, text=response.text()),
        }

        return action, stats


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("LLMAgent settings")

    group.add_argument(
        "--llm",
        required=True,
        help="LLM to be used for evaluation. You must set this for a custom agent.",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=20241001,
        help="Seed for LLM (not all endpoints support this). Default: %(default)s",
    )
    group.add_argument(
        "--act-temp",
        type=float,
        default=0.0,
        help="Temperature for LLM when taking actions. Default: %(default)s",
    )
    group.add_argument(
        "--context-limit",
        type=int,
        help="Limit context for LLM (in conversation turns). Default: no limit.",
    )
    group.add_argument(
        "--conversation",
        required=True,
        action=argparse.BooleanOptionalAction,
        help="Enable conversation mode. Otherwise, use single prompt.",
    )

    return parser


register(
    name="custom",
    desc=(
        "This agent is an example of how to run your own fine-tuned model on our benchmark."
    ),
    klass=LLMCustomAgent,
    add_arguments=build_argparser,
)
