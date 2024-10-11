import argparse

import llm
import numpy as np
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)
from termcolor import colored

import twbench
from twbench.agent import register
from twbench.utils import count_tokens, is_recoverable_error, log

SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score."
    " Upon reading the text observation, use short phrases to interact with the game, e.g. `get lamp`."
    " When stuck, try using the `help` command to see what commands are available."
)


def build_prompt(observation, history, question, qa_history):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
    ]

    for obs, action in history:
        messages.append({"role": "user", "content": obs})
        messages.append({"role": "assistant", "content": action})

    messages.append({"role": "user", "content": observation})

    for q, a in qa_history:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": question})
    return messages


class ReactAgent(twbench.Agent):

    def __init__(self, model, *args, **kwargs):
        self.model = llm.get_model(model)

        # Retrieve the API key from llm's keys store.
        self.model.key = llm.get_key(
            kwargs.get("key"), model, self.model.key_env_var
        ) or llm.get_key(None, self.model.needs_key, self.model.key_env_var)

        self.seed = kwargs.get("seed", 1234)
        self.rng = np.random.RandomState(self.seed)

        self.history = []
        self.context = kwargs.get("context_limit", -0)  # i.e. keep all.
        self.cot_temp = kwargs.get("cot_temp", 0.0)
        self.act_temp = kwargs.get("act_temp", 0.0)
        self.conversation = None
        if kwargs.get("conversation"):
            self.conversation = self.model.conversation()

    @property
    def uid(self):
        return (
            f"ReactAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context}"
            f"_t{self.act_temp}"
            f"_cot{self.cot_temp}"
            f"_conv{self.conversation is not None}"
        )

    @property
    def params(self):
        return {
            "llm": self.llm,
            "seed": self.seed,
            "context": self.context,
            "act_temp": self.act_temp,
            "cot_temp": self.cot_temp,
            "conversation": self.conversation is not None,
        }

    @retry(
        retry=retry_if_exception(is_recoverable_error),
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(100),
    )
    def _llm_call(self, conversation, *args, **kwargs):
        response = conversation.prompt(*args, **kwargs)
        response.duration_ms()  # Forces the response to be computed.
        return response

    def act(self, obs, reward, done, infos):
        assert self.conversation, "TODO: deal with non conversation mode."

        question = "// Based on the above information (history), what is the best action to take? Let's think step by step, "
        response = self._llm_call(
            self.conversation,
            prompt=f"{obs}\n\n{question}",
            system=SYSTEM_PROMPT,
            temperature=self.cot_temp,
            seed=self.seed,
            top_p=1,
        )
        answer = response.text().strip()
        log.debug(colored(question, "cyan"))
        log.debug(colored(answer, "green"))

        prompt = "// Provide your chosen action on a single line while respecting the desired format."
        response = self._llm_call(
            self.conversation,
            prompt=prompt,
            system=SYSTEM_PROMPT,
            temperature=self.act_temp,
            seed=self.seed,
            top_p=1,
        )
        action = response.text().strip()
        log.debug(colored(prompt, "cyan"))

        self.history.append((obs, "> {action}\n"))

        return action, response


def build_argparser(parser=None):
    parser = parser or argparse.ArgumentParser()
    group = parser.add_argument_group("LLMAgent settings")

    group.add_argument(
        "--llm",
        default="gpt-4o-mini",
        help="LLM to be used for evaluation. Default: %(default)s",
    )
    group.add_argument(
        "--seed",
        type=int,
        default=20241001,
        help="Seed for LLM (not all endpoints support this). Default: %(default)s",
    )
    group.add_argument(
        "--cot-temp",
        type=float,
        default=0.0,
        help="Temperature for LLM when doing chain-of-thoughts. Default: %(default)s",
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
        default=10,
        help="Limit context for LLM (in conversation turns). Default: %(default)s",
    )
    group.add_argument(
        "--conversation",
        action="store_true",
        help="Enable conversation mode. Otherwise, use single prompt.",
    )

    return parser


register(
    name="react",
    desc=(
        "This agent uses a LLM to decide which action to take by following a CoT/ReAct approach."
    ),
    klass=ReactAgent,
    add_arguments=build_argparser,
)
