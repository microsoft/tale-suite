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
from twbench.utils import TokenCounter, is_recoverable_error, log, messages2conversation

SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score."
    " Upon reading the text observation, generate a plan with subgoals when asked to think step-by-step,"
    " then provide a *single* short phrase to interact with the game when asked to do so, e.g. `get lamp` (without the backticks)."
    " When stuck, try using the `help` command to see what commands are available."
)


def format_messages_to_markdown(messages):
    """Concatenate messages into a single markdown string."""
    markdown_content = ""
    for message in messages:
        role = message["role"].capitalize()
        content = message["content"]
        markdown_content += f"#### {role}\n\n```\n{content}\n```\n\n"
    return markdown_content


class ReactAgent(twbench.Agent):

    def __init__(self, *args, **kwargs):
        self.llm = kwargs["llm"]
        self.model = llm.get_model(self.llm)
        self.token_counter = TokenCounter(self.model.model_name)
        self.allows_system_prompt = self.llm not in ["o1-mini", "o1-preview"]

        # Provide the API key, if one is needed and has been provided
        self.model.key = llm.get_key(
            kwargs.get("key"), kwargs["llm"], self.model.key_env_var
        ) or llm.get_key(None, self.model.needs_key, self.model.key_env_var)

        self.seed = kwargs.get("seed", 1234)
        self.rng = np.random.RandomState(self.seed)

        self.history = []
        self.context_limit = kwargs["context_limit"]
        if self.context_limit is not None:
            assert self.context_limit > 0, "--context-limit must be greater than 0."

        self.cot_temp = kwargs.get("cot_temp", 0.0)
        self.act_temp = kwargs.get("act_temp", 0.0)
        self.conversation = kwargs.get("conversation", False)

    @property
    def uid(self):
        return (
            f"ReactAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_t{self.act_temp}"
            f"_cot{self.cot_temp}"
            f"_conv{self.conversation is not None}"
        )

    @property
    def params(self):
        return {
            "llm": self.llm,
            "seed": self.seed,
            "context_limit": self.context_limit,
            "act_temp": self.act_temp,
            "cot_temp": self.cot_temp,
            "conversation": self.conversation is not None,
        }

    @retry(
        retry=retry_if_exception(is_recoverable_error),
        wait=wait_random_exponential(multiplier=1, max=40),
        stop=stop_after_attempt(100),
    )
    def _llm_call_from_conversation(self, conversation, *args, **kwargs):
        response = conversation.prompt(*args, **kwargs)
        response.duration_ms()  # Forces the response to be computed.
        return response

    def _llm_call_from_messages(self, messages, *args, **kwargs):
        conversation = messages2conversation(self.model, messages)
        prompt = messages[-1]["content"]
        system = messages[0]["content"] if self.allows_system_prompt else None

        return self._llm_call_from_conversation(
            conversation, prompt=prompt, system=system, *args, **kwargs
        )

    def act(self, obs, reward, done, infos):
        question = "// Based on the above information (history), what is the best action to take? Let's think step by step, "
        messages = self.build_messages(f"{obs}> ", question, [])
        response = self._llm_call_from_messages(
            messages,
            temperature=self.cot_temp,
            seed=self.seed,
            top_p=1,
            stream=False,
        )

        answer = response.text().strip()
        log.debug(colored(question, "cyan"))
        log.debug(colored(answer, "green"))

        prompt = "// Provide your chosen action on a single line while respecting the desired format."
        messages = self.build_messages(f"{obs}> ", prompt, [(question, answer)])
        response = self._llm_call_from_messages(
            messages,
            temperature=self.act_temp,
            seed=self.seed,
            top_p=1,
            stream=False,
        )

        action = response.text().strip()
        self.history.append((f"{obs}> ", f"{action}\n"))
        log.debug(colored(prompt, "cyan"))

        # Compute usage statistics
        stats = {
            "prompt": format_messages_to_markdown(messages),
            "response": response.text(),
            "nb_tokens": self.token_counter(messages=messages, text=response.text()),
        }

        return action, stats

    def build_messages(self, observation, question, qa_history):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        limit = self.context_limit or len(self.history) + 1

        for i, (obs, action) in enumerate(self.history[-limit:]):
            if len(self.history) >= limit and i == 0:
                # Add the current observation.
                obs = (
                    f"// History has been truncated to the last {limit} steps.\n...\n> "
                )

            messages.append({"role": "user", "content": obs})
            messages.append({"role": "assistant", "content": action})

        messages.append({"role": "user", "content": observation})

        for q, a in qa_history:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        messages.append({"role": "user", "content": question})

        if not self.conversation:
            # Merge all messages into a single message except for the system.
            content = "".join([msg["content"] for msg in messages[1:]])
            messages = messages[:1] + [{"role": "user", "content": content}]

        if not self.allows_system_prompt:
            # Make sure the system prompt is added to the following message.
            messages.pop(0)
            messages[1]["content"] = f"{SYSTEM_PROMPT}\n\n{messages[1]['content']}"

        return messages


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
        help="Limit context for LLM (in conversation turns). Default: no limit",
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
