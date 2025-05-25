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

import os
import json

SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score."
    " Upon reading the text observation, provide a *single* short phrase to interact with the game, e.g. `get lamp` (without the backticks)."
    " When stuck, try using the `help` command to see what commands are available."
)

RESET_PROMPT = (
    "The game has just reset. Try to get to the same score you had before but faster."
    " If you have extra steps, continue to play and try to get a higher score than before."
)


class LLMWalkthroughAgent(tales.Agent):

    def __init__(self, *args, **kwargs):
        self.llm = kwargs["llm"]
        self.model = llm.get_model(self.llm)
        self.token_counter = get_token_counter(self.model)
        self.allows_system_prompt = self.llm not in ["o1-mini", "o1-preview"]

        # Provide the API key, if one is needed and has been provided
        self.model.key = llm.get_key(
            kwargs.get("key"), kwargs["llm"], self.model.key_env_var
        ) or llm.get_key(None, self.model.needs_key, self.model.key_env_var)

        self.seed = kwargs["seed"]
        self.rng = np.random.RandomState(self.seed)

        self.history = []
        self.context_limit = kwargs["context_limit"]
        if self.context_limit is not None:
            assert self.context_limit > 0, "--context-limit must be greater than 0."

        self.act_temp = kwargs["act_temp"]
        self.conversation = kwargs["conversation"]
        self.walkthrough_mask = kwargs["walkthrough_mask"]
        self.reset_idx = -1

    @property
    def uid(self):
        return (
            f"LLMWalkthroughAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context_limit}"
            f"_t{self.act_temp}"
            f"_conv{self.conversation}"
        )

    @property
    def params(self):
        return {
            "agent_type": "llm-walkthrough",
            "llm": self.llm,
            "seed": self.seed,
            "context_limit": self.context_limit,
            "act_temp": self.act_temp,
            "conversation": self.conversation,
            "walkthrough_mask": self.walkthrough_mask
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

    def reset(self, obs, info, env_name):
        # Load the gold trajectory and dump it into the history

        with open('one_game.json', 'r') as file:
            walkthroughs = json.load(file)
        # assert len(self.history) > 0
        walkthrough = walkthroughs[env_name]
        if self.walkthrough_mask > 0:
            # Randomly mask some actions in the walkthrough.
            num_steps = len(walkthrough) - 1
            num_masked = int(num_steps * self.walkthrough_mask)
            if num_masked > 0:
                indices = self.rng.choice(
                    num_steps, size=num_masked, replace=False
                )
                for i in indices:
                    step_key = "step " + str(i + 1)
                    print(f"Masking step {i + 1} ({step_key})")
                    print(f"Original action: {walkthrough[step_key]['assistant']}")
                    walkthrough[step_key]['assistant'] = "MASKED_ACTION"

        for i in range(1, len(walkthrough)):
            step_key = "step " + str(i)
            obs = walkthrough[step_key]['user']
            action = walkthrough[step_key]['assistant']
            self.history.append((f"{obs}\n> ", f"{action}\n"))
        
        self.history.append(("game reset\n>", RESET_PROMPT))

    def act(self, obs, reward, done, infos):
        messages = self.build_messages(f"{obs}\n> ")
        llm_kwargs = {
            "temperature": self.act_temp,
            "max_tokens": 100,  # Text actions are short phrases.
            "seed": self.seed,
            "stream": False,
        }
        if self.llm in [
            "claude-3.5-haiku",
            "claude-3.5-sonnet",
            "claude-3.5-sonnet-latest",
        ]:
            # For these models, we cannot set the seed.
            llm_kwargs.pop("seed")

        if "gemini" in self.llm or "gemma" in self.llm:
            # For these models, we cannot set the seed and max_tokens has a different name.
            llm_kwargs.pop("seed")
            llm_kwargs["max_output_tokens"] = llm_kwargs.pop("max_tokens")

        response = self._llm_call_from_messages(messages, **llm_kwargs)

        action = response.text().strip()
        self.history.append((f"{obs}\n> ", f"{action}\n"))

        # Compute usage statistics
        stats = {
            "prompt": format_messages_to_markdown(messages),
            "response": response.text(),
            "nb_tokens": self.token_counter(messages=messages, text=response.text()),
            "history_length": len(self.history)
        }

        return action, stats

    def build_messages(self, observation):
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

        # Just in case, let's avoid having multiple messages from the same role.
        messages = merge_messages(messages)

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
    group = parser.add_argument_group("LLMWalkthroughAgent settings")

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
        "--walkthrough-mask",
        default=0.0,
        type=float,
        help="Percent of walkthrough actions to mask. Rounded down. Default: %(default)s",
    )
    group.add_argument(
        "--conversation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable conversation mode. Otherwise, use single prompt.",
    )

    return parser


register(
    name="llm-walkthrough",
    desc=(
        "This agent loads the trajectory (walkthrough) of another LLM and uses a LLM attempt to optimize the provided trajectory."
    ),
    klass=LLMWalkthroughAgent,
    add_arguments=build_argparser,
)
