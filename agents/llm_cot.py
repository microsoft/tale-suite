import argparse

import llm
import numpy as np
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random_exponential,
)

import twbench
from agents.llm import LLMAgent, format_messages_to_markdown
from twbench.agent import register
from twbench.utils import count_tokens, is_recoverable_error

SYSTEM_PROMPT = (
    "You are playing a text-based game and your goal is to finish it with the highest score."
    " After each set of observation, reflect on your current situation."
    " Consider what places you've explored, what places you haven't explored, and what items you may need later in the game."
    " Using this, describe a current objective and explain why this objective will help you finish the game."
    " Once you have finished reflecting, provide a *single* short phrase to interact with the game in brackets, e.g. `[[get lamp]]` (without the backticks)."
)


# For the LLMWlkThrAgent, the sysprompt is initialized in the __init__ function as we need to change it once we extract the walkthrough from the env
class LLMCoTAgent(LLMAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def uid(self):
        return (
            f"LLMAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context}"
            f"_t{self.act_temp}"
            f"_conv{self.conversation is not None}"
            f"CoT Agent"
        )

    def act(self, obs, reward, done, infos):
        conversation = self.conversation or self.model.conversation()
        conversation.responses = conversation.responses[-self.context :]
        # TODO: Add a message saying the history was truncated?

        prompt = f"{obs}\n>" if self.conversation else self.build_prompt(obs)

        if not self.allows_system_prompt:
            if len(conversation.responses) == 0:
                # Model doesn't support system prompt and this is the first message in the conversation.
                prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
            else:
                # Make sure the system prompt is added to the first message in the conversation.
                first_prompt = conversation.responses[0].prompt
                if SYSTEM_PROMPT not in first_prompt.prompt:
                    first_prompt.prompt = f"{SYSTEM_PROMPT}\n\n{first_prompt.prompt}"

        response = self._llm_call(
            conversation,
            prompt=prompt,
            system=SYSTEM_PROMPT if self.allows_system_prompt else None,
            temperature=self.act_temp,
            seed=self.seed,
            top_p=1,
            stream=False,
        )

        reflection = response.text().strip()
        self.history.append((obs, f"> {reflection}\n"))

        action = reflection.split("[[")[-1].split("]]")[0]
        # Compute usage statistics
        messages = conversation.responses[-1]._prompt_json["messages"]
        new_tokens = count_tokens(text=response.text())
        stats = {
            "prompt": format_messages_to_markdown(messages),
            "response": response.text(),
            "nb_tokens": count_tokens(messages=messages) + new_tokens,
            "new_tokens": new_tokens,
        }

        return action, stats


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
        "--act-temp",
        type=float,
        default=0.0,
        help="Temperature for LLM when taking actions. Default: %(default)s",
    )
    group.add_argument(
        "--context-limit",
        type=int,
        default=100,
        help="Limit context for LLM (in conversation turns). Default: %(default)s",
    )
    group.add_argument(
        "--conversation",
        action="store_true",
        help="Enable conversation mode. Otherwise, use single prompt.",
    )

    return parser


register(
    name="cot",
    desc=(
        "This agent uses a n-step chain of thought to attempt to progress through the game. Please see llm_cot.py for more details about the prompt."
    ),
    klass=LLMCoTAgent,
    add_arguments=build_argparser,
)
