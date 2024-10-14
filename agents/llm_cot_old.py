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
    " When prompted for an action, provide a *single* short phrase to interact with the game, e.g. `get lamp` (without the backticks)."
    " Do not provide an action until explicitly asked."
)

COT_PROMPT_1 = "Reflect on your previous experiences. What is your current objective?"

COT_PROMPT_2 = "What is important about your current state?"

COT_PROMPT_3 = "What have you already done? What have you already discovered?"

COT_PROMPT_4 = (
    "Reflect on your previous responses. Given this, what is your next action?"
)

COT_STEPS = [COT_PROMPT_1, COT_PROMPT_2, COT_PROMPT_3, COT_PROMPT_4]


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

    def build_prompt(self, conversation, observation):
        messages = []

        for obs, action in self.history[-self.context :]:
            messages.append({"role": "user", "content": obs})
            messages.append({"role": "assistant", "content": action})

        messages.append({"role": "user", "content": observation})
        prompt = "\n".join([msg["content"] for msg in messages])

        return prompt

    def act(self, obs, reward, done, infos):
        conversation = self.conversation or self.model.conversation()
        conversation.responses = conversation.responses[-self.context :]

        # CoT Turn 1
        prompt = (
            f"{obs}\n>" if self.conversation else self.build_prompt(conversation, obs)
        )
        prompt += "\n" + COT_PROMPT_1

        response = conversation.prompt(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            temperature=self.act_temp,
            seed=self.seed,
            top_p=1,
        )
        self.history.append((obs + "\n" + COT_PROMPT_1, f"{response.text().strip()}\n"))
        token_cnt = count_tokens(text=response.text())

        # CoT Turn 2-4
        for cot_prompt in COT_STEPS[1:]:
            prompt = self.build_prompt(conversation, cot_prompt)
            prev_cot_step = response.text().strip()
            response = conversation.prompt(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                temperature=self.act_temp,
                seed=self.seed,
                top_p=1,
            )
            self.history.append((prev_cot_step, f"{response.text().strip()}\n"))
            token_cnt = count_tokens(text=response.text())

        action = response.text().strip()
        # Compute usage statistics
        messages = conversation.responses[-1]._prompt_json["messages"]
        stats = {
            "prompt": response._prompt_json,
            "response": response.text(),
            "nb_tokens": count_tokens(messages=messages) + token_cnt,
            "new_tokens": token_cnt,
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
