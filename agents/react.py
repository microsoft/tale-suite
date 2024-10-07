import logging

import llm
import numpy as np
from termcolor import colored

import twbench

log = logging.getLogger("tw-bench")


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

        self.context = kwargs.get("context", 100)
        self.cot_temp = kwargs.get("cot_temp", 0.0)
        self.act_temp = kwargs.get("act_temp", 0.0)
        self.conversation = (
            self.model.conversation() if kwargs.get("conversation") else None
        )

    @property
    def uid(self):
        return (
            f"LLMAgent_{self.llm}"
            f"_s{self.seed}"
            f"_c{self.context}"
            f"_t{self.act_temp}"
            f"_cot{self.cot_temp}"
            f"_conv{self.conversation is not None}"
        )

    def act(self, obs, reward, done, infos):
        question = "// Based on the above information (history), what is the best action to take? Let's think step by step, "
        response = self.conversation.prompt(
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
        response = self.conversation.prompt(
            prompt,
            system=SYSTEM_PROMPT,
            temperature=self.act_temp,
            seed=self.seed,
            top_p=1,
        )
        action = response.text().strip()
        log.debug(colored(prompt, "cyan"))

        self.history.append((obs, "> {action}\n"))

        return action, response
