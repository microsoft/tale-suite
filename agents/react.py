import json
import logging

import numpy as np

import llm
import twbench


log = logging.getLogger("tw-bench")


def build_prompt(observation, history, question, qa_history):
    messages = [
        {"role": "system", "content" : "You are playing a text-based game. Use short phrases to interact with the game, e.g. get lamp. When you are stuck, try using the help command."},
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


QUESTIONS = [
    "// Based on the above information (history), what is the best action to take? Let's think step by step, ",
    "// Provide your chosen action on a single line while respecting the desired format.",
]
QUESTIONS_TEMPERATURE = [0.0, 0.0]


class ReactAgent(twbench.Agent):

    def __init__(self, model, *args, **kwargs):
        self.model = llm.get_model(model)

        # Provide the API key, if one is needed and has been provided
        if self.model.needs_key:
            self.model.key = llm.get_key(kwargs.get("key"), self.model.needs_key, self.model.key_env_var)

        self.seed = kwargs.get('seed', 1234)
        self.rng = np.random.RandomState(self.seed)
        self.temperature = kwargs.get('temperature', 0.0)
        self.context = kwargs.get('context', 100)
        self.admissible_commands = kwargs.get('admissible_commands', False)

        self.history = []

        self.system_prompt = "You are playing a text-based game. Use short phrases to interact with the game, e.g. get lamp."

        if kwargs.get('conversation', False):
            self.conversation = self.model.conversation()

    def act(self, obs, reward, done, infos):
        question = "// Based on the above information (history), what is the best action to take? Let's think step by step, "
        response = self.conversation.prompt(prompt=f"{obs}\n\n{question}", system=self.system_prompt, temperature=0.8, seed=self.seed, top_p=1)
        answer = response.text().strip()
        log.debug(colored(question, "cyan"))
        log.debug(colored(answer, "green"))

        prompt = "// Provide your chosen action on a single line while respecting the desired format."
        response = self.conversation.prompt(prompt, system=self.system_prompt, temperature=0.0, seed=self.seed, top_p=1)
        action = response.text().strip()
        log.debug(colored(prompt, "cyan"))

        self.history.append((obs, "> {answer}\n"))

        return action, response

        if not self.conversation and not infos.get("admissible_commands"):
            log.error("No valid actions available.")
            return 'RESTART', None

        lines = ("\n").join(self.window[-self.context:])
        context = f'The past {self.context} steps of the game play are as following to avoid taking the same step again that led to game ending:\n\n{lines}\n\n' if lines and not self.conversation else ""
        system_prompt = (f"You are a helpful assistant playing a text-based game. Reply ONLY with the ONE appropriate action selected from the list of admissible commands even if the game resets from the beginning.\n"
                  "If the game has reset to the beginnig or stuck in a loop try to take a different action from the one taken last time at the same step. Make sure you alway return ONLY an admissible command as the next step after the Output: and no other text.\n"
                  "If the game is stuck in a loop or not recognizing an invalid action reply with a random action from the previous list without giving a verbose explanation. No need to explain why.\n"
                  "We don't want anything in the output other than the next step and the output can never be an empty string.\nHere are some examples of inputs and outputs:\n\n"
                  '------------\nInput: {"feedback": "You are in a closet. There is a gun on the floor. Better get it. To exit, go east. \n\nYou can see a small black pistol here.", "admissible_commands": [\'north\', \'take pistol\', \'east\', \'push pistol to floor\']}\nOutput: take pistol\n------------\n'
                  '------------\nInput: {"feedback": "You are still on the streets. To the north is a restraunt where the mayor ate often. To the east is the Mayor\'s home.", "admissible_commands": [\'west\', \'east\', \'north\', \'put paper down\']}\nOutput: east\n------------\n'
        )
        input = json.dumps({"feedback": obs, "admissible_commands": infos["admissible_commands"]}) if self.admissible_commands else obs
        if self.conversation:
            response = self.conversation.prompt(input, system=system_prompt, temperature=self.temperature, seed=self.seed, context=self.context, max_tokens=100, top_p=1, stop="\n")
        else:
            response = self.model.prompt(system_prompt + context + f"------------\nInput: {input}\nOutput: ", temperature=self.temperature, seed=self.seed, max_tokens=100, top_p=1, stop="\n")

        action = response.text()
        action = action.encode('utf-8').decode('unicode_escape').strip()
        action = action.split("\n")[0]
        action = action.strip()

        if infos["admissible_commands"] and action not in infos["admissible_commands"]:
            log.warning(f'Invalid action "{action}" received.')

        if self.conversation and not infos["admissible_commands"]:
            self.conversation.responses[-1]._chunks = ['R', 'E', 'S', 'T', 'A', 'R', 'T']
            action = 'RESTART'

        self.window.append(f'{obs}\n> {action}\n')

        return action[:100], response

import os
import argparse
import random

from termcolor import colored
