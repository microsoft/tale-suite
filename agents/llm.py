import json
import logging

import numpy as np

import llm
import textworld

log = logging.getLogger("tw-bench")


class LLMAgent(textworld.Agent):
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
        self.window = []
        self.conversation = None
        if kwargs.get('conversation', False):
            self.conversation = self.model.conversation()

    def context_length(self):
        if self.conversation:
            return len(self.conversation.responses[-self.context:])
        else:
            return len(self.window[-self.context:])

    def act(self, obs, reward, done, infos):
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
