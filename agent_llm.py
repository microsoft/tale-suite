import os
import json
import openai
# from openai import AzureOpenAI
import numpy as np
import textworld
import logging
import llm 

log = logging.getLogger("tw-bench")

class LLMAgent(textworld.Agent):
    def __init__(self, model, *args, **kwargs):
        self.model = llm.get_model(model)
        self.seed = kwargs.get('seed', 1234)
        self.rng = np.random.RandomState(self.seed)
        self.temperature = kwargs.get('temperature', 0.0)
        self.context = kwargs.get('context', 100)
        self.admissible_commands = kwargs.get('admissible_commands', False)
        self.window = []
        self.conversation = None
        if kwargs.get('conversation', False):
            self.conversation = self.model.conversation()

    def reset(self, env):
        env.display_command_during_render = True
        # self.window = []
        # if self.conversation:
        #     self.conversation = self.model.conversation()

    def context_length(self):
        if self.conversation:
            return len(self.conversation.responses[-self.context:])
        else:
            return len(self.window[-self.context:])
    
    def act(self, game_state, reward, done):
        if not self.conversation and not game_state.admissible_commands:
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
        input = json.dumps({"feedback": game_state.feedback, "admissible_commands": game_state.admissible_commands}) if self.admissible_commands else game_state.feedback
        if self.conversation:
            response = self.conversation.prompt(input, system=system_prompt, temperature=self.temperature, seed=self.seed, context=self.context)
        else:
            response = self.model.prompt(system_prompt + context + f"------------\nInput: {input}\nOutput: ", temperature=self.temperature, seed=self.seed)
        
        action = response.text()
        action = action.split("\n")[0]
        action = action.strip()

        if game_state.admissible_commands and action not in game_state.admissible_commands:
            log.warning(f'Invalid action "{action}" received.')
        
        if self.conversation and not game_state.admissible_commands:
            self.conversation.responses[-1]._chunks = ['R', 'E', 'S', 'T', 'A', 'R', 'T']
            action = 'RESTART'

        self.window.append(f'{game_state.feedback}\n> {action}\n')

        return action[:100], response