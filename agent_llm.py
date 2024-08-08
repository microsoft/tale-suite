import os
import openai
# from openai import AzureOpenAI
import numpy as np
import textworld
import logging
import llm 

log = logging.getLogger("tw-bench")

class LLMAgent(textworld.Agent):
    def __init__(self, model, seed=1234, temperature=0.0):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.model = llm.get_model(model)
        self.temperature = temperature
        self.context = []

    def reset(self, env):
        env.display_command_during_render = True
        self.context = []

    def act(self, game_state, reward, done):
        if not game_state.admissible_commands:
            log.error("No valid actions available.")
            return 'RESTART'
        lines = ("\n").join(self.context[-500:])
        prompt = (f"Reply ONLY with the ONE appropriate action selected from the list of valid actions even if the game resets from the beginning.\n"
                  "If the game has reset to the beginnig or stuck in a loop try to take a different action from the one taken last time at the same step. Make sure you alway return ONLY a valid action as the next step after the Output: and no other text.\n"
                  "If the game is stuck in a loop or not recognizing instruction reply with a random action from the previous list without giving a verbose explanation. No need to explain why.\n"
                  "We don't want anything in the output other than the next step.\nHere are some examples of inputs and outputs:\n\n"
                  '------------\nInput: {"feedback": "You are in a closet. There is a gun on the floor. Better get it. To exit, go east. \n\nYou can see a small black pistol here.", "admissible_commands": [\'north\', \'take pistol\', \'east\', \'push pistol to floor\']}\nOutput: take pistol\n------------\n'
                  '------------\nInput: {"feedback": "You are still on the streets. To the north is a restraunt where the mayor ate often. To the east is the Mayor\'s home.", "admissible_commands": [\'west\', \'east\', \'north\', \'put paper down\']}\nOutput: east\n------------\n'
                  'The past 100 lines of the game play are as following to avoid taking the same step again that led to game ending:\n\n' + "\n".join(lines) + "\n\n"
                  '------------\nInput: {"feedback": ' + f'"{game_state.feedback}",'  + "admissible_commands:" + str(game_state.admissible_commands) + "}\nOutput: "
        )
        response = self.model.prompt(prompt, temperature=self.temperature)
        action = response.text()

        action = action.split("\n")[0]
        action = action.strip()

        if action not in game_state.admissible_commands:
            log.warning(f'Invalid action "{action}" received. Choosing a random valid action.')
        
        self.context.append(f'{game_state.feedback}\n> {action}\n')
        log.info(f'{game_state.feedback}\n> {action}\n')

        return action