import os
from openai import AzureOpenAI
import numpy as np
import textworld
import logging

log = logging.getLogger("tw-bench")

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version="2024-02-01",
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
)

def llm(prompt, stop=["\n"]):
    log.info(f'Length of the prompt: {len(prompt)}')
    completion = client.chat.completions.create(
    model="gpt-4o",
    messages= [
    {
        "role": "system",
        "content": "You are a helpful assistant",
    },
    {
      "role": "user",
      "content": prompt
    }],
    max_tokens=10,
    temperature=0,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=stop
    )
    return completion.choices[0].message.content


class LLMAgent(textworld.Agent):
    def __init__(self, seed=1234):
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.prompt = ''

    def reset(self, env):
        env.display_command_during_render = True
        self.prompt = ''

    def act(self, game_state, reward, done):
        valid_actions = game_state.valid_actions
        if not valid_actions:
            log.error("No valid actions available.")
            return 'RESTART'
        lines = self.prompt.splitlines()[-500:]
        prompt = (f"Reply ONLY with the ONE appropriate action selected from the list of valid actions even if the game resets from the beginning.\n"
                  "If the game has reset to the beginnig or stuck in a loop try to take a different action from the one taken last time at the same step. Make sure you alway return ONLY a valid action as the next step after the Output: and no other text.\n"
                  "If the game is stuck in a loop or not recognizing instruction reply with a random action from the previous list without giving a verbose explanation. No need to explain why.\n"
                  "We don't want anything in the output other than the next step.\nHere are some examples of inputs and outputs:\n\n"
                  '------------\nInput: {"feedback": "You are in a closet. There is a gun on the floor. Better get it. To exit, go east. \n\nYou can see a small black pistol here.", "valid_actions": [\'north\', \'take pistol\', \'east\', \'push pistol to floor\']}\nOutput: take pistol\n------------\n'
                  '------------\nInput: {"feedback": "You are still on the streets. To the north is a restraunt where the mayor ate often. To the east is the Mayor\'s home.", "valid_actions": [\'west\', \'east\', \'north\', \'put paper down\']}\nOutput: east\n------------\n'
                  'The past 100 lines of the game play are as following to avoid taking the same step again that led to game ending:\n\n' + "\n".join(lines) + "\n\n"
                  '------------\nInput: {"feedback": ' + f'"{game_state.feedback}",'  + "valid_actions:" + str(game_state.valid_actions) + "}\nOutput: "
        )
        # prompt = "\n".join(lines[-100:])
        action = llm(prompt, stop=["\t"])

        if (action.startswith(">")):
            action = action[1:].strip()

        if action not in game_state.valid_actions:
            log.warning(f'Invalid action "{action}" received. Choosing a random valid action.')
            action = self.rng.choice(game_state.valid_actions)
        
        self.prompt += f'{game_state.feedback}\n> {action}\n'
        log.info(f'{game_state.feedback}\n> {action}\n')

        return action[0:100]