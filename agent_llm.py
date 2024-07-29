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
    max_tokens=100,
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
        prompt = (f"Reply ONLY with the ONE action selected from the list: [{', '.join(game_state.valid_actions)}]. "
                  f"If the game is stuck in a loop or not recognizing instruction reply with a random action from the previous list without giving a verbose explanation. "
                  f"We don't want anything in the output other than the next step. The context is as following:\n{self.prompt}\n{game_state.feedback}"
        )
        lines = prompt.splitlines()
        prompt = "\n".join(lines[-100:])
        action = llm(prompt, stop=["\t"])

        if (action.startswith(">")):
            action = action[1:].strip()

        if action not in game_state.valid_actions:
            log.warning(f'Invalid action "{action}" received. Choosing a random valid action.')
            action = self.rng.choice(game_state.valid_actions)
        
        self.prompt += f'{game_state.feedback}\n> {action}\n'
        log.info(f'{game_state.feedback}\n> {action}\n')

        return action[0:100]