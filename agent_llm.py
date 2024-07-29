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
        action = llm(f"Reply only with the best action from the list ({', '.join(game_state.valid_actions)}) given the following context:\n{self.prompt}\n{game_state.feedback}", stop=["\n"])[:100]
        if (action.startswith(">")):
            action = action[1:].strip()
        
        self.prompt += f'{game_state.feedback}\n> {action}\n'
        log.info(f'{game_state.feedback}\n> {action}\n')

        return action